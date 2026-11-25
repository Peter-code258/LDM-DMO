"""Population-embedded, probe-free environment change detector (Section 4.1).

Samples ``N_sensor = max(5, int(0.1 * N))`` quantile-spaced elites from the
current non-dominated front, re-evaluates them through ``FELedger`` and
computes the normalised objective drift intensity ``D_k``. Because sensors are
drawn directly from the elite population, their re-evaluated pairs
``(Z_sensor, F_sensor_new)`` are reused in the warm-start population with zero
wasted evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from flow_dmo.core.dominance import batched_gather, batched_pareto_ranks
from flow_dmo.core.fe_ledger import FELedger

Tensor = torch.Tensor


@dataclass
class ChangeDetectionResult:
    changed: Tensor            # (B,) bool
    D_k: Tensor                # (B,) float
    sensor_Z: Tensor           # (B, N_sensor, D)
    sensor_F_old: Tensor       # (B, N_sensor, M)
    sensor_F_new: Tensor       # (B, N_sensor, M)


class PopulationChangeDetector:
    """Zero-waste embedded sensor detector."""

    def __init__(self, sensor_ratio: float = 0.1, min_sensors: int = 5, eps_change: float = 1e-4):
        self.sensor_ratio = float(sensor_ratio)
        self.min_sensors = int(min_sensors)
        self.eps_change = float(eps_change)

    def n_sensors(self, pop_size: int) -> int:
        return min(pop_size, max(self.min_sensors, int(self.sensor_ratio * pop_size)))

    def select_sensors(self, Z_pop: Tensor, F_pop: Tensor) -> tuple[Tensor, Tensor]:
        """Pick ``N_sensor`` quantile-spaced elites along the front."""
        B, N, _ = Z_pop.shape
        n_s = self.n_sensors(N)
        ranks = batched_pareto_ranks(F_pop)
        # Sort by (rank, f1) so quantile picks span the first front evenly
        f1 = F_pop[..., 0]
        f1_min = f1.min(dim=1, keepdim=True).values
        f1_span = torch.clamp(f1.max(dim=1, keepdim=True).values - f1_min, min=1e-12)
        f1_norm = (f1 - f1_min) / f1_span
        key = ranks.to(torch.float64) * 2.0 + f1_norm.to(torch.float64)
        order = torch.argsort(key, dim=1, stable=True)

        # Spread picks across rank-0 size (falling back to top-N if front is small)
        front0_counts = (ranks == 0).sum(dim=1).clamp(min=n_s, max=N)      # (B,)
        q = torch.linspace(0.0, 1.0, n_s, device=Z_pop.device, dtype=torch.float64).unsqueeze(0)
        pos = torch.round(q * (front0_counts.unsqueeze(1).to(torch.float64) - 1.0)).long()
        idx = torch.gather(order, 1, pos)
        return batched_gather(Z_pop, idx), batched_gather(F_pop, idx)

    def detect(
        self,
        Z_pop: Tensor,
        F_pop: Tensor,
        problem,
        t_new: float,
        ledger: FELedger,
    ) -> ChangeDetectionResult:
        """Sample sensors, charge real FEs, and compute normalised drift ``D_k``."""
        Z_s, F_old = self.select_sensors(Z_pop, F_pop)
        F_new = ledger.evaluate_and_charge(problem, Z_s, t_new, role="sensor")

        f_min = F_pop.min(dim=1).values
        f_max = F_pop.max(dim=1).values
        span = torch.linalg.norm(f_max - f_min, dim=-1).clamp(min=1e-6)    # (B,)
        diff_norm = torch.linalg.norm(F_new - F_old, dim=-1)               # (B, N_sensor)
        # Bounded relative drift in [0, 1) so steep g-landscapes (e.g. dMOP2) don't false-trip circuit breaker
        D_k = (diff_norm / (span.unsqueeze(1) + diff_norm)).mean(dim=1)    # (B,)
        changed = D_k > self.eps_change
        return ChangeDetectionResult(
            changed=changed,
            D_k=D_k,
            sensor_Z=Z_s,
            sensor_F_old=F_old,
            sensor_F_new=F_new,
        )
