"""Abstract base class for fully-tensorised dynamic MOEAs (Section 7)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from flow_dmo.core.dominance import batched_crowding_distance, batched_gather, batched_pareto_ranks, nsga2_sort_key
from flow_dmo.core.fe_ledger import FELedger
from flow_dmo.core.tensor_ops import orthogonal_lhs
from flow_dmo.problems.base import DynamicProblem

Tensor = torch.Tensor


class BatchedDynamicMOEA(ABC):
    """Unified interface for dynamic environment response algorithms."""

    name: str = "BaseDMOEA"

    def __init__(self, pop_size: int, n_var: int, n_obj: int, device: torch.device | str = "cpu"):
        self.pop_size = int(pop_size)
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.device = torch.device(device)
        self.history_elites: list[tuple[Tensor, Tensor]] = []

    def reset(self) -> None:
        self.history_elites.clear()

    @staticmethod
    def extract_elite_moments(Z: Tensor, F: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute mean and std ``(B, D)`` of rank-0 elites (with safe fallback)."""
        ranks = batched_pareto_ranks(F)
        mask = ranks == 0
        # Ensure at least 2 individuals per seed for valid std
        counts = mask.sum(dim=1)
        if bool((counts < 2).any()):
            top2 = torch.argsort(ranks, dim=1, stable=True)[:, :2]
            fallback = torch.zeros_like(mask)
            fallback.scatter_(1, top2, True)
            mask = torch.where((counts < 2).unsqueeze(1), fallback, mask)

        w = mask.to(Z.dtype).unsqueeze(-1)                                 # (B, N, 1)
        denom = w.sum(dim=1).clamp(min=1.0)                                # (B, 1)
        mu = (Z * w).sum(dim=1) / denom                                    # (B, D)
        var = (((Z - mu.unsqueeze(1)) ** 2) * w).sum(dim=1) / (denom - 1.0).clamp(min=1.0)
        sigma = torch.sqrt(var.clamp(min=1e-6))
        return mu, sigma, mask

    @staticmethod
    def select_diverse_elites(Z: Tensor, F: Tensor, count: int) -> Tensor:
        """Select ``count`` diverse elites by (rank, -crowding_distance)."""
        B, _, D = Z.shape
        if count <= 0:
            return torch.empty((B, 0, D), device=Z.device, dtype=Z.dtype)
        ranks = batched_pareto_ranks(F)
        cd = batched_crowding_distance(F, ranks)
        order = torch.argsort(nsga2_sort_key(ranks, cd), dim=1, stable=True)
        idx = order[:, :count]
        return batched_gather(Z, idx)

    @staticmethod
    def sample_orthogonal_lhs(batch_size: int, count: int, dim: int, device, dtype,
                              generator: torch.Generator | None) -> Tensor:
        return orthogonal_lhs(batch_size, count, dim, device=device, dtype=dtype, generator=generator)

    @abstractmethod
    def respond_batch(
        self,
        prev_Z: Tensor,
        prev_F: Tensor,
        sensor_Z: Tensor,
        sensor_F_new: Tensor,
        D_k: Tensor,
        problem: DynamicProblem,
        t_new: float,
        ledger: FELedger,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """Generate and evaluate the warm-start population ``(Z_warm, F_warm)`` of size ``N``.

        Must reuse ``(sensor_Z, sensor_F_new)`` and evaluate exactly ``N - N_sensor``
        additional individuals via ``ledger.evaluate_and_charge(..., role='warmstart')``.
        """
