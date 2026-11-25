"""Strict function-evaluation budget ledger (Section 3.3).

Every real objective evaluation -- environment-change sensing, warm-start
individuals and ordinary evolution offspring -- goes through
``FELedger.evaluate_and_charge``. No free external probe points exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from flow_dmo.core.tensor_ops import denormalize

Tensor = torch.Tensor
ROLES = ("sensor", "warmstart", "evolution")


class BudgetExceededError(RuntimeError):
    """Raised when a strict ledger would overspend the per-environment quota."""


@dataclass
class EnvAccount:
    env_index: int
    t: float
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(ROLES, 0))

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))


class FELedger:
    """Per-seed, per-environment, per-role accounting of real FEs.

    Attributes:
        total_fe_per_seed: ``(B,)`` int64 cumulative real FEs of every seed.
        env_fe_quota: ``tau_t * N`` real FEs allowed per environment.
    """

    def __init__(self, n_seeds: int, pop_size: int, tau_t: int, device: torch.device | str = "cpu",
                 strict_fe_budget: bool = True):
        self.n_seeds = int(n_seeds)
        self.pop_size = int(pop_size)
        self.tau_t = int(tau_t)
        self.device = torch.device(device)
        self.strict_fe_budget = bool(strict_fe_budget)
        self.env_fe_quota = self.tau_t * self.pop_size
        self.total_fe_per_seed = torch.zeros(self.n_seeds, dtype=torch.int64, device=self.device)
        self.accounts: list[EnvAccount] = []
        self.n_problem_calls = 0

    # ------------------------------------------------------------ environments
    def start_environment(self, env_index: int, t: float = 0.0) -> None:
        if self.strict_fe_budget and self.accounts:
            prev = self.accounts[-1]
            if prev.total > self.env_fe_quota:
                raise BudgetExceededError(
                    f"environment {prev.env_index} used {prev.total} > quota {self.env_fe_quota}")
        self.accounts.append(EnvAccount(env_index=int(env_index), t=float(t)))

    @property
    def current(self) -> EnvAccount:
        if not self.accounts:
            self.start_environment(0, 0.0)
        return self.accounts[-1]

    @property
    def env_fe_used(self) -> int:
        return self.current.total

    def remaining_fe_in_env(self) -> int:
        return self.env_fe_quota - self.env_fe_used

    def remaining_generations_in_env(self) -> int:
        """Whole generations of ``N`` real FEs still affordable in this environment.

        With ``strict_fe_budget=True`` the warm-start generation only evaluates
        ``N - N_sensor`` new individuals (the re-evaluated sensors fill the
        rest), so ``sensor + warmstart == N`` and the environment total is
        exactly ``tau_t * N``.
        """
        return max(0, self.remaining_fe_in_env()) // self.pop_size

    # ------------------------------------------------------------ charging
    def charge(self, n_per_seed: int, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        n = int(n_per_seed)
        if n < 0:
            raise ValueError("cannot charge a negative number of evaluations")
        acc = self.current
        if self.strict_fe_budget and acc.total + n > self.env_fe_quota:
            raise BudgetExceededError(
                f"charging {n} {role} FEs would exceed the environment quota "
                f"({acc.total} + {n} > {self.env_fe_quota})")
        acc.counts[role] += n
        self.total_fe_per_seed += n

    def evaluate_and_charge(self, problem, Z: Tensor, t: float, role: str) -> Tensor:
        """Denormalise ``Z (B, K, D)``, evaluate at time ``t`` and charge ``K`` FEs per seed."""
        if Z.dim() != 3:
            raise ValueError("Z must have shape (B, K, D)")
        B, K, _ = Z.shape
        if B != self.n_seeds:
            raise ValueError(f"ledger tracks {self.n_seeds} seeds, got batch {B}")
        if K == 0:
            return torch.empty((B, 0, problem.n_obj), device=Z.device, dtype=Z.dtype)
        self.charge(K, role)
        X_real = denormalize(Z, problem.xl.to(Z.device, Z.dtype), problem.xu.to(Z.device, Z.dtype))
        F = problem.evaluate_batch(X_real, t)
        self.n_problem_calls += 1
        return F.to(Z.dtype)

    # ------------------------------------------------------------ reporting
    def role_totals(self) -> dict[str, int]:
        out = dict.fromkeys(ROLES, 0)
        for acc in self.accounts:
            for r in ROLES:
                out[r] += acc.counts[r]
        return out

    def env_totals(self) -> list[int]:
        return [acc.total for acc in self.accounts]

    def all_envs_exact(self) -> bool:
        return all(total == self.env_fe_quota for total in self.env_totals())

    def summary(self) -> dict:
        roles = self.role_totals()
        return {
            "n_envs": len(self.accounts),
            "env_fe_quota": self.env_fe_quota,
            "sensor_fe": roles["sensor"],
            "warmstart_fe": roles["warmstart"],
            "evolution_fe": roles["evolution"],
            "total_fe_per_seed": int(self.total_fe_per_seed[0].item()) if self.n_seeds else 0,
            "all_envs_exact": self.all_envs_exact(),
        }
