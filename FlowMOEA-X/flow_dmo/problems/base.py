"""Vectorised DynamicProblem base class with GPU/CPU true-PF caching (Section 5)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from flow_dmo.core.dominance import nondominated_mask

Tensor = torch.Tensor


class DynamicProblem(ABC):
    """Base class for all dynamic multi-objective test problems.

    Subclasses implement ``evaluate_batch(X_real, t)`` without any Python loops
    over batch/population dimensions, and ``_compute_true_pf(t, n_points, device, dtype)``.
    """

    name: str = "DynamicProblem"

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        xl: Tensor | list[float],
        xu: Tensor | list[float],
        tau_t: int = 10,
        n_t: int = 10,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.tau_t = int(tau_t)
        self.n_t = int(n_t)
        self.device = torch.device(device)
        self.dtype = dtype

        xl_t = torch.as_tensor(xl, device=self.device, dtype=self.dtype).reshape(1, 1, self.n_var)
        xu_t = torch.as_tensor(xu, device=self.device, dtype=self.dtype).reshape(1, 1, self.n_var)
        if torch.any(xu_t <= xl_t):
            raise ValueError("upper bounds must strictly exceed lower bounds")
        self.xl = xl_t
        self.xu = xu_t
        self._pf_cache: dict[tuple[str, int, int, float, str, torch.dtype], Tensor] = {}

    def time_from_step(self, env_step: int) -> float:
        """Standard discrete time parameter ``t = env_step / n_t``."""
        return float(env_step) / float(self.n_t)

    def time_from_generation(self, gen: int) -> float:
        """Standard discrete time parameter ``t = (1 / n_t) * floor(gen / tau_t)``."""
        return float(gen // self.tau_t) / float(self.n_t)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> DynamicProblem:
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self.xl = self.xl.to(device=self.device, dtype=self.dtype)
        self.xu = self.xu.to(device=self.device, dtype=self.dtype)
        return self

    @abstractmethod
    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        """Evaluate ``X_real`` of shape ``(B, N, D)`` at time ``t`` -> ``(B, N, M)``."""

    def get_true_ps(self, t: float, n_points: int = 500, device=None, dtype=None) -> Tensor:
        """Return analytical Pareto-optimal decision vectors ``(P, D)`` in real bounds."""
        raise NotImplementedError(f"{self.name} does not define an analytical Pareto set")

    def _compute_true_pf(self, t: float, n_points: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Default PF builder: evaluate analytical PS and filter non-dominated points."""
        X_ps = self.get_true_ps(t, n_points=n_points, device=device, dtype=dtype)
        F = self.evaluate_batch(X_ps.unsqueeze(0), t)[0]
        return filter_nondominated(F)

    def get_true_pf(
        self,
        t: float,
        n_points: int = 500,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return cached true PF ``(P, M)`` at time ``t``."""
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        key = (self.name, self.n_obj, int(n_points), round(float(t), 6), str(dev), dt)
        if key not in self._pf_cache:
            self._pf_cache[key] = self._compute_true_pf(float(t), int(n_points), dev, dt)
        return self._pf_cache[key]


def filter_nondominated(F: Tensor) -> Tensor:
    """Return non-dominated rows of a 2-D objective tensor ``(P, M)``."""
    mask = nondominated_mask(F)
    return F[mask]


def grid_2d(n_points: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Deterministic 2-D grid on ``[0, 1]^2`` with ``~n_points`` points."""
    side = max(4, int(round(n_points ** 0.5)))
    u = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
    U1, U2 = torch.meshgrid(u, u, indexing="ij")
    return U1.reshape(-1), U2.reshape(-1)
