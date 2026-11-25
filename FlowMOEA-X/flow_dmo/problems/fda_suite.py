"""Classic FDA dynamic multi-objective suite: FDA1, FDA2, FDA3, FDA4 (Section 5.1)."""
from __future__ import annotations

import math

import torch

from flow_dmo.problems.base import DynamicProblem

Tensor = torch.Tensor


class FDA1(DynamicProblem):
    """Type I: moving PS, static convex PF ``f2 = 1 - sqrt(f1)``."""

    name = "FDA1"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0] + [-1.0] * (n_var - 1)
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, tau_t=tau_t, n_t=n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = math.sin(0.5 * math.pi * t)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        g = 1.0 + ((X_real[..., 1:] - G) ** 2).sum(dim=-1)
        f1 = x1
        f2 = g * (1.0 - torch.sqrt(torch.clamp(f1 / g, min=0.0)))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t: float, n_points: int = 500, device=None, dtype=None) -> Tensor:
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = math.sin(0.5 * math.pi * t)
        X = torch.full((n_points, self.n_var), G, device=dev, dtype=dt)
        X[:, 0] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        return torch.stack([u, 1.0 - torch.sqrt(u)], dim=-1)


class FDA2(DynamicProblem):
    """Modified FDA2 (well-posed convex-to-concave PF): moving PF & PS_III."""

    name = "FDA2"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0] + [-1.0] * (n_var - 1)
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, tau_t=tau_t, n_t=n_t, **kw)
        self._split = 1 + (self.n_var - 1) // 2

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        H = 0.75 + 0.25 * math.sin(0.5 * math.pi * t)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        x_ii = X_real[..., 1 : self._split]
        x_iii = X_real[..., self._split :]
        g = 1.0 + (x_ii ** 2).sum(dim=-1)
        H_eff = H + ((x_iii - H) ** 2).sum(dim=-1)
        f1 = x1
        f2 = g * (1.0 - torch.pow(torch.clamp(f1 / g, min=0.0), H_eff))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t: float, n_points: int = 500, device=None, dtype=None) -> Tensor:
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        H = 0.75 + 0.25 * math.sin(0.5 * math.pi * t)
        X = torch.zeros((n_points, self.n_var), device=dev, dtype=dt)
        X[:, 0] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        X[:, self._split :] = H
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        H = 0.75 + 0.25 * math.sin(0.5 * math.pi * t)
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        return torch.stack([u, 1.0 - torch.pow(u, H)], dim=-1)


class FDA3(DynamicProblem):
    """Type II: time-varying PS, PF shift and non-uniform density ``f1 = x1^{F(t)}``."""

    name = "FDA3"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0] + [-1.0] * (n_var - 1)
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, tau_t=tau_t, n_t=n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = abs(math.sin(0.5 * math.pi * t))
        F_exp = 10.0 ** (2.0 * math.sin(0.5 * math.pi * t))
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        f1 = torch.pow(x1, F_exp)
        g = 1.0 + G + ((X_real[..., 1:] - G) ** 2).sum(dim=-1)
        f2 = g * (1.0 - torch.sqrt(torch.clamp(f1 / g, min=0.0)))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t: float, n_points: int = 500, device=None, dtype=None) -> Tensor:
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = abs(math.sin(0.5 * math.pi * t))
        F_exp = 10.0 ** (2.0 * math.sin(0.5 * math.pi * t))
        u = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        X = torch.full((n_points, self.n_var), G, device=dev, dtype=dt)
        X[:, 0] = torch.pow(u, 1.0 / F_exp)
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        G = abs(math.sin(0.5 * math.pi * t))
        g_pf = 1.0 + G
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        f2 = g_pf * (1.0 - torch.sqrt(torch.clamp(u / g_pf, min=0.0)))
        return torch.stack([u, f2], dim=-1)


class FDA4(DynamicProblem):
    """Many-objective FDA4 (configurable ``M in {2, 3, 5}``, ``D = M + 9``).

    Matches PlatEMO ``FDA4.m``: position variables ``X[..., :M-1]``, distance
    variables ``X[..., M-1:]``. Static spherical PF, moving PS.
    """

    name = "FDA4"

    def __init__(self, n_obj: int = 3, n_var: int | None = None, tau_t: int = 10, n_t: int = 10, **kw):
        M = int(n_obj)
        D = int(n_var) if n_var is not None else M + 9
        if D <= M - 1:
            raise ValueError(f"FDA4 requires n_var >= n_obj (got D={D}, M={M})")
        super().__init__(n_var=D, n_obj=M, xl=[0.0] * D, xu=[1.0] * D, tau_t=tau_t, n_t=n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        M = self.n_obj
        G = abs(math.sin(0.5 * math.pi * t))
        g = ((X_real[..., M - 1 :] - G) ** 2).sum(dim=-1, keepdim=True)
        theta = torch.clamp(X_real[..., : M - 1], 0.0, 1.0) * (0.5 * math.pi)
        cos_th = torch.cos(theta)
        ones = torch.ones_like(g)
        cumprod = torch.cumprod(torch.cat([ones, cos_th], dim=-1), dim=-1)
        cumprod_flip = torch.flip(cumprod, dims=[-1])
        sin_rev = torch.sin(torch.flip(theta, dims=[-1]))
        sin_part = torch.cat([ones, sin_rev], dim=-1)
        return (1.0 + g) * cumprod_flip * sin_part

    def get_true_ps(self, t: float, n_points: int = 500, device=None, dtype=None) -> Tensor:
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        M = self.n_obj
        G = abs(math.sin(0.5 * math.pi * t))
        if M == 2:
            pos = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt).unsqueeze(1)
        elif M == 3:
            side = max(4, int(math.ceil(math.sqrt(n_points))))
            a = torch.linspace(0.0, 1.0, side, device=dev, dtype=dt)
            A, B = torch.meshgrid(a, a, indexing="ij")
            pos = torch.stack([A.reshape(-1), B.reshape(-1)], dim=-1)
        else:
            # Deterministic Sobol angles in [0, 1]^{M-1}
            eng = torch.quasirandom.SobolEngine(dimension=M - 1, scramble=True, seed=2026)
            pos = eng.draw(n_points).to(device=dev, dtype=dt)
        P = pos.shape[0]
        X = torch.full((P, self.n_var), G, device=dev, dtype=dt)
        X[:, : M - 1] = pos
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        X_ps = self.get_true_ps(t, n_points=n_points, device=device, dtype=dtype)
        return self.evaluate_batch(X_ps.unsqueeze(0), t)[0]
