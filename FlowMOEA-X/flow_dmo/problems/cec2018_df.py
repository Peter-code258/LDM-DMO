"""Full CEC 2018 Dynamic Multi-Objective Competition suite: DF1 -- DF14 (Section 5.2).

All 14 benchmark problems are implemented in pure batched PyTorch tensors
(``X_real: (B, N, D) -> F: (B, N, M)``) with exact analytical Pareto sets
``get_true_ps`` and Pareto fronts ``get_true_pf``.
"""
from __future__ import annotations

import math

import torch

from flow_dmo.problems.base import DynamicProblem, filter_nondominated, grid_2d

Tensor = torch.Tensor


class DF1(DynamicProblem):
    name = "DF1"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        super().__init__(n_var, 2, [0.0] * n_var, [1.0] * n_var, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = abs(math.sin(0.5 * math.pi * t))
        H = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        g = 1.0 + ((X_real[..., 1:] - G) ** 2).sum(dim=-1)
        f1 = x1
        f2 = g * (1.0 - torch.pow(torch.clamp(x1 / g, min=0.0), H))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = abs(math.sin(0.5 * math.pi * t))
        X = torch.full((n_points, self.n_var), G, device=dev, dtype=dt)
        X[:, 0] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        H = 0.75 * math.sin(0.5 * math.pi * t) + 1.25
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        return torch.stack([u, 1.0 - torch.pow(u, H)], dim=-1)


class DF2(DynamicProblem):
    name = "DF2"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        super().__init__(n_var, 2, [0.0] * n_var, [1.0] * n_var, tau_t, n_t, **kw)

    def _r(self, t: float) -> int:
        G = abs(math.sin(0.5 * math.pi * t))
        return min(self.n_var - 1, int(math.floor((self.n_var - 1) * G)))

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = abs(math.sin(0.5 * math.pi * t))
        r = self._r(t)
        f1 = torch.clamp(X_real[..., r], 0.0, 1.0)
        sq = (X_real - G) ** 2
        g = 1.0 + sq.sum(dim=-1) - sq[..., r]
        f2 = g * (1.0 - torch.sqrt(torch.clamp(f1 / g, min=0.0)))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = abs(math.sin(0.5 * math.pi * t))
        r = self._r(t)
        X = torch.full((n_points, self.n_var), G, device=dev, dtype=dt)
        X[:, r] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        return torch.stack([u, 1.0 - torch.sqrt(u)], dim=-1)


class DF3(DynamicProblem):
    name = "DF3"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0] + [-1.0] * (n_var - 1)
        xu = [1.0] + [2.0] * (n_var - 1)
        super().__init__(n_var, 2, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = math.sin(0.5 * math.pi * t)
        H = 1.5 + G
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        target = G + torch.pow(x1, H).unsqueeze(-1)
        g = 1.0 + ((X_real[..., 1:] - target) ** 2).sum(dim=-1)
        f1 = x1
        f2 = g * (1.0 - torch.pow(torch.clamp(x1 / g, min=0.0), H))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = math.sin(0.5 * math.pi * t)
        H = 1.5 + G
        u = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        X = torch.empty((n_points, self.n_var), device=dev, dtype=dt)
        X[:, 0] = u
        X[:, 1:] = (G + torch.pow(u, H)).unsqueeze(1)
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        H = 1.5 + math.sin(0.5 * math.pi * t)
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        return torch.stack([u, 1.0 - torch.pow(u, H)], dim=-1)


class DF4(DynamicProblem):
    name = "DF4"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        super().__init__(n_var, 2, [-2.0] * n_var, [2.0] * n_var, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        a = math.sin(0.5 * math.pi * t)
        b = 1.0 + abs(math.cos(0.5 * math.pi * t))
        H = 1.5 + a
        x1 = X_real[..., 0]
        idx = torch.arange(2, self.n_var + 1, device=X_real.device, dtype=X_real.dtype)
        target = a * ((x1.unsqueeze(-1) / idx) ** 2)
        g = 1.0 + ((X_real[..., 1:] - target) ** 2).sum(dim=-1)
        f1 = g * torch.pow(torch.abs(x1 - a), H)
        f2 = g * torch.pow(torch.abs(x1 - a - b), H)
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        a = math.sin(0.5 * math.pi * t)
        b = 1.0 + abs(math.cos(0.5 * math.pi * t))
        lo, hi = max(-2.0, a), min(2.0, a + b)
        x1 = torch.linspace(lo, hi, n_points, device=dev, dtype=dt)
        idx = torch.arange(2, self.n_var + 1, device=dev, dtype=dt)
        X = torch.empty((n_points, self.n_var), device=dev, dtype=dt)
        X[:, 0] = x1
        X[:, 1:] = torch.clamp(a * ((x1.unsqueeze(1) / idx) ** 2), -2.0, 2.0)
        return X


class DF5(DynamicProblem):
    name = "DF5"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0] + [-1.0] * (n_var - 1)
        xu = [1.0] * n_var
        super().__init__(n_var, 2, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = math.sin(0.5 * math.pi * t)
        w = math.floor(10.0 * G)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        g = 1.0 + ((X_real[..., 1:] - G) ** 2).sum(dim=-1)
        ripple = 0.02 * torch.sin(w * math.pi * x1)
        f1 = g * (x1 + ripple)
        f2 = g * (1.0 - x1 + ripple)
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = math.sin(0.5 * math.pi * t)
        X = torch.full((n_points, self.n_var), G, device=dev, dtype=dt)
        X[:, 0] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        return X


class DF6(DynamicProblem):
    name = "DF6"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0] + [-1.0] * (n_var - 1)
        xu = [1.0] * n_var
        super().__init__(n_var, 2, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = math.sin(0.5 * math.pi * t)
        a = 0.2 + 2.8 * abs(G)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        y = X_real[..., 1:] - G
        g = 1.0 + (abs(G) * (y ** 2) - 10.0 * torch.cos(2.0 * math.pi * y) + 10.0).sum(dim=-1)
        wave = 0.1 * torch.sin(3.0 * math.pi * x1)
        f1 = g * torch.pow(torch.clamp(x1 + wave, min=0.0), a)
        f2 = g * torch.pow(torch.clamp(1.0 - x1 + wave, min=0.0), a)
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = math.sin(0.5 * math.pi * t)
        X = torch.full((n_points, self.n_var), G, device=dev, dtype=dt)
        X[:, 0] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        return X


class DF7(DynamicProblem):
    name = "DF7"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [1.0] + [0.0] * (n_var - 1)
        xu = [4.0] + [1.0] * (n_var - 1)
        super().__init__(n_var, 2, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        a = 5.0 * math.cos(0.5 * math.pi * t)
        x1 = torch.clamp(X_real[..., 0], 1.0, 4.0)
        target = torch.sigmoid(-a * (x1 - 2.5)).unsqueeze(-1)
        g = 1.0 + ((X_real[..., 1:] - target) ** 2).sum(dim=-1)
        f1 = g * (1.0 + t) / x1
        f2 = g * x1 / (1.0 + t)
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        a = 5.0 * math.cos(0.5 * math.pi * t)
        x1 = torch.linspace(1.0, 4.0, n_points, device=dev, dtype=dt)
        X = torch.empty((n_points, self.n_var), device=dev, dtype=dt)
        X[:, 0] = x1
        X[:, 1:] = torch.sigmoid(-a * (x1 - 2.5)).unsqueeze(1)
        return X


class DF8(DynamicProblem):
    name = "DF8"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0] + [-1.0] * (n_var - 1)
        xu = [1.0] * n_var
        super().__init__(n_var, 2, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = math.sin(0.5 * math.pi * t)
        alpha = 2.25 + 2.0 * math.cos(2.0 * math.pi * t)
        beta = 1.0
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        A = G * torch.sin(4.0 * math.pi * torch.pow(x1, beta)) / (1.0 + abs(G))
        g = 1.0 + ((X_real[..., 1:] - A.unsqueeze(-1)) ** 2).sum(dim=-1)
        wave = 0.1 * torch.sin(3.0 * math.pi * x1)
        f1 = g * (x1 + wave)
        f2 = g * torch.pow(torch.clamp(1.0 - x1 + wave, min=0.0), alpha)
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = math.sin(0.5 * math.pi * t)
        x1 = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        A = G * torch.sin(4.0 * math.pi * x1) / (1.0 + abs(G))
        X = torch.empty((n_points, self.n_var), device=dev, dtype=dt)
        X[:, 0] = x1
        X[:, 1:] = A.unsqueeze(1)
        return X


class DF9(DynamicProblem):
    name = "DF9"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0] + [-1.0] * (n_var - 1)
        xu = [1.0] * n_var
        super().__init__(n_var, 2, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        N_t = 1.0 + math.floor(10.0 * abs(math.sin(0.5 * math.pi * t)))
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        prev = X_real[..., :-1]
        target = torch.cos(4.0 * t + x1.unsqueeze(-1) + prev)
        g = 1.0 + ((X_real[..., 1:] - target) ** 2).sum(dim=-1)
        jump = torch.clamp((1.0 / (2.0 * N_t) + 0.1) * torch.sin(2.0 * N_t * math.pi * x1), min=0.0)
        f1 = g * (x1 + jump)
        f2 = g * (1.0 - x1 + jump)
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        X = torch.empty((n_points, self.n_var), device=dev, dtype=dt)
        x1 = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        X[:, 0] = x1
        for i in range(1, self.n_var):
            X[:, i] = torch.cos(4.0 * t + x1 + X[:, i - 1])
        return X


class DF10(DynamicProblem):
    name = "DF10"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0, 0.0] + [-1.0] * (n_var - 2)
        xu = [1.0] * n_var
        super().__init__(n_var, 3, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = math.sin(0.5 * math.pi * t)
        H = 2.25 + 2.0 * math.cos(0.5 * math.pi * t)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        x2 = torch.clamp(X_real[..., 1], 0.0, 1.0)
        target = torch.sin(2.0 * math.pi * (x1 + x2)) / (1.0 + abs(G))
        g = 1.0 + ((X_real[..., 2:] - target.unsqueeze(-1)) ** 2).sum(dim=-1)
        s1 = torch.pow(torch.clamp(torch.sin(0.5 * math.pi * x1), min=0.0), H)
        c1 = torch.pow(torch.clamp(torch.cos(0.5 * math.pi * x1), min=0.0), H)
        s2 = torch.pow(torch.clamp(torch.sin(0.5 * math.pi * x2), min=0.0), H)
        c2 = torch.pow(torch.clamp(torch.cos(0.5 * math.pi * x2), min=0.0), H)
        return torch.stack([g * s1, g * s2 * c1, g * c2 * c1], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = math.sin(0.5 * math.pi * t)
        u1, u2 = grid_2d(n_points, dev, dt)
        X = torch.empty((u1.shape[0], self.n_var), device=dev, dtype=dt)
        X[:, 0] = u1
        X[:, 1] = u2
        X[:, 2:] = (torch.sin(2.0 * math.pi * (u1 + u2)) / (1.0 + abs(G))).unsqueeze(1)
        return X


class DF11(DynamicProblem):
    name = "DF11"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        super().__init__(n_var, 3, [0.0] * n_var, [1.0] * n_var, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = abs(math.sin(0.5 * math.pi * t))
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        x2 = torch.clamp(X_real[..., 1], 0.0, 1.0)
        y1 = math.pi * G / 6.0 + (0.5 * math.pi - math.pi * G / 3.0) * x1
        y2 = math.pi * G / 6.0 + (0.5 * math.pi - math.pi * G / 3.0) * x2
        g = 1.0 + G + ((X_real[..., 2:] - (0.5 * G * x1).unsqueeze(-1)) ** 2).sum(dim=-1)
        return torch.stack([g * torch.sin(y1), g * torch.sin(y2) * torch.cos(y1),
                            g * torch.cos(y2) * torch.cos(y1)], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = abs(math.sin(0.5 * math.pi * t))
        u1, u2 = grid_2d(n_points, dev, dt)
        X = torch.empty((u1.shape[0], self.n_var), device=dev, dtype=dt)
        X[:, 0] = u1
        X[:, 1] = u2
        X[:, 2:] = (0.5 * G * u1).unsqueeze(1)
        return X


class DF12(DynamicProblem):
    name = "DF12"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0, 0.0] + [-1.0] * (n_var - 2)
        xu = [1.0] * n_var
        super().__init__(n_var, 3, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        k_t = 10.0 * math.sin(math.pi * t)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        x2 = torch.clamp(X_real[..., 1], 0.0, 1.0)
        hole = torch.abs(
            torch.sin(torch.floor(k_t * (2.0 * x1 - 1.0)) * 0.5 * math.pi)
            * torch.sin(torch.floor(k_t * (2.0 * x2 - 1.0)) * 0.5 * math.pi)
        )
        g = 1.0 + ((X_real[..., 2:] - torch.sin(t * x1).unsqueeze(-1)) ** 2).sum(dim=-1) + hole
        f1 = g * torch.cos(0.5 * math.pi * x2) * torch.cos(0.5 * math.pi * x1)
        f2 = g * torch.sin(0.5 * math.pi * x2) * torch.cos(0.5 * math.pi * x1)
        f3 = g * torch.sin(0.5 * math.pi * x1)
        return torch.stack([f1, f2, f3], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        u1, u2 = grid_2d(n_points, dev, dt)
        X = torch.empty((u1.shape[0], self.n_var), device=dev, dtype=dt)
        X[:, 0] = u1
        X[:, 1] = u2
        X[:, 2:] = torch.sin(t * u1).unsqueeze(1)
        F = self.evaluate_batch(X.unsqueeze(0), t)[0]
        from flow_dmo.core.dominance import nondominated_mask
        return X[nondominated_mask(F)]


class DF13(DynamicProblem):
    name = "DF13"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0, 0.0] + [-1.0] * (n_var - 2)
        xu = [1.0] * n_var
        super().__init__(n_var, 3, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = math.sin(0.5 * math.pi * t)
        p_t = math.floor(6.0 * G)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        x2 = torch.clamp(X_real[..., 1], 0.0, 1.0)
        g = 1.0 + ((X_real[..., 2:] - G) ** 2).sum(dim=-1)
        f1 = g * (torch.cos(0.5 * math.pi * x1) ** 2)
        f2 = g * (torch.cos(0.5 * math.pi * x2) ** 2)
        s1 = torch.sin(0.5 * math.pi * x1)
        s2 = torch.sin(0.5 * math.pi * x2)
        f3 = g * (
            s1 ** 2 + s1 * (torch.cos(p_t * math.pi * x1) ** 2)
            + s2 ** 2 + s2 * (torch.cos(p_t * math.pi * x2) ** 2)
        )
        return torch.stack([f1, f2, f3], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = math.sin(0.5 * math.pi * t)
        u1, u2 = grid_2d(n_points, dev, dt)
        X = torch.full((u1.shape[0], self.n_var), G, device=dev, dtype=dt)
        X[:, 0] = u1
        X[:, 1] = u2
        F = self.evaluate_batch(X.unsqueeze(0), t)[0]
        from flow_dmo.core.dominance import nondominated_mask
        return X[nondominated_mask(F)]


class DF14(DynamicProblem):
    """Degenerate PF: collapses to a 1-D curve when ``G(t) == 0``."""

    name = "DF14"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        xl = [0.0, 0.0] + [-1.0] * (n_var - 2)
        xu = [1.0] * n_var
        super().__init__(n_var, 3, xl, xu, tau_t, n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = math.sin(0.5 * math.pi * t)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        x2 = torch.clamp(X_real[..., 1], 0.0, 1.0)
        y1 = 0.5 + G * (x1 - 0.5)
        g = 1.0 + ((X_real[..., 2:] - y1.unsqueeze(-1)) ** 2).sum(dim=-1)
        w1 = 0.05 * torch.sin(6.0 * math.pi * y1)
        w2 = 0.05 * torch.sin(6.0 * math.pi * x2)
        f1 = g * (1.0 - y1 + w1)
        f2 = g * (1.0 - x2 + w2) * (y1 + w1)
        f3 = g * (x2 + w2) * (y1 + w1)
        return torch.stack([f1, f2, f3], dim=-1)

    def get_true_ps(self, t, n_points=500, device=None, dtype=None):
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = math.sin(0.5 * math.pi * t)
        u1, u2 = grid_2d(n_points, dev, dt)
        y1 = 0.5 + G * (u1 - 0.5)
        X = torch.empty((u1.shape[0], self.n_var), device=dev, dtype=dt)
        X[:, 0] = u1
        X[:, 1] = u2
        X[:, 2:] = y1.unsqueeze(1)
        F = self.evaluate_batch(X.unsqueeze(0), t)[0]
        from flow_dmo.core.dominance import nondominated_mask
        return X[nondominated_mask(F)]


DF_PROBLEMS = {
    f"DF{i}": globals()[f"DF{i}"] for i in range(1, 15)
}
