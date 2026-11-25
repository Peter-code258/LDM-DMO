"""Authoritative dMOP suite (Goh & Tan 2009, IEEE TEVC) covering Type I/II/III
PS/PF dynamic combinations (Section 5.1)."""
from __future__ import annotations

import math

import torch

from flow_dmo.problems.base import DynamicProblem

Tensor = torch.Tensor


def _dmop_H(t: float) -> float:
    return 0.75 * math.sin(0.5 * math.pi * t) + 1.25


class dMOP1(DynamicProblem):
    """Type III: static PS ``x_{2:D} = 0``, time-varying PF ``f2 = 1 - f1^{H(t)}``."""

    name = "dMOP1"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        super().__init__(n_var=n_var, n_obj=2, xl=[0.0] * n_var, xu=[1.0] * n_var,
                         tau_t=tau_t, n_t=n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        H = _dmop_H(t)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        g = 1.0 + 9.0 * (X_real[..., 1:] ** 2).sum(dim=-1)
        f1 = x1
        f2 = g * (1.0 - torch.pow(torch.clamp(f1 / g, min=0.0), H))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t: float, n_points: int = 500, device=None, dtype=None) -> Tensor:
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        X = torch.zeros((n_points, self.n_var), device=dev, dtype=dt)
        X[:, 0] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        H = _dmop_H(t)
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        return torch.stack([u, 1.0 - torch.pow(u, H)], dim=-1)


class dMOP2(DynamicProblem):
    """Type II: moving PS ``x_{2:D} = G(t)``, moving PF ``f2 = 1 - f1^{H(t)}``."""

    name = "dMOP2"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        super().__init__(n_var=n_var, n_obj=2, xl=[0.0] * n_var, xu=[1.0] * n_var,
                         tau_t=tau_t, n_t=n_t, **kw)

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = abs(math.sin(0.5 * math.pi * t))
        H = _dmop_H(t)
        x1 = torch.clamp(X_real[..., 0], 0.0, 1.0)
        g = 1.0 + 9.0 * ((X_real[..., 1:] - G) ** 2).sum(dim=-1)
        f1 = x1
        f2 = g * (1.0 - torch.pow(torch.clamp(f1 / g, min=0.0), H))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t: float, n_points: int = 500, device=None, dtype=None) -> Tensor:
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = abs(math.sin(0.5 * math.pi * t))
        X = torch.full((n_points, self.n_var), G, device=dev, dtype=dt)
        X[:, 0] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        H = _dmop_H(t)
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        return torch.stack([u, 1.0 - torch.pow(u, H)], dim=-1)


class dMOP3(DynamicProblem):
    """Type I: moving PS with time-varying position index ``r(t)``, static PF ``1 - sqrt(f1)``."""

    name = "dMOP3"

    def __init__(self, n_var: int = 10, tau_t: int = 10, n_t: int = 10, **kw):
        super().__init__(n_var=n_var, n_obj=2, xl=[0.0] * n_var, xu=[1.0] * n_var,
                         tau_t=tau_t, n_t=n_t, **kw)

    def _pos_index(self, t: float) -> int:
        return int(round(abs(t) * self.n_t)) % self.n_var

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        G = abs(math.sin(0.5 * math.pi * t))
        r = self._pos_index(t)
        f1 = torch.clamp(X_real[..., r], 0.0, 1.0)
        sq = (X_real - G) ** 2
        g = 1.0 + 9.0 * (sq.sum(dim=-1) - sq[..., r])
        f2 = g * (1.0 - torch.sqrt(torch.clamp(f1 / g, min=0.0)))
        return torch.stack([f1, f2], dim=-1)

    def get_true_ps(self, t: float, n_points: int = 500, device=None, dtype=None) -> Tensor:
        dev = self.device if device is None else torch.device(device)
        dt = self.dtype if dtype is None else dtype
        G = abs(math.sin(0.5 * math.pi * t))
        r = self._pos_index(t)
        X = torch.full((n_points, self.n_var), G, device=dev, dtype=dt)
        X[:, r] = torch.linspace(0.0, 1.0, n_points, device=dev, dtype=dt)
        return X

    def _compute_true_pf(self, t, n_points, device, dtype):
        u = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        return torch.stack([u, 1.0 - torch.sqrt(u)], dim=-1)
