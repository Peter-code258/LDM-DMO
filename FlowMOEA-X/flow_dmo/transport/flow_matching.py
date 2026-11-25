"""Ultra-lightweight Time-Conditioned Flow Matching ODE module (Section 4.4).

Used in ``FlowMOEA-X (Pro)`` mode for strongly nonlinear / coupled manifolds:
  * ``ConditionalVelocityNet``: ``Linear(3D + 2, 64) -> SiLU -> Linear(64, 64) -> SiLU -> Linear(64, D)``
    (~6,000 parameters, fits in L1/L2 cache).
  * Online CFM fine-tuning: 8 AdamW steps (``lr = 1e-2``) on paired elites
    ``(z_0, z_1) = (Z_prev_aligned, Z_curr)``.
  * Two-step midpoint ODE solver (``dtau = 0.5``) with reflective clamping.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from flow_dmo.core.dominance import batched_gather
from flow_dmo.core.tensor_ops import reflect_clamp
from flow_dmo.transport.sliced_ot import _extract_elites, _sinkhorn_match

Tensor = torch.Tensor


class ConditionalVelocityNet(nn.Module):
    """MLP velocity field ``v_theta(z, tau, e_k)`` with input dim ``3D + 2``."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.dim = int(dim)
        in_dim = 3 * self.dim + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.dim),
        )
        # Near-zero init on output head so initial flow starts close to identity
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-2)

    def forward(self, z: Tensor, tau: Tensor, cond: Tensor) -> Tensor:
        """``z: (B, P, D)``, ``tau: (B, P, 1)``, ``cond: (B, P, 2D + 1)`` -> ``(B, P, D)``."""
        x = torch.cat([z, tau, cond], dim=-1)
        return self.net(x)


class FlowMatchingTransport:
    """Online Conditional Flow Matching (CFM) push-forward with Midpoint ODE."""

    def __init__(
        self,
        dim: int,
        hidden: int = 64,
        n_steps: int = 8,
        lr: float = 1e-2,
        tangent_noise: float = 0.08,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dim = int(dim)
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.tangent_noise = float(tangent_noise)
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = ConditionalVelocityNet(dim=self.dim, hidden=hidden).to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def _build_cond(self, delta_mu: Tensor, delta_sigma: Tensor, D_k: Tensor, n_pts: int) -> Tensor:
        # e_k = [delta_mu, delta_sigma, D_k] of shape (B, 2D + 1) -> broadcast to (B, n_pts, 2D + 1)
        e = torch.cat([delta_mu, delta_sigma, D_k.unsqueeze(-1)], dim=-1)
        return e.unsqueeze(1).expand(-1, n_pts, -1)

    def step_ode_pushforward(
        self,
        prev_elites: tuple[Tensor, Tensor],
        curr_elites: tuple[Tensor, Tensor],
        mu_hat: Tensor,
        sigma_hat: Tensor,
        count: int,
        D_k: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Fine-tune CFM on ``(prev_elites -> curr_elites)`` and integrate ODE from ``curr_elites``."""
        Z_prev_all, F_prev_all = prev_elites
        Z_curr_all, F_curr_all = curr_elites
        B, N_curr, D = Z_curr_all.shape
        dev, dt = Z_curr_all.device, Z_curr_all.dtype
        if count <= 0:
            return torch.empty((B, 0, D), device=dev, dtype=dt)

        if self.model.net[0].weight.device != dev or self.model.net[0].weight.dtype != dt:
            self.model.to(device=dev, dtype=dt)

        n_elite = min(N_curr, Z_prev_all.shape[1])
        Z_curr, F_curr = _extract_elites(Z_curr_all, F_curr_all, n_elite)
        Z_prev, F_prev = _extract_elites(Z_prev_all, F_prev_all, n_elite)

        # Pair prev and curr elites via objective-space Sinkhorn
        with torch.no_grad():
            P = _sinkhorn_match(F_curr, F_prev)
            z0 = torch.bmm(P, Z_prev)
            z1 = Z_curr
            v_target = z1 - z0
            d_mu_train = z1.mean(dim=1) - z0.mean(dim=1)
            d_sig_train = z1.std(dim=1) - z0.std(dim=1)
            dk_vec = torch.zeros(B, device=dev, dtype=dt) if D_k is None else D_k.to(dev, dt)
            cond_train = self._build_cond(d_mu_train, d_sig_train, dk_vec, n_elite)

        # 8-step online CFM fine-tuning
        self.model.train()
        for _ in range(self.n_steps):
            tau = torch.rand((B, n_elite, 1), device=dev, dtype=dt, generator=generator)
            z_tau = (1.0 - tau) * z0 + tau * z1
            v_pred = self.model(z_tau, tau, cond_train)
            loss = ((v_pred - v_target) ** 2).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()

        # 2-step Midpoint ODE integration from Z_curr to Z_next
        self.model.eval()
        with torch.no_grad():
            if count == n_elite:
                Z_start = Z_curr
            else:
                pick = torch.randint(0, n_elite, (B, count), device=dev, generator=generator)
                Z_start = batched_gather(Z_curr, pick)

            d_mu_next = mu_hat - Z_curr.mean(dim=1)
            d_sig_next = sigma_hat - Z_curr.std(dim=1)
            cond_next = self._build_cond(d_mu_next, d_sig_next, dk_vec, count)

            tau0 = torch.zeros((B, count, 1), device=dev, dtype=dt)
            tau_half = torch.full((B, count, 1), 0.5, device=dev, dtype=dt)
            v0 = self.model(Z_start, tau0, cond_next)
            z_half = Z_start + 0.5 * v0
            v_half = self.model(z_half, tau_half, cond_next)
            Z_next = Z_start + v_half

            noise = torch.randn((B, count, D), device=dev, dtype=dt, generator=generator)
            Z_next = Z_next + noise * (self.tangent_noise * sigma_hat.unsqueeze(1))
            return reflect_clamp(Z_next)
