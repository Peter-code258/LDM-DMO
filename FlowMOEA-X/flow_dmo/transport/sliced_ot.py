"""Reference-direction aligned Sliced Wasserstein Optimal Transport push-forward
(Section 4.3).

Replaces topology-destroying isotropic Gaussian resampling ``z ~ N(0, I)`` with:
  1. Objective-space unit-simplex direction matching via log-domain Sinkhorn.
  2. Individual-level local tangent extrapolation weighted by ``alpha_vec``.
  3. 1-D sliced marginal Wasserstein quantile alignment along ``L = 32``
     random unit directions, tangent micro-perturbation and reflective clamping.
"""
from __future__ import annotations

import torch

from flow_dmo.core.dominance import batched_gather, batched_pareto_ranks
from flow_dmo.core.tensor_ops import reflect_clamp

Tensor = torch.Tensor


def _extract_elites(Z: Tensor, F: Tensor, n_elite: int) -> tuple[Tensor, Tensor]:
    """Select ``n_elite`` non-dominated (Rank-0) elites with diverse crowding order.

    If a seed has fewer than ``n_elite`` rank-0 individuals, wraps around its
    rank-0 front (minimum 4 elites) so dominated garbage never pollutes OT.
    """
    B, N, _ = Z.shape
    ranks = batched_pareto_ranks(F)
    order = torch.argsort(ranks, dim=1, stable=True)
    # Count rank-0 elites per seed (floor at min(N, max(4, N // 4)) for diversity)
    min_k = min(N, max(4, N // 4))
    r0_cnt = (ranks == 0).sum(dim=1).clamp(min=min_k, max=N)               # (B,)
    pos = torch.arange(n_elite, device=Z.device).unsqueeze(0) % r0_cnt.unsqueeze(1)
    idx = torch.gather(order, 1, pos)
    return batched_gather(Z, idx), batched_gather(F, idx)


def _sinkhorn_match(F_curr: Tensor, F_prev: Tensor, eps: float = 0.05, iters: int = 3) -> Tensor:
    """Soft doubly-stochastic alignment matrix ``P (B, N, N)`` in objective direction space."""
    z_c = F_curr.min(dim=1, keepdim=True).values
    z_p = F_prev.min(dim=1, keepdim=True).values
    Fc_dir = (F_curr - z_c) / (torch.linalg.norm(F_curr - z_c, dim=-1, keepdim=True) + 1e-8)
    Fp_dir = (F_prev - z_p) / (torch.linalg.norm(F_prev - z_p, dim=-1, keepdim=True) + 1e-8)

    # Cosine distance cost matrix C in [0, 2]
    C = 1.0 - torch.bmm(Fc_dir, Fp_dir.transpose(1, 2))
    log_K = -C / max(eps, 1e-4)
    for _ in range(iters):
        log_K = log_K - torch.logsumexp(log_K, dim=-1, keepdim=True)
        log_K = log_K - torch.logsumexp(log_K, dim=-2, keepdim=True)
    P = torch.exp(log_K)
    P = P / torch.clamp(P.sum(dim=-1, keepdim=True), min=1e-12)
    return P


class SlicedOTTransport:
    """Reference-aligned Sliced Wasserstein push-forward map."""

    def __init__(
        self,
        n_slices: int = 32,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 3,
        tangent_noise: float = 0.12,
        align_strength: float = 0.25,
    ):
        self.n_slices = int(n_slices)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.tangent_noise = float(tangent_noise)
        self.align_strength = float(align_strength)

    def push_forward(
        self,
        prev_elites: tuple[Tensor, Tensor],
        curr_elites: tuple[Tensor, Tensor],
        mu_hat: Tensor,
        sigma_hat: Tensor,
        alpha_vec: Tensor,
        count: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate ``count`` topology-preserving transferred candidates ``(B, count, D)``."""
        Z_prev_all, F_prev_all = prev_elites
        Z_curr_all, F_curr_all = curr_elites
        B, N_curr, D = Z_curr_all.shape
        dev, dt = Z_curr_all.device, Z_curr_all.dtype
        if count <= 0:
            return torch.empty((B, 0, D), device=dev, dtype=dt)

        n_elite = min(N_curr, Z_prev_all.shape[1])
        Z_curr, F_curr = _extract_elites(Z_curr_all, F_curr_all, n_elite)
        Z_prev, F_prev = _extract_elites(Z_prev_all, F_prev_all, n_elite)

        # 1. Objective-space reference-direction Sinkhorn pairing & manifold denoising
        #    Neighbors in objective direction share nearly identical position u_i,
        #    so local averaging denoises distance-variable residuals while keeping
        #    curved manifold geometry intact.
        P = _sinkhorn_match(F_curr, F_prev, eps=self.sinkhorn_eps, iters=self.sinkhorn_iters)
        P_self = _sinkhorn_match(F_curr, F_curr, eps=self.sinkhorn_eps, iters=2)
        Z_curr_denoised = 0.85 * Z_curr + 0.15 * torch.bmm(P_self, Z_curr)
        Z_prev_aligned = torch.bmm(P, Z_prev)

        # 2. James-Stein shrunk tangent flow: combines low-variance global centroid
        #    velocity (mu_k - mu_{k-1}) with local manifold deformation (Z_curr - Z_prev_aligned).
        global_delta = (Z_curr.mean(dim=1, keepdim=True) - Z_prev.mean(dim=1, keepdim=True))
        local_delta = Z_curr_denoised - Z_prev_aligned
        shrunk_delta = 0.70 * global_delta + 0.30 * local_delta
        Z_base = Z_curr_denoised + alpha_vec.unsqueeze(1) * shrunk_delta

        # Expand / subsample from n_elite to `count`
        if count == n_elite:
            Z_raw = Z_base
        else:
            pick = torch.randint(0, n_elite, (B, count), device=dev, generator=generator)
            Z_raw = batched_gather(Z_base, pick)

        # 3. 1-D Sliced Marginal Wasserstein quantile alignment along L random unit directions
        U = torch.randn((D, self.n_slices), device=dev, dtype=dt, generator=generator)
        U = U / torch.clamp(torch.linalg.norm(U, dim=0, keepdim=True), min=1e-12)

        # Project Z_raw: (B, count, L)
        proj = torch.matmul(Z_raw, U)
        proj_mean = proj.mean(dim=1, keepdim=True)
        proj_std = proj.std(dim=1, keepdim=True).clamp(min=1e-4)

        # Target distribution moments along each slice u_l:
        # mean_l = mu_hat @ U, std_l = sqrt((sigma_hat^2) @ (U^2))
        tgt_mean = torch.matmul(mu_hat, U).unsqueeze(1)                           # (B, 1, L)
        tgt_var = torch.matmul(sigma_hat ** 2, U ** 2).unsqueeze(1).clamp(min=1e-8)
        tgt_std = torch.sqrt(tgt_var)

        # Bounded 1-D Wasserstein scale alignment (prevents blow-up on collapsed dims)
        scale_ratio = torch.clamp(tgt_std / proj_std, 0.5, 2.0)
        proj_aligned = tgt_mean + (proj - proj_mean) * scale_ratio
        slice_delta = (proj_aligned - proj)                                       # (B, count, L)
        # Back-project average correction (exact for isotropic U with E[U U^T] = (L/D) I)
        correction = torch.matmul(slice_delta, U.T) * (float(D) / float(self.n_slices))
        Z_aligned = Z_raw + self.align_strength * correction

        # 4. Tangent micro-perturbation & reflective boundary clamping
        noise = torch.randn((B, count, D), device=dev, dtype=dt, generator=generator)
        Z_out = Z_aligned + noise * (self.tangent_noise * sigma_hat.unsqueeze(1))
        return reflect_clamp(Z_out)
