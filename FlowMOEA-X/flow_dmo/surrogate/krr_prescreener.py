"""Zero-extra-FE closed-form Kernel Ridge Regression (KRR) surrogate pre-screener
(Section 4.5).

Uses the already-evaluated individuals in the new environment ``(Z_train, F_train)``
to fit a batched RBF Kernel Ridge Regression model via Cholesky factorisation,
scores ``4 * N_trans`` oversampled transfer candidates with Lower Confidence
Bound ``F_lcb = F_hat - beta * s_hat``, and selects the top ``N_trans`` via
batched non-dominated sorting + crowding distance — consuming 0 extra real FEs.
"""
from __future__ import annotations

import torch

from flow_dmo.core.dominance import batched_gather, nsga2_select

Tensor = torch.Tensor


class KRRPrescreener:
    """Closed-form batched RBF-KRR + LCB non-dominated pre-screener."""

    def __init__(self, reg_lambda: float = 1e-3, lcb_beta: float = 0.2, gamma: float | None = None):
        self.reg_lambda = float(reg_lambda)
        self.lcb_beta = float(lcb_beta)
        self.gamma = gamma

    @torch.no_grad()
    def predict(
        self,
        Z_train: Tensor,
        F_train: Tensor,
        Z_candidates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict candidate objectives ``F_hat (B, N_cand, M)`` and std ``s_hat (B, N_cand, 1)``."""
        B, N_tr, D = Z_train.shape
        dev, dt = Z_train.device, Z_train.dtype
        gamma = (1.0 / float(D)) if self.gamma is None else float(self.gamma)

        # Fast BLAS GEMM squared Euclidean distances (no sqrt + square overhead)
        tr_sq = (Z_train ** 2).sum(dim=-1, keepdim=True)                    # (B, N_tr, 1)
        ca_sq = (Z_candidates ** 2).sum(dim=-1, keepdim=True)               # (B, N_cand, 1)
        d2_tr = (tr_sq + tr_sq.transpose(1, 2) - 2.0 * torch.bmm(Z_train, Z_train.transpose(1, 2))).clamp(min=0.0)
        d2_ct = (ca_sq + tr_sq.transpose(1, 2) - 2.0 * torch.bmm(Z_candidates, Z_train.transpose(1, 2))).clamp(min=0.0)

        # Adaptive bandwidth scaled to training cloud spread avoids OOD collapse
        if self.gamma is None:
            med_d2 = d2_tr.mean(dim=(1, 2), keepdim=True).clamp(min=0.1)
            gamma_t = 0.5 / med_d2
        else:
            gamma_t = torch.tensor(float(self.gamma), device=dev, dtype=dt)

        K_tr = torch.exp(-gamma_t * d2_tr)
        eye = torch.eye(N_tr, device=dev, dtype=dt).unsqueeze(0)
        K_reg = K_tr + self.reg_lambda * eye

        # Strictly monotone log1p compression prevents high-g LHS outliers from distorting RBF fit
        f_shift = F_train.min(dim=1, keepdim=True).values
        F_log = torch.log1p(torch.clamp(F_train - f_shift, min=0.0))
        f_mean = F_log.mean(dim=1, keepdim=True)
        f_std = F_log.std(dim=1, keepdim=True).clamp(min=1e-6)
        F_norm = (F_log - f_mean) / f_std

        K_ct = torch.exp(-gamma_t * d2_ct)                                  # (B, N_cand, N_tr)

        # Batched Cholesky solve
        L, info = torch.linalg.cholesky_ex(K_reg)
        if bool((info != 0).any()):
            K_reg = K_reg + (10.0 * self.reg_lambda) * eye
            L = torch.linalg.cholesky(K_reg)

        W = torch.cholesky_solve(F_norm, L)                                 # (B, N_tr, M)
        F_hat_norm = torch.bmm(K_ct, W)                                     # (B, N_cand, M)

        # Posterior variance s^2(x) = 1 - k^T (K + lam I)^{-1} k
        V = torch.linalg.solve_triangular(L, K_ct.transpose(1, 2), upper=False)  # (B, N_tr, N_cand)
        var = torch.clamp(1.0 - (V ** 2).sum(dim=1), min=0.0)               # (B, N_cand)
        s_hat_norm = torch.sqrt(var).unsqueeze(-1)                          # (B, N_cand, 1)

        F_hat_log = F_hat_norm * f_std + f_mean
        s_hat = s_hat_norm * f_std
        return F_hat_log, s_hat

    def select_best_candidates(
        self,
        Z_train: Tensor,
        F_train: Tensor,
        Z_candidates: Tensor,
        select_count: int,
    ) -> Tensor:
        """Select top ``select_count`` candidates from ``Z_candidates`` with 0 real FEs."""
        B, N_cand, D = Z_candidates.shape
        if select_count <= 0:
            return torch.empty((B, 0, D), device=Z_candidates.device, dtype=Z_candidates.dtype)
        if select_count >= N_cand or Z_train.shape[1] < 2:
            return Z_candidates[:, :select_count]

        F_hat, s_hat = self.predict(Z_train, F_train, Z_candidates)
        F_lcb = F_hat - self.lcb_beta * s_hat
        idx, _, _ = nsga2_select(F_lcb, select_count)
        return batched_gather(Z_candidates, idx)
