"""Dynamic population portfolio gate (Section 4.6).

Replaces fixed ``50% / 20% / 30%`` allocations with three regimes governed by
normalised drift intensity ``D_k`` and historical centroid prediction residual
``e_pred = ||mu_k - mu_hat_k||_2``:

  * Micro-perturbation steady regime (``D_k < 0.03``): ``0.45 / 0.45 / 0.10``
  * Normal evolution regime (``0.03 <= D_k <= 0.45`` & ``e_pred <= 0.25``): ``0.60 / 0.20 / 0.20``
  * Abrupt change / negative-transfer circuit breaker (otherwise): ``0.25 / 0.15 / 0.60``
"""
from __future__ import annotations

import torch

Tensor = torch.Tensor


class PortfolioGate:
    """Computes integer counts ``(n_trans, n_extra_retain, n_rand)`` summing to ``N - n_sensor``."""

    def __init__(
        self,
        low_drift: float = 0.03,
        high_drift: float = 0.78,
        pred_err_thresh: float = 0.25,
    ):
        self.low_drift = float(low_drift)
        self.high_drift = float(high_drift)
        self.pred_err_thresh = float(pred_err_thresh)

    def regime_ratios(self, D_k: Tensor, pred_err: Tensor | None = None) -> tuple[float, float, float, str]:
        """Return ``(r_trans, r_retain, r_rand, regime_name)`` using batch-median severity."""
        d_val = float(D_k.median().item()) if D_k.numel() else 0.0
        e_val = float(pred_err.median().item()) if (pred_err is not None and pred_err.numel()) else 0.0

        if d_val > self.high_drift or e_val > self.pred_err_thresh:
            return 0.25, 0.15, 0.60, "circuit_breaker"
        if d_val < self.low_drift:
            return 0.45, 0.45, 0.10, "steady"
        return 0.60, 0.20, 0.20, "normal"

    def compute_counts(
        self,
        N: int,
        n_sensor: int,
        D_k: Tensor,
        pred_err: Tensor | None = None,
    ) -> tuple[int, int, int]:
        """Return ``(n_trans, n_extra_retain, n_rand)`` with exact sum ``N - n_sensor``."""
        r_trans, r_retain, r_rand, _ = self.regime_ratios(D_k, pred_err)
        budget = max(0, int(N) - int(n_sensor))
        if budget == 0:
            return 0, 0, 0

        n_retain_total = int(round(r_retain * N))
        n_extra_retain = max(0, min(budget, n_retain_total - n_sensor))
        n_trans = max(1, min(budget - n_extra_retain, int(round(r_trans * N))))
        n_rand = budget - n_extra_retain - n_trans
        if n_rand < 0:
            n_trans = max(0, n_trans + n_rand)
            n_rand = 0
        return n_trans, n_extra_retain, n_rand
