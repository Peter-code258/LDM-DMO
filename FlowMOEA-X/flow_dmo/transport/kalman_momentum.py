"""Per-dimension heterogeneous Kalman momentum filter (Section 4.2).

Decouples static position variables (large spread, near-zero centroid drift)
from monotonic distance variables and sinusoidal reversal points, assigning an
individual step-size vector ``alpha_vec`` of shape ``(B, D)`` instead of a
single scalar.
"""
from __future__ import annotations

import torch

Tensor = torch.Tensor


class KalmanMomentumFilter:
    """Tracks the last three elite centroids ``(mu_{k-2}, mu_{k-1}, mu_k)`` and
    spreads ``sigma_k``, predicting ``(mu_hat_{k+1}, sigma_hat_{k+1}, alpha_k)``.
    """

    def __init__(
        self,
        sigma_min: float = 1e-3,
        sigma_max: float = 0.5,
        static_delta_thresh: float = 5e-3,
        static_sigma_thresh: float = 0.12,
        reversal_alpha: float = 0.15,
    ):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.static_delta_thresh = float(static_delta_thresh)
        self.static_sigma_thresh = float(static_sigma_thresh)
        self.reversal_alpha = float(reversal_alpha)
        self.reset()

    def reset(self) -> None:
        self.mu_history: list[Tensor] = []
        self.sigma_history: list[Tensor] = []
        self.last_mu_hat: Tensor | None = None
        self.last_prediction_error: Tensor | None = None
        self.last_alpha: Tensor | None = None

    def update_and_predict(self, mu_k: Tensor, sigma_k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Update with current elite moments ``(B, D)`` and predict next environment.

        Returns ``(mu_hat, sigma_hat, alpha_vec)``, each of shape ``(B, D)``.
        """
        mu_k = torch.clamp(mu_k, 0.0, 1.0)
        sigma_k = torch.clamp(sigma_k, self.sigma_min, self.sigma_max)

        if self.last_mu_hat is not None:
            # Dimension-invariant RMS prediction error in [0, 1]
            self.last_prediction_error = torch.sqrt(((mu_k - self.last_mu_hat) ** 2).mean(dim=-1))
        else:
            self.last_prediction_error = torch.zeros(mu_k.shape[0], device=mu_k.device, dtype=mu_k.dtype)

        self.mu_history.append(mu_k.clone())
        self.sigma_history.append(sigma_k.clone())
        if len(self.mu_history) > 3:
            self.mu_history.pop(0)
            self.sigma_history.pop(0)

        n_hist = len(self.mu_history)
        if n_hist == 1:
            alpha = torch.zeros_like(mu_k)
            mu_hat = mu_k.clone()
            sigma_hat = torch.clamp(sigma_k * 1.15, self.sigma_min, self.sigma_max)
        elif n_hist == 2:
            delta_k = self.mu_history[-1] - self.mu_history[-2]
            is_static = (torch.abs(delta_k) < self.static_delta_thresh) & (sigma_k > self.static_sigma_thresh)
            alpha = torch.where(is_static, torch.zeros_like(mu_k), torch.ones_like(mu_k))
            mu_hat = torch.clamp(mu_k + alpha * delta_k, 0.0, 1.0)
            sigma_hat = torch.where(is_static, sigma_k, sigma_k * 0.90)
            sigma_hat = torch.clamp(sigma_hat, self.sigma_min, self.sigma_max)
        else:
            delta_k = self.mu_history[-1] - self.mu_history[-2]
            delta_prev = self.mu_history[-2] - self.mu_history[-3]

            is_static = (torch.abs(delta_k) < self.static_delta_thresh) & (sigma_k > self.static_sigma_thresh)
            prod = delta_k * delta_prev
            is_monotonic = (prod > 0.0) & (~is_static)
            is_reversal = (prod <= 0.0) & (~is_static)

            ratio = torch.abs(delta_k) / (torch.abs(delta_prev) + 1e-5)
            alpha_mono = torch.clamp(0.75 + 0.35 * ratio, 0.65, 1.2)
            alpha = torch.where(
                is_static,
                torch.zeros_like(mu_k),
                torch.where(is_monotonic, alpha_mono, torch.full_like(mu_k, self.reversal_alpha)),
            )

            v_hat = alpha * delta_k
            mu_hat = torch.clamp(mu_k + v_hat, 0.0, 1.0)

            scale = torch.where(
                is_static,
                torch.ones_like(sigma_k),
                torch.where(is_monotonic, torch.full_like(sigma_k, 0.85), torch.full_like(sigma_k, 1.45)),
            )
            sigma_hat = torch.clamp(sigma_k * scale, self.sigma_min, self.sigma_max)

        self.last_mu_hat = mu_hat.clone()
        self.last_alpha = alpha.clone()
        return mu_hat, sigma_hat, alpha
