"""Flagship algorithm: FlowMOEA-X (Section 7).

Supports both ``mode='lite'`` (zero-training Reference-Aligned Sliced-OT) and
``mode='pro'`` (8-step Conditional Flow Matching ODE), as well as ablation
toggles (``use_kalman``, ``use_sliced_ot``, ``use_krr``) for the interactive
dashboard and ablation report.
"""
from __future__ import annotations

import torch

from flow_dmo.algorithms.base import BatchedDynamicMOEA
from flow_dmo.config import TransportConfig
from flow_dmo.core.fe_ledger import FELedger
from flow_dmo.core.tensor_ops import reflect_clamp
from flow_dmo.problems.base import DynamicProblem
from flow_dmo.surrogate.krr_prescreener import KRRPrescreener
from flow_dmo.transport.flow_matching import FlowMatchingTransport
from flow_dmo.transport.kalman_momentum import KalmanMomentumFilter
from flow_dmo.transport.portfolio_gate import PortfolioGate
from flow_dmo.transport.sliced_ot import SlicedOTTransport

Tensor = torch.Tensor


class FlowMOEAX(BatchedDynamicMOEA):
    """FlowMOEA-X: Reference-aligned Sliced-OT + per-dimension Kalman momentum
    + zero-FE closed-form KRR LCB pre-screening.
    """

    def __init__(
        self,
        pop_size: int,
        n_var: int,
        n_obj: int,
        device: torch.device | str = "cpu",
        transport_cfg: TransportConfig | None = None,
        mode: str | None = None,
        name: str | None = None,
    ):
        super().__init__(pop_size=pop_size, n_var=n_var, n_obj=n_obj, device=device)
        cfg = transport_cfg if transport_cfg is not None else TransportConfig()
        self.cfg = cfg
        self.mode = (mode or cfg.mode).lower()
        self.name = name or ("FlowMOEA-X-Pro" if self.mode == "pro" else "FlowMOEA-X")

        self.kalman_filter = KalmanMomentumFilter()
        self.portfolio_gate = PortfolioGate()
        self.sliced_ot = SlicedOTTransport(
            n_slices=cfg.n_slices,
            sinkhorn_eps=cfg.sinkhorn_eps,
            sinkhorn_iters=cfg.sinkhorn_iters,
            tangent_noise=cfg.tangent_noise,
            align_strength=cfg.align_strength,
        )
        self.flow_matcher = (
            FlowMatchingTransport(
                dim=n_var,
                hidden=cfg.fm_hidden,
                n_steps=cfg.fm_steps,
                lr=cfg.fm_lr,
                device=self.device,
            )
            if self.mode == "pro"
            else None
        )
        self.krr_prescreener = KRRPrescreener(reg_lambda=cfg.krr_lambda, lcb_beta=cfg.lcb_beta)
        self.alpha_history: list[Tensor] = []
        self.regime_history: list[str] = []

    def reset(self) -> None:
        super().reset()
        self.kalman_filter.reset()
        self.alpha_history.clear()
        self.regime_history.clear()
        if self.mode == "pro":
            self.flow_matcher = FlowMatchingTransport(
                dim=self.n_var,
                hidden=self.cfg.fm_hidden,
                n_steps=self.cfg.fm_steps,
                lr=self.cfg.fm_lr,
                device=self.device,
            )

    def respond_batch(
        self,
        prev_Z: Tensor,
        prev_F: Tensor,
        sensor_Z: Tensor,
        sensor_F_new: Tensor,
        D_k: Tensor,
        problem: DynamicProblem,
        t_new: float,
        ledger: FELedger,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        B, _, D = prev_Z.shape
        dev, dt = prev_Z.device, prev_Z.dtype

        # 1. Extract current elite moments and update per-dimension Kalman momentum filter
        mu_k, sigma_k, _ = self.extract_elite_moments(prev_Z, prev_F)
        if self.cfg.use_kalman:
            mu_hat, sigma_hat, alpha_vec = self.kalman_filter.update_and_predict(mu_k, sigma_k)
        else:
            # Ablation: scalar alpha = 1.0 linear extrapolation
            prev_mu = self.kalman_filter.mu_history[-1] if self.kalman_filter.mu_history else mu_k
            self.kalman_filter.update_and_predict(mu_k, sigma_k)
            alpha_vec = torch.ones_like(mu_k)
            mu_hat = torch.clamp(mu_k + (mu_k - prev_mu), 0.0, 1.0)
            sigma_hat = sigma_k

        self.alpha_history.append(alpha_vec.detach().cpu())

        # 2. Dynamic portfolio gating
        _, _, _, regime = self.portfolio_gate.regime_ratios(D_k, self.kalman_filter.last_prediction_error)
        self.regime_history.append(regime)
        n_trans, n_extra_retain, n_rand = self.portfolio_gate.compute_counts(
            N=self.pop_size,
            n_sensor=sensor_Z.shape[1],
            D_k=D_k,
            pred_err=self.kalman_filter.last_prediction_error,
        )

        # 3. Evaluate extra retained elites + orthogonal LHS exploration
        Z_extra_retain = self.select_diverse_elites(prev_Z, prev_F, count=n_extra_retain)
        Z_rand = self.sample_orthogonal_lhs(B, n_rand, D, device=dev, dtype=dt, generator=generator)
        Z_known_new = torch.cat([Z_extra_retain, Z_rand], dim=1)
        F_known_new = ledger.evaluate_and_charge(problem, Z_known_new, t_new, role="warmstart")

        Z_train = torch.cat([sensor_Z, Z_known_new], dim=1)
        F_train = torch.cat([sensor_F_new, F_known_new], dim=1)

        # 4. Generate oversampled transfer candidates (K * n_trans)
        #    Unified candidate pool combines topology-preserving OT/ODE push-forward
        #    (65%) with Kalman-predicted moment-matched proposals (35%), letting the
        #    zero-FE KRR pre-screener select the best manifold + exploratory points.
        oversample = (self.cfg.oversample_factor if self.cfg.use_krr else 1) * n_trans
        if n_trans > 0:
            n_ot = int(round(0.65 * oversample)) if self.cfg.use_krr else oversample
            n_mom = oversample - n_ot
            if self.mode == "pro" and self.flow_matcher is not None and len(self.history_elites) >= 1:
                Z_ot = self.flow_matcher.step_ode_pushforward(
                    prev_elites=self.history_elites[-1],
                    curr_elites=(prev_Z, prev_F),
                    mu_hat=mu_hat,
                    sigma_hat=sigma_hat,
                    count=n_ot,
                    D_k=D_k,
                    generator=generator,
                )
            elif self.cfg.use_sliced_ot and len(self.history_elites) >= 1:
                Z_ot = self.sliced_ot.push_forward(
                    prev_elites=self.history_elites[-1],
                    curr_elites=(prev_Z, prev_F),
                    mu_hat=mu_hat,
                    sigma_hat=sigma_hat,
                    alpha_vec=alpha_vec,
                    count=n_ot,
                    generator=generator,
                )
            else:
                noise = torch.randn((B, n_ot, D), device=dev, dtype=dt, generator=generator)
                Z_ot = reflect_clamp(mu_hat.unsqueeze(1) + noise * sigma_hat.unsqueeze(1))

            if n_mom > 0:
                noise_m = torch.randn((B, n_mom, D), device=dev, dtype=dt, generator=generator)
                Z_mom = reflect_clamp(mu_hat.unsqueeze(1) + noise_m * sigma_hat.unsqueeze(1))
                Z_candidates = torch.cat([Z_ot, Z_mom], dim=1)
            else:
                Z_candidates = Z_ot

            # 5. Two-stage Zero-FE closed-form KRR + LCB pre-screening:
            #    Evaluate a balanced anchor subset (half OT, half Kalman moment) on
            #    the new environment so KRR observes both proposals before screening
            #    the remaining candidates; total real FEs remain strictly N.
            if self.cfg.use_krr and oversample > n_trans and n_trans >= 4:
                n_anchor = max(4, n_trans // 4)
                n_a_ot = min(Z_ot.shape[1], (n_anchor + 1) // 2)
                n_a_mom = n_anchor - n_a_ot
                Z_anchor = (
                    torch.cat([Z_ot[:, :n_a_ot], Z_mom[:, :n_a_mom]], dim=1)
                    if n_a_mom > 0 and n_mom > 0
                    else Z_candidates[:, :n_anchor]
                )
                Z_pool = (
                    torch.cat([Z_ot[:, n_a_ot:], Z_mom[:, n_a_mom:]], dim=1)
                    if n_a_mom > 0 and n_mom > 0
                    else Z_candidates[:, n_anchor:]
                )
                n_screen = n_trans - n_anchor
                F_anchor = ledger.evaluate_and_charge(problem, Z_anchor, t_new, role="warmstart")

                Z_krr_tr = torch.cat([Z_train, Z_anchor], dim=1)
                F_krr_tr = torch.cat([F_train, F_anchor], dim=1)
                Z_screen = self.krr_prescreener.select_best_candidates(
                    Z_train=Z_krr_tr,
                    F_train=F_krr_tr,
                    Z_candidates=Z_pool,
                    select_count=n_screen,
                )
                F_screen = ledger.evaluate_and_charge(problem, Z_screen, t_new, role="warmstart")
                Z_warm = torch.cat([Z_krr_tr, Z_screen], dim=1)
                F_warm = torch.cat([F_krr_tr, F_screen], dim=1)
            else:
                Z_selected_trans = Z_candidates[:, :n_trans]
                F_selected_trans = ledger.evaluate_and_charge(problem, Z_selected_trans, t_new, role="warmstart")
                Z_warm = torch.cat([Z_train, Z_selected_trans], dim=1)
                F_warm = torch.cat([F_train, F_selected_trans], dim=1)
        else:
            Z_warm, F_warm = Z_train, F_train

        # 7. Update elite archive (keep last 3 environments)
        self.history_elites.append((prev_Z.detach().clone(), prev_F.detach().clone()))
        if len(self.history_elites) > 3:
            self.history_elites.pop(0)
        return Z_warm, F_warm
