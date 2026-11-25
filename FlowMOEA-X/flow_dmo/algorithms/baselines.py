"""Five fully-tensorised baseline DMOEAs (Section 2 & 10):
  1. ``TensorDNSGA2``: Deb's DNSGA-II-A (20% random re-initialisation, 80% retain).
  2. ``TensorLDT_v32``: Legacy LDT-NSGA-II v3.2 (diagonal Gaussian resampling,
     scalar step alpha=1.0, fixed 50%/20%/30% split).
  3. ``TensorPPS``: Population Prediction Strategy (AR(1) centre prediction +
     manifold shift from previous non-dominated set).
  4. ``TensorAR``: Auto-Regressive AR(2) elite centroid + Gaussian spread.
  5. ``TensorGMM``: 2-component Gaussian Mixture Model transfer (split along
     principal objective axis).
"""
from __future__ import annotations

import torch

from flow_dmo.algorithms.base import BatchedDynamicMOEA
from flow_dmo.core.dominance import batched_gather, batched_pareto_ranks
from flow_dmo.core.fe_ledger import FELedger
from flow_dmo.core.tensor_ops import reflect_clamp, uniform_population
from flow_dmo.problems.base import DynamicProblem

Tensor = torch.Tensor


class TensorDNSGA2(BatchedDynamicMOEA):
    """DNSGA-II-A: replaces ``zeta = 0.2`` of population with random solutions."""

    name = "DNSGA2"

    def __init__(self, pop_size: int, n_var: int, n_obj: int, device="cpu", zeta: float = 0.2):
        super().__init__(pop_size, n_var, n_obj, device)
        self.zeta = float(zeta)

    def respond_batch(self, prev_Z, prev_F, sensor_Z, sensor_F_new, D_k, problem, t_new, ledger, generator):
        B, _, D = prev_Z.shape
        budget = self.pop_size - sensor_Z.shape[1]
        n_rand = min(budget, max(1, int(round(self.zeta * self.pop_size))))
        n_retain = budget - n_rand
        Z_ret = self.select_diverse_elites(prev_Z, prev_F, n_retain)
        Z_rnd = uniform_population(B, n_rand, D, device=prev_Z.device, dtype=prev_Z.dtype, generator=generator)
        Z_new = torch.cat([Z_ret, Z_rnd], dim=1)
        F_new = ledger.evaluate_and_charge(problem, Z_new, t_new, role="warmstart")
        return torch.cat([sensor_Z, Z_new], dim=1), torch.cat([sensor_F_new, F_new], dim=1)


class TensorLDT_v32(BatchedDynamicMOEA):
    """Legacy LDT-NSGA-II v3.2: isotropic/diagonal Gaussian transfer with fixed 50/20/30 split."""

    name = "LDT_v32"

    def __init__(self, pop_size: int, n_var: int, n_obj: int, device="cpu"):
        super().__init__(pop_size, n_var, n_obj, device)
        self._prev_mu: Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self._prev_mu = None

    def respond_batch(self, prev_Z, prev_F, sensor_Z, sensor_F_new, D_k, problem, t_new, ledger, generator):
        B, _, D = prev_Z.shape
        dev, dt = prev_Z.device, prev_Z.dtype
        mu_k, sigma_k, _ = self.extract_elite_moments(prev_Z, prev_F)
        mu_hat = torch.clamp(mu_k + (mu_k - self._prev_mu), 0.0, 1.0) if self._prev_mu is not None else mu_k
        self._prev_mu = mu_k.clone()

        n_sensor = sensor_Z.shape[1]
        budget = self.pop_size - n_sensor
        n_trans = min(budget, int(round(0.50 * self.pop_size)))
        n_extra_ret = max(0, min(budget - n_trans, int(round(0.20 * self.pop_size)) - n_sensor))
        n_rand = budget - n_trans - n_extra_ret

        noise = torch.randn((B, n_trans, D), device=dev, dtype=dt, generator=generator)
        Z_trans = torch.clamp(mu_hat.unsqueeze(1) + noise * sigma_k.unsqueeze(1), 0.0, 1.0)
        Z_ret = self.select_diverse_elites(prev_Z, prev_F, n_extra_ret)
        Z_rnd = uniform_population(B, n_rand, D, device=dev, dtype=dt, generator=generator)
        Z_new = torch.cat([Z_trans, Z_ret, Z_rnd], dim=1)
        F_new = ledger.evaluate_and_charge(problem, Z_new, t_new, role="warmstart")
        return torch.cat([sensor_Z, Z_new], dim=1), torch.cat([sensor_F_new, F_new], dim=1)


class TensorPPS(BatchedDynamicMOEA):
    """Population Prediction Strategy (Zhou et al. 2014): centre AR(1) + previous manifold."""

    name = "PPS"

    def __init__(self, pop_size: int, n_var: int, n_obj: int, device="cpu"):
        super().__init__(pop_size, n_var, n_obj, device)
        self._centers: list[Tensor] = []

    def reset(self) -> None:
        super().reset()
        self._centers.clear()

    def respond_batch(self, prev_Z, prev_F, sensor_Z, sensor_F_new, D_k, problem, t_new, ledger, generator):
        B, N, D = prev_Z.shape
        dev, dt = prev_Z.device, prev_Z.dtype
        mu_k, sigma_k, _ = self.extract_elite_moments(prev_Z, prev_F)
        if self._centers:
            shift = mu_k - self._centers[-1]
            noise_scale = torch.clamp(torch.linalg.norm(shift, dim=-1, keepdim=True) * 0.15, 0.01, 0.2)
        else:
            shift = torch.zeros_like(mu_k)
            noise_scale = torch.full((B, 1), 0.05, device=dev, dtype=dt)
        self._centers.append(mu_k.clone())
        if len(self._centers) > 3:
            self._centers.pop(0)

        budget = self.pop_size - sensor_Z.shape[1]
        n_pred = int(round(0.8 * budget))
        n_rand = budget - n_pred

        Z_elites = self.select_diverse_elites(prev_Z, prev_F, n_pred)
        eps = torch.randn((B, n_pred, D), device=dev, dtype=dt, generator=generator) * noise_scale.unsqueeze(-1)
        Z_pred = reflect_clamp(Z_elites + shift.unsqueeze(1) + eps)
        Z_rnd = uniform_population(B, n_rand, D, device=dev, dtype=dt, generator=generator)
        Z_new = torch.cat([Z_pred, Z_rnd], dim=1)
        F_new = ledger.evaluate_and_charge(problem, Z_new, t_new, role="warmstart")
        return torch.cat([sensor_Z, Z_new], dim=1), torch.cat([sensor_F_new, F_new], dim=1)


class TensorAR(BatchedDynamicMOEA):
    """Second-order auto-regressive AR(2) centre prediction with Gaussian spread."""

    name = "AR"

    def __init__(self, pop_size: int, n_var: int, n_obj: int, device="cpu"):
        super().__init__(pop_size, n_var, n_obj, device)
        self._centers: list[Tensor] = []

    def reset(self) -> None:
        super().reset()
        self._centers.clear()

    def respond_batch(self, prev_Z, prev_F, sensor_Z, sensor_F_new, D_k, problem, t_new, ledger, generator):
        B, _, D = prev_Z.shape
        dev, dt = prev_Z.device, prev_Z.dtype
        mu_k, sigma_k, _ = self.extract_elite_moments(prev_Z, prev_F)
        if len(self._centers) >= 2:
            # AR(2): mu_{k+1} = mu_k + 0.7*(mu_k - mu_{k-1}) + 0.3*(mu_{k-1} - mu_{k-2})
            d1 = mu_k - self._centers[-1]
            d2 = self._centers[-1] - self._centers[-2]
            mu_hat = torch.clamp(mu_k + 0.7 * d1 + 0.3 * d2, 0.0, 1.0)
        elif len(self._centers) == 1:
            mu_hat = torch.clamp(mu_k + (mu_k - self._centers[-1]), 0.0, 1.0)
        else:
            mu_hat = mu_k
        self._centers.append(mu_k.clone())
        if len(self._centers) > 3:
            self._centers.pop(0)

        budget = self.pop_size - sensor_Z.shape[1]
        n_pred = int(round(0.65 * budget))
        n_rand = budget - n_pred
        eps = torch.randn((B, n_pred, D), device=dev, dtype=dt, generator=generator)
        Z_pred = reflect_clamp(mu_hat.unsqueeze(1) + eps * sigma_k.unsqueeze(1))
        Z_rnd = uniform_population(B, n_rand, D, device=dev, dtype=dt, generator=generator)
        Z_new = torch.cat([Z_pred, Z_rnd], dim=1)
        F_new = ledger.evaluate_and_charge(problem, Z_new, t_new, role="warmstart")
        return torch.cat([sensor_Z, Z_new], dim=1), torch.cat([sensor_F_new, F_new], dim=1)


class TensorGMM(BatchedDynamicMOEA):
    """2-component Gaussian Mixture transfer (splits elites along f1 median)."""

    name = "GMM"

    def __init__(self, pop_size: int, n_var: int, n_obj: int, device="cpu"):
        super().__init__(pop_size, n_var, n_obj, device)
        self._prev_mus: tuple[Tensor, Tensor] | None = None

    def reset(self) -> None:
        super().reset()
        self._prev_mus = None

    def respond_batch(self, prev_Z, prev_F, sensor_Z, sensor_F_new, D_k, problem, t_new, ledger, generator):
        B, N, D = prev_Z.shape
        dev, dt = prev_Z.device, prev_Z.dtype
        # Sort elites by f1 and split into two mixture components
        ranks = batched_pareto_ranks(prev_F)
        order = torch.argsort(ranks.to(torch.float64) * 10.0 + prev_F[..., 0].to(torch.float64), dim=1)
        half = max(2, N // 4)
        Z1 = batched_gather(prev_Z, order[:, :half])
        Z2 = batched_gather(prev_Z, order[:, half : min(N, 2 * half)])
        mu1, sig1 = Z1.mean(dim=1), Z1.std(dim=1).clamp(min=1e-3)
        mu2, sig2 = Z2.mean(dim=1), Z2.std(dim=1).clamp(min=1e-3)

        if self._prev_mus is not None:
            pm1, pm2 = self._prev_mus
            mu1_hat = torch.clamp(mu1 + (mu1 - pm1), 0.0, 1.0)
            mu2_hat = torch.clamp(mu2 + (mu2 - pm2), 0.0, 1.0)
        else:
            mu1_hat, mu2_hat = mu1, mu2
        self._prev_mus = (mu1.clone(), mu2.clone())

        budget = self.pop_size - sensor_Z.shape[1]
        n_gmm = int(round(0.65 * budget))
        n1 = n_gmm // 2
        n2 = n_gmm - n1
        n_rand = budget - n_gmm

        s1 = reflect_clamp(mu1_hat.unsqueeze(1) + torch.randn((B, n1, D), device=dev, dtype=dt, generator=generator) * sig1.unsqueeze(1))
        s2 = reflect_clamp(mu2_hat.unsqueeze(1) + torch.randn((B, n2, D), device=dev, dtype=dt, generator=generator) * sig2.unsqueeze(1))
        Z_rnd = uniform_population(B, n_rand, D, device=dev, dtype=dt, generator=generator)
        Z_new = torch.cat([s1, s2, Z_rnd], dim=1)
        F_new = ledger.evaluate_and_charge(problem, Z_new, t_new, role="warmstart")
        return torch.cat([sensor_Z, Z_new], dim=1), torch.cat([sensor_F_new, F_new], dim=1)
