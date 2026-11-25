"""Real-world engineering benchmark: 24-hour Dynamic Microgrid Economic-Emission
Dispatch (DEED) with wind/PV uncertainty, BESS storage and flexible loads
(Section 5.3).
"""
from __future__ import annotations

import math

import torch

from flow_dmo.problems.base import DynamicProblem, filter_nondominated

Tensor = torch.Tensor


class DynamicMicrogridDEED(DynamicProblem):
    """10-variable, 24-hour dynamic microgrid economic & emission dispatch.

    Decision vector ``X`` (``D = 10``):
      * ``X[..., 0:6]``: active power output (MW) of 6 thermal/micro-turbine units
      * ``X[..., 6:8]``: 2 BESS battery units (positive = discharge, negative = charge, MW)
      * ``X[..., 8:10]``: 2 flexible load curtailment ratios in ``[0, 0.25]``

    Objectives (minimise both):
      * ``f1``: hourly fuel + valve-point + storage wear + curtailment + ramping cost ($)
      * ``f2``: hourly CO2/NOx emissions (kg/h) + power-balance violation penalty
    """

    name = "Microgrid-DEED"

    # Realistic 6-unit IEEE-style thermal generator parameters
    _P_MIN = [10.0, 10.0, 15.0, 15.0, 20.0, 20.0]
    _P_MAX = [80.0, 75.0, 90.0, 85.0, 110.0, 100.0]
    _COST_A = [100.0, 120.0, 80.0, 95.0, 110.0, 90.0]
    _COST_B = [24.0, 22.0, 27.0, 25.5, 21.0, 26.0]
    _COST_C = [0.06, 0.075, 0.045, 0.055, 0.040, 0.065]
    _VALVE_E = [12.0, 10.0, 15.0, 11.0, 14.0, 9.0]
    _VALVE_F = [0.06, 0.055, 0.05, 0.065, 0.045, 0.05]

    # Emission coefficients (intentionally in conflict with fuel cost:
    # cheaper coal/diesel units emit more, cleaner gas turbines cost more)
    _EM_ALPHA = [22.0, 26.0, 14.0, 18.0, 30.0, 15.0]
    _EM_BETA = [-0.18, -0.22, -0.10, -0.14, -0.25, -0.12]
    _EM_GAMMA = [0.012, 0.015, 0.006, 0.009, 0.018, 0.007]

    def __init__(self, tau_t: int = 10, n_t: int = 24, **kw):
        xl = self._P_MIN + [-25.0, -25.0, 0.0, 0.0]
        xu = self._P_MAX + [25.0, 25.0, 0.25, 0.25]
        super().__init__(n_var=10, n_obj=2, xl=xl, xu=xu, tau_t=tau_t, n_t=n_t, **kw)

    def _hour_profile(self, t: float) -> tuple[float, float, float]:
        """Return ``(P_load, P_renew, prev_dispatch_anchor)`` at time ``t``."""
        hour = (t * 24.0) % 24.0
        # Double-peaked diurnal load curve (morning + evening peaks, 260..420 MW)
        load = (
            320.0
            + 55.0 * math.sin(2.0 * math.pi * (hour - 8.0) / 24.0)
            + 30.0 * math.sin(4.0 * math.pi * (hour - 10.0) / 24.0)
        )
        # Solar bell curve (06:00-18:00) + smooth wind wave
        solar = 75.0 * max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0)) if 6.0 <= hour <= 18.0 else 0.0
        wind = 35.0 + 18.0 * math.cos(2.0 * math.pi * (hour - 2.0) / 24.0)
        renew = solar + wind
        anchor = max(120.0, load - renew) / 6.0
        return load, renew, anchor

    def evaluate_batch(self, X_real: Tensor, t: float) -> Tensor:
        dev, dt = X_real.device, X_real.dtype
        load, renew, anchor = self._hour_profile(t)

        P_gen = X_real[..., :6]
        P_bess = X_real[..., 6:8]
        curt = X_real[..., 8:10]

        p_min = torch.tensor(self._P_MIN, device=dev, dtype=dt)
        a = torch.tensor(self._COST_A, device=dev, dtype=dt)
        b = torch.tensor(self._COST_B, device=dev, dtype=dt)
        c = torch.tensor(self._COST_C, device=dev, dtype=dt)
        ve = torch.tensor(self._VALVE_E, device=dev, dtype=dt)
        vf = torch.tensor(self._VALVE_F, device=dev, dtype=dt)

        em_a = torch.tensor(self._EM_ALPHA, device=dev, dtype=dt)
        em_b = torch.tensor(self._EM_BETA, device=dev, dtype=dt)
        em_g = torch.tensor(self._EM_GAMMA, device=dev, dtype=dt)

        # Fuel + valve-point effect + ramping soft penalty
        fuel_cost = (a + b * P_gen + c * (P_gen ** 2) + torch.abs(ve * torch.sin(vf * (p_min - P_gen)))).sum(dim=-1)
        bess_wear = 12.0 * torch.abs(P_bess).sum(dim=-1)
        curt_power = curt.sum(dim=-1) * (0.15 * load)
        curt_cost = 65.0 * curt_power
        ramp_pen = 0.08 * ((P_gen - anchor) ** 2).sum(dim=-1)

        # Power balance mismatch
        supply = P_gen.sum(dim=-1) + P_bess.sum(dim=-1) + renew + curt_power
        mismatch = torch.abs(supply - load)
        balance_pen = 40.0 * mismatch + 0.5 * (mismatch ** 2)

        # Economic cost (scaled to k$/h for numerical conditioning)
        f1 = (fuel_cost + bess_wear + curt_cost + ramp_pen + balance_pen) / 1000.0

        # Emissions (scaled to 100 kg/h)
        em = (em_a + em_b * P_gen + em_g * (P_gen ** 2)).sum(dim=-1)
        f2 = (em + 25.0 * mismatch + 0.25 * (mismatch ** 2)) / 100.0
        return torch.stack([f1, f2], dim=-1)

    def _compute_true_pf(self, t: float, n_points: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Approximate reference PF via cost-vs-emission weight sweep + Sobol augmentation."""
        load, renew, _ = self._hour_profile(t)
        xl = self.xl.to(device=device, dtype=dtype).squeeze(0)
        xu = self.xu.to(device=device, dtype=dtype).squeeze(0)

        # Deterministic weight sweep between min-cost units (0,1,4) and min-emission units (2,3,5)
        n_sweep = max(400, n_points * 2)
        w = torch.linspace(0.0, 1.0, n_sweep, device=device, dtype=dtype).unsqueeze(1)
        cheap_pref = torch.tensor([[0.22, 0.22, 0.08, 0.12, 0.26, 0.10]], device=device, dtype=dtype)
        clean_pref = torch.tensor([[0.10, 0.08, 0.28, 0.20, 0.08, 0.26]], device=device, dtype=dtype)
        prefs = (1.0 - w) * cheap_pref + w * clean_pref
        prefs = prefs / prefs.sum(dim=1, keepdim=True)

        X_sweep = torch.zeros((n_sweep, self.n_var), device=device, dtype=dtype)
        X_sweep[:, 6:8] = xu[:, 6:8]
        X_sweep[:, 8:10] = 0.0
        bess_total = float(xu[:, 6:8].sum().item())
        needed = max(90.0, min(540.0, load - renew - bess_total))
        gen = prefs * needed
        for _ in range(4):
            gen = torch.clamp(gen, xl[:, :6], xu[:, :6])
            rem = needed - gen.sum(dim=1, keepdim=True)
            slack = torch.where(rem >= 0, xu[:, :6] - gen, gen - xl[:, :6]).clamp(min=0.0)
            s_sum = slack.sum(dim=1, keepdim=True).clamp(min=1e-6)
            gen = torch.clamp(gen + rem * (slack / s_sum), xl[:, :6], xu[:, :6])
        X_sweep[:, :6] = gen

        # Augment with Sobol samples projected to power balance
        n_sobol = max(1000, n_points * 4)
        eng = torch.quasirandom.SobolEngine(dimension=self.n_var, scramble=True, seed=2026)
        Z = eng.draw(n_sobol).to(device=device, dtype=dtype)
        X_sob = xl + Z * (xu - xl)
        curt_power = X_sob[:, 8:10].sum(dim=-1) * (0.15 * load)
        need_s = (load - renew - X_sob[:, 6:8].sum(dim=-1) - curt_power).clamp(min=90.0, max=540.0)
        g_sum = X_sob[:, :6].sum(dim=-1, keepdim=True).clamp(min=1e-6)
        X_sob[:, :6] = torch.clamp(X_sob[:, :6] * (need_s.unsqueeze(-1) / g_sum), xl[:, :6], xu[:, :6])

        X_all = torch.cat([X_sweep, X_sob], dim=0)
        F = self.evaluate_batch(X_all.unsqueeze(0), t)[0]
        nd = filter_nondominated(F)
        order = torch.argsort(nd[:, 0])
        return nd[order]
