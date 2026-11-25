"""Batch 3 acceptance tests (part A): Kalman momentum filter, Sliced-OT manifold
fidelity, Flow Matching ODE bounds, change detector and portfolio gate."""
from __future__ import annotations

import torch
from conftest import SEED, assert_finite_in_unit_box

from flow_dmo.config import make_generator
from flow_dmo.core.fe_ledger import FELedger
from flow_dmo.problems import make_problem
from flow_dmo.transport import (
    FlowMatchingTransport,
    KalmanMomentumFilter,
    PopulationChangeDetector,
    PortfolioGate,
    SlicedOTTransport,
)


def test_kalman_per_dimension_decoupling(device):
    """Spec acceptance assertion:
      * dim 0: static position variable -> alpha[0] == 0.0
      * dim 1: monotonic drift         -> alpha[1] > 0.7
      * dim 2: sinusoidal reversal     -> alpha[2] <= 0.2
    """
    kf = KalmanMomentumFilter()
    B = 4
    sigma = torch.tensor([[0.28, 0.04, 0.04]], device=device).expand(B, 3)

    mu0 = torch.tensor([[0.500, 0.20, 0.50]], device=device).expand(B, 3)
    mu1 = torch.tensor([[0.501, 0.35, 0.65]], device=device).expand(B, 3)
    mu2 = torch.tensor([[0.500, 0.50, 0.55]], device=device).expand(B, 3)

    kf.update_and_predict(mu0, sigma)
    kf.update_and_predict(mu1, sigma)
    mu_hat, sigma_hat, alpha = kf.update_and_predict(mu2, sigma)

    assert alpha.shape == (B, 3)
    assert torch.all(alpha[:, 0] == 0.0)
    assert torch.all(alpha[:, 1] > 0.7)
    assert torch.all(alpha[:, 2] <= 0.2)
    assert_finite_in_unit_box(mu_hat)
    # Reversal dimension expands variance; monotonic shrinks it
    assert torch.all(sigma_hat[:, 2] > sigma[:, 2])
    assert torch.all(sigma_hat[:, 1] < sigma[:, 1])


def test_sliced_ot_beats_isotropic_gaussian_on_curved_manifold(device):
    """Spec acceptance assertion: on curved parabola x2 = 0.5 * x1^2 + t,
    Sliced-OT manifold MSE is reduced by > 50% vs isotropic Gaussian resampling.
    """
    B, N, D = 5, 80, 4
    g = make_generator(SEED, device)

    def make_parabola_elites(shift: float):
        u = torch.linspace(0.05, 0.95, N, device=device).unsqueeze(0).expand(B, N)
        Z = torch.zeros((B, N, D), device=device)
        Z[..., 0] = u
        Z[..., 1:] = torch.clamp(0.45 * (u ** 2) + shift, 0.0, 1.0).unsqueeze(-1)
        F = torch.stack([u, 1.0 - torch.sqrt(u)], dim=-1)
        return Z, F

    Z_prev, F_prev = make_parabola_elites(0.10)
    Z_curr, F_curr = make_parabola_elites(0.25)
    t_next = 0.40

    kf = KalmanMomentumFilter()
    kf.update_and_predict(Z_prev.mean(dim=1), Z_prev.std(dim=1))
    mu_hat, sigma_hat, alpha = kf.update_and_predict(Z_curr.mean(dim=1), Z_curr.std(dim=1))
    # Give a third step with exact linear velocity so alpha_vec ~ 1.0 on distance dims
    alpha_exact = torch.tensor([[0.0, 1.0, 1.0, 1.0]], device=device).expand(B, D)
    mu_next = Z_curr.mean(dim=1) + alpha_exact * (Z_curr.mean(dim=1) - Z_prev.mean(dim=1))
    sigma_next = Z_curr.std(dim=1)

    ot = SlicedOTTransport(n_slices=32, tangent_noise=0.05, align_strength=0.15)
    Z_ot = ot.push_forward(
        (Z_prev, F_prev), (Z_curr, F_curr), mu_next, sigma_next, alpha_exact, count=N, generator=g
    )
    assert Z_ot.shape == (B, N, D)
    assert_finite_in_unit_box(Z_ot)

    # Isotropic/diagonal Gaussian baseline centred at mu_next with sigma_next
    Z_gauss = torch.clamp(
        mu_next.unsqueeze(1) + torch.randn((B, N, D), device=device, generator=g) * sigma_next.unsqueeze(1),
        0.0,
        1.0,
    )

    def manifold_mse(Z):
        x1 = Z[..., 0]
        target = torch.clamp(0.45 * (x1 ** 2) + t_next, 0.0, 1.0).unsqueeze(-1)
        return float(((Z[..., 1:] - target) ** 2).mean().item())

    mse_ot = manifold_mse(Z_ot)
    mse_gauss = manifold_mse(Z_gauss)
    assert mse_ot < 0.5 * mse_gauss, f"Expected >50% MSE reduction: OT={mse_ot:.5f} vs Gauss={mse_gauss:.5f}"


def test_flow_matching_ode_stays_in_unit_box(device):
    B, N, D = 4, 40, 6
    g = make_generator(SEED, device)
    Z0 = torch.rand((B, N, D), device=device, generator=g) * 0.6 + 0.1
    Z1 = torch.clamp(Z0 + 0.15, 0.0, 1.0)
    F0 = torch.stack([Z0[..., 0], 1.0 - Z0[..., 0]], dim=-1)
    F1 = torch.stack([Z1[..., 0], 1.0 - Z1[..., 0]], dim=-1)

    fm = FlowMatchingTransport(dim=D, hidden=64, n_steps=8, device=device)
    mu_hat = torch.clamp(Z1.mean(dim=1) + 0.15, 0.0, 1.0)
    sig_hat = Z1.std(dim=1).clamp(min=1e-2)
    Z_next = fm.step_ode_pushforward((Z0, F0), (Z1, F1), mu_hat, sig_hat, count=60, generator=g)
    assert Z_next.shape == (B, 60, D)
    assert_finite_in_unit_box(Z_next)


def test_change_detector_and_portfolio_gate(device):
    B, N, D = 4, 50, 10
    prob = make_problem("DF1", n_var=D, device=device)
    ledger = FELedger(n_seeds=B, pop_size=N, tau_t=5, device=device)
    ledger.start_environment(0, 0.0)
    g = make_generator(SEED, device)
    Z = torch.rand((B, N, D), device=device, generator=g)
    F = ledger.evaluate_and_charge(prob, Z, 0.0, "warmstart")

    det = PopulationChangeDetector()
    ledger.start_environment(1, 0.5)
    res = det.detect(Z, F, prob, t_new=0.5, ledger=ledger)
    assert res.changed.all()
    assert res.sensor_Z.shape == (B, 5, D)
    assert ledger.role_totals()["sensor"] == 5

    gate = PortfolioGate()
    assert gate.regime_ratios(torch.full((B,), 0.01))[3] == "steady"
    assert gate.regime_ratios(torch.full((B,), 0.15), torch.full((B,), 0.05))[3] == "normal"
    assert gate.regime_ratios(torch.full((B,), 0.85))[3] == "circuit_breaker"
    assert gate.regime_ratios(torch.full((B,), 0.15), torch.full((B,), 0.40))[3] == "circuit_breaker"

    for d_val in (0.01, 0.15, 0.70):
        nt, ner, nr = gate.compute_counts(N=100, n_sensor=10, D_k=torch.full((B,), d_val))
        assert nt + ner + nr == 90
