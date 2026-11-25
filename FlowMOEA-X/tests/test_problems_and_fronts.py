"""Batch 2 acceptance tests: FDA1-4, dMOP1-3, CEC 2018 DF1-14 and Microgrid-DEED."""
from __future__ import annotations

import pytest
import torch
from conftest import SEED

from flow_dmo.config import make_generator
from flow_dmo.core.fe_ledger import FELedger
from flow_dmo.core.kernels import TensorNSGA2
from flow_dmo.core.tensor_ops import uniform_population
from flow_dmo.problems import ALL_ANALYTICAL_PROBLEMS, DynamicMicrogridDEED, FDA4, make_problem


def _igd(F: torch.Tensor, PF: torch.Tensor) -> float:
    """IGD(F, PF) in float64."""
    d = torch.cdist(PF.double(), F.double())
    return float(d.min(dim=-1).values.mean().item())


@pytest.mark.parametrize("name", ALL_ANALYTICAL_PROBLEMS)
@pytest.mark.parametrize("t", [0.0, 0.3, 1.0, 1.7])
def test_analytical_ps_matches_true_pf(device, name, t):
    prob = make_problem(name, device=device)
    X_ps = prob.get_true_ps(t, n_points=500, device=device)
    assert X_ps.shape[1] == prob.n_var
    # PS points must stay inside physical bounds
    assert torch.all(X_ps >= prob.xl.squeeze(0) - 1e-5)
    assert torch.all(X_ps <= prob.xu.squeeze(0) + 1e-5)

    F_ps = prob.evaluate_batch(X_ps.unsqueeze(0), t)[0]
    assert torch.isfinite(F_ps).all()
    assert F_ps.shape[1] == prob.n_obj

    PF = prob.get_true_pf(t, n_points=500, device=device)
    assert torch.isfinite(PF).all()
    assert PF.shape[1] == prob.n_obj
    assert _igd(F_ps, PF) < 5e-3


@pytest.mark.parametrize("M", [2, 3, 5])
def test_fda4_many_objective_configurable(device, M):
    prob = FDA4(n_obj=M, device=device)
    assert prob.n_obj == M and prob.n_var == M + 9
    X_ps = prob.get_true_ps(0.5, n_points=400, device=device)
    F_ps = prob.evaluate_batch(X_ps.unsqueeze(0), 0.5)[0]
    norms = torch.linalg.norm(F_ps, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    PF = prob.get_true_pf(0.5, n_points=400, device=device)
    assert _igd(F_ps, PF) < 5e-3


def test_batched_evaluation_shapes_and_pf_cache(device):
    prob = make_problem("DF8", device=device)
    g = make_generator(SEED, device)
    X = prob.xl + torch.rand((8, 40, prob.n_var), device=device, generator=g) * (prob.xu - prob.xl)
    F = prob.evaluate_batch(X, 0.4)
    assert F.shape == (8, 40, 2)
    assert torch.isfinite(F).all()
    pf1 = prob.get_true_pf(0.4, n_points=200, device=device)
    pf2 = prob.get_true_pf(0.4, n_points=200, device=device)
    assert pf1 is pf2


def test_deed_microgrid_problem(device):
    prob = DynamicMicrogridDEED(device=device)
    assert prob.n_var == 10 and prob.n_obj == 2
    g = make_generator(SEED, device)
    Z = uniform_population(4, 30, 10, device=device, generator=g)
    X = prob.xl + Z * (prob.xu - prob.xl)
    F_noon = prob.evaluate_batch(X, t=0.5)
    F_night = prob.evaluate_batch(X, t=0.0)
    assert F_noon.shape == (4, 30, 2)
    assert torch.isfinite(F_noon).all() and torch.isfinite(F_night).all()
    # Dynamic environment: same dispatch gives different cost/emission at noon vs midnight
    assert not torch.allclose(F_noon, F_night)
    pf = prob.get_true_pf(0.5, n_points=200, device=device)
    assert pf.shape[1] == 2 and pf.shape[0] >= 10


def test_strict_fe_budget_over_20_environments(device):
    """Acceptance check: 20 environments with strict_fe_budget=True -> exact 20*tau_t*N FEs."""
    B, N, tau_t, n_envs = 4, 20, 5, 20
    prob = make_problem("DF1", n_var=10, tau_t=tau_t, n_t=10, device=device)
    ledger = FELedger(n_seeds=B, pop_size=N, tau_t=tau_t, device=device, strict_fe_budget=True)
    kernel = TensorNSGA2(pop_size=N, n_obj=prob.n_obj)
    g = make_generator(SEED, device)

    Z = uniform_population(B, N, prob.n_var, device=device, generator=g)
    for env in range(n_envs):
        t = prob.time_from_step(env)
        ledger.start_environment(env, t)
        if env == 0:
            F = ledger.evaluate_and_charge(prob, Z, t, "warmstart")
        else:
            n_sensor = max(5, int(0.1 * N))
            Z_s = Z[:, :n_sensor]
            F_s = ledger.evaluate_and_charge(prob, Z_s, t, "sensor")
            Z_new = uniform_population(B, N - n_sensor, prob.n_var, device=device, generator=g)
            F_new = ledger.evaluate_and_charge(prob, Z_new, t, "warmstart")
            Z = torch.cat([Z_s, Z_new], dim=1)
            F = torch.cat([F_s, F_new], dim=1)
        for _ in range(ledger.remaining_generations_in_env()):
            Z, F = kernel.step_generation(
                Z, F, lambda z: ledger.evaluate_and_charge(prob, z, t, "evolution"), g
            )

    expected = n_envs * tau_t * N
    assert ledger.total_fe_per_seed.tolist() == [expected] * B
    assert ledger.all_envs_exact()
