"""Batch 4 acceptance tests: GPU/CPU IGD, IGD+, exact 2D HV, QMC 3D HV,
Wilcoxon + Holm step-down, Friedman + Nemenyi CD, and end-to-end driver regression."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import SEED

from flow_dmo.algorithms import make_algorithm
from flow_dmo.core.driver import BatchedDynamicDriver
from flow_dmo.evaluation import (
    batched_hv,
    batched_igd,
    batched_igd_plus,
    compare_against_control,
    compute_dynamic_metrics,
    friedman_nemenyi_test,
    holm_step_down,
    paired_wilcoxon,
    summarize_wtl,
)
from flow_dmo.problems import make_problem


def test_hv_2d_exact_known_areas(device):
    """Spec acceptance assertion: 2-D exact tensor HV error < 1e-6 on known fronts."""
    ref = torch.tensor([1.0, 1.0], device=device)
    # Single point at (0.2, 0.3) -> area = (1 - 0.2) * (1 - 0.3) = 0.56
    F1 = torch.tensor([[[0.2, 0.3]]], device=device)
    hv1 = batched_hv(F1, ref)
    assert abs(float(hv1[0]) - 0.56) < 1e-6

    # Two non-dominated points (0.2, 0.6) and (0.5, 0.2) + dominated (0.7, 0.7) + out-of-box (1.2, 0.1)
    # Exact area = (0.5 - 0.2)*(1.0 - 0.6) + (1.0 - 0.5)*(1.0 - 0.2) = 0.12 + 0.40 = 0.52
    F2 = torch.tensor([[[0.2, 0.6], [0.5, 0.2], [0.7, 0.7], [1.2, 0.1]]], device=device)
    hv2 = batched_hv(F2, ref)
    assert abs(float(hv2[0]) - 0.52) < 1e-6


def test_hv_3d_sobol_matches_exact_cube(device):
    ref = torch.tensor([1.0, 1.0, 1.0], device=device)
    ideal = torch.tensor([0.0, 0.0, 0.0], device=device)
    # Single point at (0.2, 0.3, 0.4) -> exact volume = 0.8 * 0.7 * 0.6 = 0.336
    F = torch.tensor([[[0.2, 0.3, 0.4]]], device=device)
    hv = batched_hv(F, ref, ideal_point=ideal, n_sobol=16384)
    assert abs(float(hv[0]) - 0.336) < 5e-3


def test_igd_and_igd_plus_properties(device):
    PF = torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], device=device)
    # Exact match -> IGD == 0, IGD+ == 0
    F_exact = PF.unsqueeze(0).expand(3, -1, -1)
    assert torch.allclose(batched_igd(F_exact, PF), torch.zeros(3, device=device, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(batched_igd_plus(F_exact, PF), torch.zeros(3, device=device, dtype=torch.float64), atol=1e-12)
    # Uniformly shifted by +0.1 in both objectives -> distance = sqrt(2)*0.1
    F_shift = F_exact + 0.1
    expected = (2.0 ** 0.5) * 0.1
    assert torch.allclose(batched_igd(F_shift, PF), torch.full((3,), expected, device=device, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(batched_igd_plus(F_shift, PF), torch.full((3,), expected, device=device, dtype=torch.float64), atol=1e-6)
    # 4-D batched form (T, B, N, M)
    F4 = F_shift.unsqueeze(0).expand(2, -1, -1, -1)
    PF4 = PF.unsqueeze(0).expand(2, -1, -1)
    assert batched_igd(F4, PF4).shape == (2, 3)


def test_holm_monotonicity_and_wilcoxon_zero_edge_case():
    """Spec acceptance assertions: Holm preserves order monotonicity; all-zero diffs handled."""
    raw = [0.04, 0.001, 0.02, 0.25, 0.015]
    adj = holm_step_down(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adj[order]) >= -1e-12)
    assert np.all((adj >= 0.0) & (adj <= 1.0))

    # All-zero difference edge case
    x = np.ones(30)
    p, r, d = paired_wilcoxon(x, x.copy())
    assert p == 1.0 and r == 0.0 and d == 0

    # Clearly better control (lower is better)
    rng = np.random.default_rng(SEED)
    ctrl = rng.normal(0.05, 0.005, size=30)
    comp_worse = rng.normal(0.15, 0.01, size=30)
    comp_same = ctrl.copy()
    comps = compare_against_control({"FlowMOEA-X": ctrl, "DNSGA2": comp_worse, "TieAlgo": comp_same})
    by_name = {c.competitor: c for c in comps}
    assert by_name["DNSGA2"].symbol == "+" and by_name["DNSGA2"].p_holm < 0.01
    assert by_name["TieAlgo"].symbol == "=" and by_name["TieAlgo"].p_holm == 1.0
    wtl = summarize_wtl(comps)
    assert wtl["DNSGA2"] == {"W": 1, "T": 0, "L": 0}


def test_friedman_and_nemenyi_cd():
    # 6 configs x 4 algorithms with clear ordering A < B < C < D
    mat = np.array([
        [0.01, 0.05, 0.10, 0.20],
        [0.02, 0.06, 0.11, 0.19],
        [0.01, 0.04, 0.09, 0.22],
        [0.03, 0.07, 0.12, 0.25],
        [0.02, 0.05, 0.08, 0.18],
        [0.01, 0.03, 0.07, 0.15],
    ])
    res = friedman_nemenyi_test(mat, ["FlowMOEA-X", "PPS", "LDT_v32", "DNSGA2"])
    assert res.avg_ranks["FlowMOEA-X"] == 1.0
    assert res.avg_ranks["DNSGA2"] == 4.0
    assert res.p_value < 0.01
    assert res.nemenyi_cd > 0.0


@pytest.mark.parametrize("algo_name", ["FlowMOEA-X", "FlowMOEA-X-Pro", "DNSGA2", "LDT_v32", "PPS", "AR", "GMM"])
def test_end_to_end_driver_strict_fe_budget(device, algo_name):
    """All algorithms run end-to-end with exact strict FE budget = n_envs * tau_t * N."""
    B, N, tau_t, n_envs = 3, 24, 4, 4
    prob = make_problem("DF3", n_var=8, tau_t=tau_t, n_t=10, device=device)
    algo = make_algorithm(algo_name, pop_size=N, n_var=8, n_obj=2, device=device)
    driver = BatchedDynamicDriver(
        problem=prob,
        algorithm=algo,
        kernel="nsga2",
        n_seeds=B,
        pop_size=N,
        tau_t=tau_t,
        n_envs=n_envs,
        seed=SEED,
        strict_fe_budget=True,
        n_pf_points=200,
    )
    traj = driver.run()
    assert traj.F_env_end.shape == (n_envs, B, N, 2)
    assert traj.igd_per_gen.shape == (n_envs * tau_t, B)
    assert traj.ledger_summary["total_fe_per_seed"] == n_envs * tau_t * N
    assert traj.ledger_summary["all_envs_exact"] is True

    metrics = compute_dynamic_metrics(traj.F_env_end, traj.F_warmstart, traj.PF_true, traj.igd_per_gen)
    assert torch.isfinite(metrics["migd"]).all()
    assert torch.isfinite(metrics["mhv"]).all()
