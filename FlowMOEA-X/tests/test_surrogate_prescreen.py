"""Batch 3 acceptance tests (part B): KRR closed-form pre-screener latency and
strict zero-FE leakage guarantee."""
from __future__ import annotations

import time

import torch
from conftest import SEED, assert_finite_in_unit_box

from flow_dmo.config import make_generator
from flow_dmo.core.fe_ledger import FELedger
from flow_dmo.problems import make_problem
from flow_dmo.surrogate import KRRPrescreener


def test_krr_prescreener_latency_and_accuracy(device):
    """Spec acceptance assertion:
    B=30, N_train=50, N_cand=200 single batched call < 15 ms (CPU) or < 2 ms (GPU).
    """
    B, N_train, N_cand, D = 30, 50, 200, 10
    prob = make_problem("DF1", n_var=D, device=device)
    g = make_generator(SEED, device)

    Z_train = torch.rand((B, N_train, D), device=device, generator=g)
    X_train = prob.xl + Z_train * (prob.xu - prob.xl)
    F_train = prob.evaluate_batch(X_train, t=0.4)

    Z_cand = torch.rand((B, N_cand, D), device=device, generator=g)
    prescreener = KRRPrescreener(reg_lambda=1e-3, lcb_beta=0.2)

    # Warm-up call
    _ = prescreener.select_best_candidates(Z_train, F_train, Z_cand, select_count=50)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        Z_sel = prescreener.select_best_candidates(Z_train, F_train, Z_cand, select_count=50)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    median_ms = sorted(times)[len(times) // 2]
    limit_ms = 2.0 if device.type == "cuda" else 15.0
    assert Z_sel.shape == (B, 50, D)
    assert_finite_in_unit_box(Z_sel)
    assert median_ms < limit_ms, f"KRR prescreen took {median_ms:.2f} ms (limit {limit_ms} ms)"

    # Verify selected candidates have better mean f2 than full pool
    F_cand = prob.evaluate_batch(prob.xl + Z_cand * (prob.xu - prob.xl), t=0.4)
    F_sel = prob.evaluate_batch(prob.xl + Z_sel * (prob.xu - prob.xl), t=0.4)
    assert float(F_sel[..., 1].mean()) < float(F_cand[..., 1].mean())


def test_krr_prescreener_zero_fe_leak(device):
    """Spec acceptance assertion: FELedger.total_fe_per_seed increment is strictly 0."""
    B, N_train, N_cand, D = 5, 30, 120, 10
    prob = make_problem("DF3", n_var=D, device=device)
    ledger = FELedger(n_seeds=B, pop_size=60, tau_t=5, device=device)
    ledger.start_environment(0, 0.2)

    g = make_generator(SEED, device)
    Z_train = torch.rand((B, N_train, D), device=device, generator=g)
    F_train = ledger.evaluate_and_charge(prob, Z_train, 0.2, "warmstart")

    fe_before = ledger.total_fe_per_seed.clone()
    calls_before = ledger.n_problem_calls

    Z_cand = torch.rand((B, N_cand, D), device=device, generator=g)
    Z_sel = KRRPrescreener().select_best_candidates(Z_train, F_train, Z_cand, select_count=30)

    assert Z_sel.shape == (B, 30, D)
    assert torch.equal(ledger.total_fe_per_seed, fe_before)
    assert ledger.n_problem_calls == calls_before
