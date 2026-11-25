"""Batch 1 acceptance tests: tensor operators, batched NDS, crowding distance,
evolution kernels and the FE ledger."""
from __future__ import annotations

import math

import pytest
import torch
from conftest import SEED, assert_cpu_gpu_consistent, assert_finite_in_unit_box

from flow_dmo.config import ExperimentConfig, make_generator
from flow_dmo.core.dominance import (
    associate_to_reference,
    batched_crowding_distance,
    batched_pareto_ranks,
    das_dennis,
    nondominated_mask,
    uniform_reference_vectors,
)
from flow_dmo.core.fe_ledger import BudgetExceededError, FELedger
from flow_dmo.core.kernels import TensorMOEAD, TensorNSGA2, TensorNSGA3, make_kernel
from flow_dmo.core.tensor_ops import (
    batched_pm,
    batched_sbx,
    denormalize,
    normalize,
    orthogonal_lhs,
    reflect_clamp,
)


# ----------------------------------------------------------------- reference implementations
def reference_ranks(F: torch.Tensor) -> list[int]:
    """Deb's O(MN^2) fast non-dominated sort, written per individual."""
    P = F.shape[0]
    S = [[] for _ in range(P)]
    n = [0] * P
    rank = [-1] * P
    for p in range(P):
        for q in range(P):
            if p == q:
                continue
            fp, fq = F[p], F[q]
            if bool((fp <= fq).all() and (fp < fq).any()):
                S[p].append(q)
            elif bool((fq <= fp).all() and (fq < fp).any()):
                n[p] += 1
    front = [p for p in range(P) if n[p] == 0]
    r = 0
    while front:
        nxt = []
        for p in front:
            rank[p] = r
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        front = nxt
        r += 1
    return rank


def reference_crowding(F: torch.Tensor, ranks: list[int]) -> list[float]:
    P, M = F.shape
    fmin, fmax = F.min(0).values, F.max(0).values
    Fn = (F - fmin) / torch.clamp(fmax - fmin, min=1e-12)
    cd = [0.0] * P
    for r in set(ranks):
        members = [i for i in range(P) if ranks[i] == r]
        for m in range(M):
            srt = sorted(members, key=lambda i: (float(Fn[i, m]), i))
            for pos, i in enumerate(srt):
                if pos == 0 or pos == len(srt) - 1:
                    cd[i] += 1e9
                else:
                    cd[i] += float(Fn[srt[pos + 1], m] - Fn[srt[pos - 1], m])
    return [min(c, 1e9) for c in cd]


def zdt1(Z: torch.Tensor) -> torch.Tensor:
    f1 = Z[..., 0]
    g = 1.0 + 9.0 * Z[..., 1:].mean(dim=-1)
    f2 = g * (1.0 - torch.sqrt(f1 / g))
    return torch.stack([f1, f2], dim=-1)


def dtlz2(Z: torch.Tensor, M: int = 3) -> torch.Tensor:
    g = ((Z[..., M - 1:] - 0.5) ** 2).sum(dim=-1, keepdim=True)
    th = Z[..., : M - 1] * math.pi / 2
    f1 = (1 + g[..., 0]) * torch.cos(th[..., 0]) * torch.cos(th[..., 1])
    f2 = (1 + g[..., 0]) * torch.cos(th[..., 0]) * torch.sin(th[..., 1])
    f3 = (1 + g[..., 0]) * torch.sin(th[..., 0])
    return torch.stack([f1, f2, f3], dim=-1)


# ----------------------------------------------------------------- 1. operators
@pytest.mark.parametrize("B,N,D", [(30, 100, 10), (5, 51, 3), (1, 2, 20)])
def test_sbx_shapes_bounds_finite(device, B, N, D):
    g = make_generator(SEED, device)
    p1 = torch.rand((B, N // 2 if N > 1 else 1, D), device=device, generator=g)
    p2 = torch.rand(p1.shape, device=device, generator=g)
    c1, c2 = batched_sbx(p1, p2, generator=g)
    assert c1.shape == p1.shape and c2.shape == p2.shape
    assert_finite_in_unit_box(c1)
    assert_finite_in_unit_box(c2)


def test_sbx_extreme_parents_stay_finite(device):
    g = make_generator(SEED, device)
    p1 = torch.zeros((4, 50, 6), device=device)
    p2 = torch.ones((4, 50, 6), device=device)
    p1[:, :10] = 1.0
    c1, c2 = batched_sbx(p1, p2, eta_c=2.0, prob_c=1.0, generator=g)
    assert_finite_in_unit_box(c1)
    assert_finite_in_unit_box(c2)
    same = torch.full((4, 50, 6), 0.3, device=device)
    s1, s2 = batched_sbx(same, same.clone(), generator=g)
    assert torch.equal(s1, same) and torch.equal(s2, same)


def test_sbx_preserves_parent_mean(device):
    g = make_generator(SEED, device)
    p1 = torch.full((1, 20000, 1), 0.4, device=device)
    p2 = torch.full((1, 20000, 1), 0.6, device=device)
    c1, c2 = batched_sbx(p1, p2, prob_c=1.0, generator=g)
    mean = torch.cat([c1, c2], dim=1).mean().item()
    assert abs(mean - 0.5) < 5e-3


@pytest.mark.parametrize("B,N,D", [(30, 100, 10), (3, 7, 2)])
def test_pm_shapes_bounds_finite(device, B, N, D):
    g = make_generator(SEED, device)
    Z = torch.rand((B, N, D), device=device, generator=g)
    Z[:, 0] = 0.0
    Z[:, 1] = 1.0
    out = batched_pm(Z, generator=g)
    assert out.shape == (B, N, D)
    assert_finite_in_unit_box(out)
    out_all = batched_pm(Z, prob_m=1.0, generator=g)
    assert_finite_in_unit_box(out_all)
    assert (out_all != Z).float().mean() > 0.9


def test_pm_default_rate_is_one_over_d(device):
    g = make_generator(SEED, device)
    Z = torch.full((10, 1000, 10), 0.5, device=device)
    out = batched_pm(Z, generator=g)
    rate = (out != Z).float().mean().item()
    assert abs(rate - 0.1) < 0.01


def test_normalize_denormalize_roundtrip(device):
    xl = torch.tensor([-1.0, 0.0, 2.0], device=device).view(1, 1, 3)
    xu = torch.tensor([1.0, 4.0, 3.0], device=device).view(1, 1, 3)
    g = make_generator(SEED, device)
    Z = torch.rand((4, 9, 3), device=device, generator=g)
    X = denormalize(Z, xl, xu)
    assert torch.all(X >= xl) and torch.all(X <= xu)
    assert torch.allclose(normalize(X, xl, xu), Z, atol=1e-6)
    assert torch.all(normalize(X + 100.0, xl, xu) <= 1.0)


def test_reflect_clamp_and_lhs(device):
    Z = torch.tensor([[[-0.2, 1.3, 0.5, -1.5, 2.7]]], device=device)
    R = reflect_clamp(Z)
    assert torch.allclose(R, torch.tensor([[[0.2, 0.7, 0.5, 1.0, 0.0]]], device=device), atol=1e-6)
    g = make_generator(SEED, device)
    L = orthogonal_lhs(3, 20, 4, device=device, generator=g)
    assert L.shape == (3, 20, 4)
    assert_finite_in_unit_box(L)
    strata = torch.floor(L * 20).long()
    for d in range(4):  # every stratum appears exactly once per dimension
        assert torch.equal(torch.sort(strata[0, :, d]).values, torch.arange(20, device=device))


# ----------------------------------------------------------------- 2. NDS / crowding
HANDMADE_3D = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 3.0],
    [3.0, 3.0, 3.0],   # dominated by 0 and 1
    [1.0, 2.0, 3.0],   # duplicate of 0 (mutually non-dominating)
    [0.5, 4.0, 4.0],
    [4.0, 4.0, 4.0],   # dominated by 2
    [5.0, 5.0, 5.0],   # dominated by 5
    [0.0, 0.0, 9.0],
    [2.0, 2.0, 2.0],
    [6.0, 0.5, 6.0],
])
HANDMADE_EXPECTED = [0, 0, 1, 0, 0, 2, 3, 0, 0, 0]


def test_pareto_ranks_handmade_3d():
    ranks = batched_pareto_ranks(HANDMADE_3D.unsqueeze(0))[0].tolist()
    assert ranks == HANDMADE_EXPECTED
    assert ranks == reference_ranks(HANDMADE_3D)
    nd = nondominated_mask(HANDMADE_3D).tolist()
    assert nd == [r == 0 for r in HANDMADE_EXPECTED]


@pytest.mark.parametrize("M", [2, 3, 5])
def test_pareto_ranks_match_reference_random(M):
    g = make_generator(SEED + M, "cpu")
    F = torch.rand((6, 60, M), generator=g)
    F[:, :10] = torch.round(F[:, :10] * 4) / 4      # inject ties / duplicates
    ranks = batched_pareto_ranks(F)
    for b in range(F.shape[0]):
        assert ranks[b].tolist() == reference_ranks(F[b])


def test_pareto_ranks_cpu_gpu_consistent():
    g = make_generator(SEED, "cpu")
    F = torch.rand((5, 80, 3), generator=g)
    assert_cpu_gpu_consistent(lambda dev, X: batched_pareto_ranks(X.to(dev)).double(), F)


@pytest.mark.parametrize("M", [2, 3])
def test_crowding_distance_matches_reference(M):
    g = make_generator(SEED, "cpu")
    F = torch.rand((4, 40, M), generator=g, dtype=torch.float64)
    ranks = batched_pareto_ranks(F)
    cd = batched_crowding_distance(F, ranks)
    assert cd.shape == (4, 40)
    for b in range(4):
        ref = torch.tensor(reference_crowding(F[b], ranks[b].tolist()), dtype=torch.float64)
        assert torch.allclose(cd[b], ref, atol=1e-9)


def test_crowding_boundaries_are_infinite():
    F = torch.tensor([[[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [1.0, 0.0]]])
    ranks = batched_pareto_ranks(F)
    cd = batched_crowding_distance(F, ranks)[0]
    assert cd[0] >= 1e9 - 1 and cd[3] >= 1e9 - 1
    assert torch.allclose(cd[1:3], torch.tensor([1.0, 1.5]), atol=1e-6)


def test_reference_vectors():
    W = das_dennis(12, 3)
    assert W.shape == (91, 3)
    assert torch.allclose(W.sum(1), torch.ones(91, dtype=torch.float64))
    for n, M in [(100, 2), (150, 3), (60, 5)]:
        U = uniform_reference_vectors(n, M)
        assert U.shape == (n, M)
        assert torch.allclose(U.sum(1), torch.ones(n), atol=1e-6)
        assert torch.unique(U, dim=0).shape[0] == n
    F = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.2, 0.2]]])
    Wc = torch.eye(3)
    assoc, dist = associate_to_reference(torch.cat([F], 0), torch.cat([Wc, torch.ones(1, 3) / 3]))
    assert assoc[0].tolist() == [0, 1, 3]
    assert dist[0, 2].item() < 1e-6


# ----------------------------------------------------------------- 3. kernels
def _run_kernel(kernel, fn, D, steps, device, seed=SEED):
    g = make_generator(seed, device)
    Z = torch.rand((4, kernel.pop_size, D), device=device, generator=g)
    F = fn(Z)
    for _ in range(steps):
        Z, F = kernel.step_generation(Z, F, fn, g)
    return Z, F


def test_nsga2_step_is_deterministic(device):
    k = TensorNSGA2(pop_size=40, n_obj=2)
    Z1, F1 = _run_kernel(k, zdt1, 8, 5, device)
    Z2, F2 = _run_kernel(k, zdt1, 8, 5, device)
    max_diff = max((Z1 - Z2).abs().max().item(), (F1 - F2).abs().max().item())
    assert max_diff == 0.0


@pytest.mark.parametrize("kernel_name,fn,M", [("nsga2", zdt1, 2), ("nsga3", dtlz2, 3), ("moead", zdt1, 2)])
def test_kernels_shapes_determinism_and_progress(device, kernel_name, fn, M):
    k = make_kernel(kernel_name, pop_size=36, n_obj=M)
    Z0g = make_generator(SEED, device)
    Z0 = torch.rand((4, 36, 8), device=device, generator=Z0g)
    F0 = fn(Z0)
    Z1, F1 = _run_kernel(k, fn, 8, 25, device)
    Z2, F2 = _run_kernel(k, fn, 8, 25, device)
    assert Z1.shape == (4, 36, 8) and F1.shape == (4, 36, M)
    assert_finite_in_unit_box(Z1)
    assert (Z1 - Z2).abs().max().item() == 0.0 and (F1 - F2).abs().max().item() == 0.0
    # convergence: distance-variable error must drop
    if kernel_name == "nsga3":
        g0 = ((Z0[..., M - 1:] - 0.5) ** 2).sum(-1).mean()
        g1 = ((Z1[..., M - 1:] - 0.5) ** 2).sum(-1).mean()
    else:
        g0 = Z0[..., 1:].mean()
        g1 = Z1[..., 1:].mean()
    assert g1 < 0.5 * g0


def test_kernel_classes_exposed():
    assert isinstance(make_kernel("NSGA-II", 10, 2), TensorNSGA2)
    assert isinstance(make_kernel("nsga3", 10, 3), TensorNSGA3)
    assert isinstance(make_kernel("MOEA/D", 10, 2), TensorMOEAD)
    with pytest.raises(ValueError):
        make_kernel("spea2", 10, 2)


# ----------------------------------------------------------------- 4. FE ledger
class _ToyProblem:
    n_obj = 2
    n_var = 3

    def __init__(self, device):
        self.xl = torch.full((1, 1, 3), -1.0, device=device)
        self.xu = torch.full((1, 1, 3), 1.0, device=device)

    def evaluate_batch(self, X, t):
        return torch.stack([X[..., 0] + t, (X ** 2).sum(-1)], dim=-1)


def test_fe_ledger_accounting_and_strictness(device):
    prob = _ToyProblem(device)
    led = FELedger(n_seeds=3, pop_size=10, tau_t=2, device=device, strict_fe_budget=True)
    led.start_environment(0, 0.0)
    Z = torch.full((3, 10, 3), 0.5, device=device)
    F = led.evaluate_and_charge(prob, Z, 0.0, "warmstart")
    assert F.shape == (3, 10, 2)
    assert torch.allclose(F[..., 1], torch.zeros(3, 10, device=device))   # Z=0.5 -> X=0
    assert led.remaining_generations_in_env() == 1
    led.evaluate_and_charge(prob, Z[:, :4], 0.0, "sensor")
    led.evaluate_and_charge(prob, Z[:, :6], 0.0, "warmstart")
    assert led.remaining_generations_in_env() == 0
    with pytest.raises(BudgetExceededError):
        led.evaluate_and_charge(prob, Z[:, :1], 0.0, "evolution")
    assert led.total_fe_per_seed.tolist() == [20, 20, 20]
    assert led.all_envs_exact()
    roles = led.role_totals()
    assert roles == {"sensor": 4, "warmstart": 16, "evolution": 0}
    empty = led.evaluate_and_charge(prob, Z[:, :0], 0.0, "evolution")
    assert empty.shape == (3, 0, 2)
    with pytest.raises(ValueError):
        led.charge(1, "probe")


def test_config_yaml_roundtrip(tmp_path):
    cfg = ExperimentConfig(n_seeds=7, problems=["DF3"], kernel="nsga3")
    cfg.transport.mode = "pro"
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    back = ExperimentConfig.from_yaml(path)
    assert back == cfg
    assert back.pop_size_for(3) == 150 and back.pop_size_for(2) == 100
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"bogus": 1})
