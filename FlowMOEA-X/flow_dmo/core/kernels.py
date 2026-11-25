"""Pluggable fully-tensorised evolution kernels (Section 3.2.3).

All kernels share ``step_generation(Z_pop, F_pop, evaluate_fn, generator)``:
they create exactly ``N`` offspring per generation (one real FE per
offspring, charged by ``evaluate_fn``) and return the next ``(Z, F)``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from flow_dmo.core.dominance import (
    associate_to_reference,
    batched_crowding_distance,
    batched_gather,
    batched_pareto_ranks,
    nsga2_select,
    nsga2_sort_key,
    uniform_reference_vectors,
)
from flow_dmo.core.tensor_ops import batched_pm, batched_sbx

Tensor = torch.Tensor
EvaluateFn = Callable[[Tensor], Tensor]


def within_group_rank(group: Tensor, key: Tensor, valid: Tensor) -> Tensor:
    """Rank (0, 1, 2, ...) of each element inside its group, ordered by ``key``.

    ``group`` (long), ``key`` (float), ``valid`` (bool) all ``(B, P)``. Invalid
    elements get a large rank. Fully vectorised via a single argsort.
    """
    B, P = group.shape
    n_groups = int(group.max().item()) + 2 if group.numel() else 1
    k = key.to(torch.float64)
    k = (k - k.min()) / (k.max() - k.min() + 1e-12) if k.numel() else k
    composite = torch.where(valid, group.to(torch.float64) * 2.0 + k,
                            torch.full_like(k, float(n_groups) * 2.0 + 1.0))
    order = torch.argsort(composite, dim=1, stable=True)
    g_sorted = torch.gather(torch.where(valid, group, torch.full_like(group, n_groups)), 1, order)
    pos = torch.arange(P, device=group.device).expand(B, P)
    is_start = torch.ones_like(g_sorted, dtype=torch.bool)
    is_start[:, 1:] = g_sorted[:, 1:] != g_sorted[:, :-1]
    start = torch.where(is_start, pos, torch.zeros_like(pos)).cummax(dim=1).values
    within_sorted = pos - start
    within = torch.empty_like(within_sorted)
    within.scatter_(1, order, within_sorted)
    return torch.where(valid, within, torch.full_like(within, P + 1))


class EvolutionKernel(ABC):
    """Common variation machinery (tournament + SBX + PM)."""

    name = "base"

    def __init__(self, pop_size: int, n_obj: int, eta_c: float = 15.0, eta_m: float = 20.0,
                 prob_c: float = 0.9, prob_m: float | None = None):
        self.pop_size = int(pop_size)
        self.n_obj = int(n_obj)
        self.eta_c = float(eta_c)
        self.eta_m = float(eta_m)
        self.prob_c = float(prob_c)
        self.prob_m = prob_m

    # ------------------------------------------------------------- variation
    def _binary_tournament(self, key: Tensor, n: int, generator: torch.Generator | None) -> Tensor:
        B, P = key.shape
        a = torch.randint(0, P, (B, n), device=key.device, generator=generator)
        b = torch.randint(0, P, (B, n), device=key.device, generator=generator)
        ka = torch.gather(key, 1, a)
        kb = torch.gather(key, 1, b)
        return torch.where(ka <= kb, a, b)

    def make_offspring(self, Z_pop: Tensor, mating_key: Tensor | None,
                       generator: torch.Generator | None, n_off: int | None = None) -> Tensor:
        """Create ``n_off`` (default ``N``) offspring ``(B, n_off, D)``."""
        B, P, D = Z_pop.shape
        n_off = self.pop_size if n_off is None else int(n_off)
        n_pairs = (n_off + 1) // 2
        if mating_key is None:
            i1 = torch.randint(0, P, (B, n_pairs), device=Z_pop.device, generator=generator)
            i2 = torch.randint(0, P, (B, n_pairs), device=Z_pop.device, generator=generator)
        else:
            i1 = self._binary_tournament(mating_key, n_pairs, generator)
            i2 = self._binary_tournament(mating_key, n_pairs, generator)
        p1 = batched_gather(Z_pop, i1)
        p2 = batched_gather(Z_pop, i2)
        c1, c2 = batched_sbx(p1, p2, self.eta_c, self.prob_c, generator)
        off = torch.cat([c1, c2], dim=1)[:, :n_off]
        return batched_pm(off, self.eta_m, self.prob_m, generator)

    # ------------------------------------------------------------- selection
    @abstractmethod
    def environmental_selection(self, Z: Tensor, F: Tensor, n_select: int,
                                generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Reduce a merged population ``(B, P, ·)`` to ``n_select`` individuals."""

    def mating_key(self, F_pop: Tensor) -> Tensor | None:
        ranks = batched_pareto_ranks(F_pop)
        cd = batched_crowding_distance(F_pop, ranks)
        return nsga2_sort_key(ranks, cd)

    def step_generation(self, Z_pop: Tensor, F_pop: Tensor, evaluate_fn: EvaluateFn,
                        generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        Z_off = self.make_offspring(Z_pop, self.mating_key(F_pop), generator)
        F_off = evaluate_fn(Z_off)
        Z_all = torch.cat([Z_pop, Z_off], dim=1)
        F_all = torch.cat([F_pop, F_off], dim=1)
        return self.environmental_selection(Z_all, F_all, self.pop_size, generator)


class TensorNSGA2(EvolutionKernel):
    """NSGA-II: (rank, crowding) elitist truncation."""

    name = "nsga2"

    def environmental_selection(self, Z, F, n_select, generator=None):
        idx, _, _ = nsga2_select(F, n_select)
        return batched_gather(Z, idx), batched_gather(F, idx)


class TensorNSGA3(EvolutionKernel):
    """NSGA-III with Das-Dennis reference vectors and vectorised niching.

    The iterative niche-filling loop of the original algorithm is replaced by
    an equivalent one-shot priority: a last-front candidate that is the
    ``q``-th closest member of reference niche ``j`` receives priority
    ``rho_j + q`` (``rho_j`` = niche count of already-selected members), i.e.
    niches are filled round-robin from the least crowded, closest first.
    """

    name = "nsga3"

    def __init__(self, pop_size: int, n_obj: int, n_ref: int | None = None, **kw):
        super().__init__(pop_size, n_obj, **kw)
        self.n_ref = int(n_ref or pop_size)
        self._W_cache: dict[tuple, Tensor] = {}

    def reference_vectors(self, device, dtype) -> Tensor:
        key = (str(device), dtype)
        if key not in self._W_cache:
            self._W_cache[key] = uniform_reference_vectors(self.n_ref, self.n_obj, device=device, dtype=dtype)
        return self._W_cache[key]

    def mating_key(self, F_pop):
        return None  # NSGA-III uses random mating selection

    def environmental_selection(self, Z, F, n_select, generator=None):
        B, P, M = F.shape
        ranks = batched_pareto_ranks(F)
        sorted_ranks = torch.sort(ranks, dim=1).values
        L = sorted_ranks[:, n_select - 1: n_select]                 # (B, 1) last admitted front
        sel = ranks < L
        cand = ranks == L

        considered = ranks <= L
        big = torch.finfo(F.dtype).max
        ideal = torch.where(considered.unsqueeze(-1), F, torch.full_like(F, big)).min(dim=1, keepdim=True).values
        front0 = (ranks == 0).unsqueeze(-1)
        nadir = torch.where(front0, F, torch.full_like(F, -big)).max(dim=1, keepdim=True).values
        span = nadir - ideal
        span = torch.where(span > 1e-10, span, torch.ones_like(span))
        F_norm = (F - ideal) / span

        W = self.reference_vectors(F.device, F.dtype)
        assoc, dist = associate_to_reference(F_norm, W)
        rho = torch.zeros((B, W.shape[0]), device=F.device, dtype=torch.float64)
        rho.scatter_add_(1, assoc, sel.to(torch.float64))
        within = within_group_rank(assoc, dist, cand)
        dist64 = dist.to(torch.float64)
        dist64 = dist64 / (dist64.max() + 1e-12)
        priority_cand = torch.gather(rho, 1, assoc) + within.to(torch.float64) + 1e-3 * dist64
        priority = torch.where(sel, torch.full_like(priority_cand, -1.0),
                               torch.where(cand, priority_cand, torch.full_like(priority_cand, float("inf"))))
        idx = torch.argsort(priority, dim=1, stable=True)[:, :n_select]
        return batched_gather(Z, idx), batched_gather(F, idx)


class TensorMOEAD(EvolutionKernel):
    """Parallel MOEA/D (Tchebycheff or PBI) with neighbourhood replacement.

    All ``N`` sub-problems create one child simultaneously; each sub-problem
    ``j`` is then updated by the best child generated inside its
    neighbourhood, and each child may replace at most ``nr`` solutions.
    """

    name = "moead"

    def __init__(self, pop_size: int, n_obj: int, n_neighbors: int = 20, delta: float = 0.9,
                 nr: int = 2, decomposition: str = "tchebycheff", pbi_theta: float = 5.0, **kw):
        super().__init__(pop_size, n_obj, **kw)
        self.T = int(min(n_neighbors, pop_size))
        self.delta = float(delta)
        self.nr = int(nr)
        self.decomposition = decomposition
        self.pbi_theta = float(pbi_theta)
        self._cache: dict[tuple, tuple[Tensor, Tensor]] = {}

    def weights_and_neighbors(self, device, dtype) -> tuple[Tensor, Tensor]:
        key = (str(device), dtype)
        if key not in self._cache:
            W = uniform_reference_vectors(self.pop_size, self.n_obj, device=device, dtype=dtype)
            nb = torch.argsort(torch.cdist(W, W), dim=1, stable=True)[:, : self.T]
            self._cache[key] = (W, nb)
        return self._cache[key]

    def scalarize(self, F: Tensor, W: Tensor, z: Tensor) -> Tensor:
        """``F (..., M)``, ``W`` broadcastable ``(..., M)``, ``z (B, 1, M)`` style."""
        if self.decomposition == "pbi":
            Wn = W / torch.clamp(W.norm(dim=-1, keepdim=True), min=1e-12)
            diff = F - z
            d1 = (diff * Wn).sum(dim=-1)
            d2 = torch.norm(diff - d1.unsqueeze(-1) * Wn, dim=-1)
            return d1 + self.pbi_theta * d2
        Wc = torch.clamp(W, min=1e-6)
        return (Wc * torch.abs(F - z)).max(dim=-1).values

    def environmental_selection(self, Z, F, n_select, generator=None):
        idx, _, _ = nsga2_select(F, n_select)
        return batched_gather(Z, idx), batched_gather(F, idx)

    def step_generation(self, Z_pop, F_pop, evaluate_fn, generator=None):
        B, N, D = Z_pop.shape
        if N != self.pop_size:
            raise ValueError("MOEA/D population size must equal the number of weight vectors")
        dev = Z_pop.device
        W, nb = self.weights_and_neighbors(dev, Z_pop.dtype)

        # ---- mating: neighbourhood (prob delta) or whole population
        use_nb = torch.rand((B, N, 1), device=dev, generator=generator) < self.delta
        pick = torch.randint(0, self.T, (B, N, 2), device=dev, generator=generator)
        nb_idx = torch.gather(nb.unsqueeze(0).expand(B, -1, -1), 2, pick)       # (B, N, 2)
        glob_idx = torch.randint(0, N, (B, N, 2), device=dev, generator=generator)
        parent_idx = torch.where(use_nb, nb_idx, glob_idx)
        p1 = batched_gather(Z_pop, parent_idx[..., 0])
        p2 = batched_gather(Z_pop, parent_idx[..., 1])
        c1, _ = batched_sbx(p1, p2, self.eta_c, self.prob_c, generator)
        Z_child = batched_pm(c1, self.eta_m, self.prob_m, generator)
        F_child = evaluate_fn(Z_child)

        # ---- ideal point
        z = torch.minimum(F_pop.min(dim=1, keepdim=True).values, F_child.min(dim=1, keepdim=True).values)

        # ---- neighbourhood replacement (best child inside B(j) for every j)
        flat = nb.reshape(-1)
        cand_F = F_child[:, flat].reshape(B, N, self.T, -1)                        # (B, N, T, M)
        g_cand = self.scalarize(cand_F, W.unsqueeze(1), z.unsqueeze(1))           # (B, N, T)
        g_best, arg = g_cand.min(dim=-1)                                          # (B, N)
        child_id = torch.gather(nb.unsqueeze(0).expand(B, -1, -1), 2, arg.unsqueeze(-1)).squeeze(-1)
        g_cur = self.scalarize(F_pop, W.unsqueeze(0), z)                          # (B, N)
        improve = g_cur - g_best
        valid = improve > 0
        rank_in_child = within_group_rank(child_id, -improve, valid)
        replace = valid & (rank_in_child < self.nr)
        Z_new = torch.where(replace.unsqueeze(-1), batched_gather(Z_child, child_id), Z_pop)
        F_new = torch.where(replace.unsqueeze(-1), batched_gather(F_child, child_id), F_pop)
        return Z_new, F_new


_KERNELS = {"nsga2": TensorNSGA2, "nsga3": TensorNSGA3, "moead": TensorMOEAD}


def make_kernel(name: str, pop_size: int, n_obj: int, **kw) -> EvolutionKernel:
    key = name.lower().replace("-", "").replace("/", "").replace("_", "")
    alias = {"nsgaii": "nsga2", "nsgaiii": "nsga3", "moead": "moead"}
    key = alias.get(key, key)
    if key not in _KERNELS:
        raise ValueError(f"Unknown kernel {name!r}; choose from {sorted(_KERNELS)}")
    return _KERNELS[key](pop_size, n_obj, **kw)
