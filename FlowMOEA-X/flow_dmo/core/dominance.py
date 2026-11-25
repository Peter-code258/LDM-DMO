"""Batched (3-D) fast non-dominated sorting, crowding distance and reference
vectors (Section 3.2). No Python loop over individuals is used anywhere.
"""
from __future__ import annotations

import itertools
import math

import torch

Tensor = torch.Tensor

INF_CD = 1e9


def batched_gather(X: Tensor, idx: Tensor) -> Tensor:
    """Gather rows: ``X (B, P, K)``, ``idx (B, Q)`` -> ``(B, Q, K)``."""
    return torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, X.shape[-1]))


def dominance_matrix(F: Tensor) -> Tensor:
    """``D[b, i, j] = True`` iff individual ``i`` Pareto-dominates ``j`` in seed ``b``."""
    M = F.shape[-1]
    if M == 2:
        f0, f1 = F[..., 0], F[..., 1]
        f0_i, f0_j = f0.unsqueeze(2), f0.unsqueeze(1)
        f1_i, f1_j = f1.unsqueeze(2), f1.unsqueeze(1)
        return (f0_i <= f0_j) & (f1_i <= f1_j) & ((f0_i < f0_j) | (f1_i < f1_j))
    F_i = F.unsqueeze(2)   # (B, P, 1, M)
    F_j = F.unsqueeze(1)   # (B, 1, P, M)
    return (F_i <= F_j).all(dim=-1) & (F_i < F_j).any(dim=-1)


def batched_pareto_ranks(F: Tensor, stop_after: int | None = None) -> Tensor:
    """Front index (0 = non-dominated) for every individual; ``F (B, P, M)`` -> ``(B, P)``.

    If ``stop_after`` is given, front peeling stops as soon as every seed has
    at least ``stop_after`` ranked individuals (remaining individuals receive
    ``current_rank``, preserving the exact top-``stop_after`` ordering).
    """
    if F.dim() == 2:
        return batched_pareto_ranks(F.unsqueeze(0), stop_after=stop_after)[0]
    B, P, _ = F.shape
    D_mat = dominance_matrix(F)
    dom_count = D_mat.sum(dim=1)                       # times each j is dominated
    ranks = torch.full((B, P), -1, dtype=torch.long, device=F.device)
    assigned = torch.zeros(B, dtype=torch.long, device=F.device)
    current_rank = 0
    while bool((ranks == -1).any()):
        is_front = (ranks == -1) & (dom_count == 0)
        if not bool(is_front.any()):
            ranks[ranks == -1] = current_rank          # numerical safety net
            break
        ranks = torch.where(is_front, torch.full_like(ranks, current_rank), ranks)
        current_rank += 1
        if stop_after is not None:
            assigned = assigned + is_front.sum(dim=1)
            if bool((assigned >= stop_after).all()):
                ranks[ranks == -1] = current_rank
                break
        removed_dom = (D_mat & is_front.unsqueeze(2)).sum(dim=1)
        dom_count = dom_count - removed_dom
        # members of the current front must never be re-selected
        dom_count = torch.where(is_front, torch.full_like(dom_count, P + 1), dom_count)
    return ranks


def nondominated_mask(F: Tensor) -> Tensor:
    """Boolean mask of first-front members, ``(B, P)``."""
    if F.dim() == 2:
        return nondominated_mask(F.unsqueeze(0))[0]
    return dominance_matrix(F).sum(dim=1) == 0


def batched_crowding_distance(F: Tensor, ranks: Tensor) -> Tensor:
    """Per-front crowding distance, ``(B, P)``; boundary members get ``1e9``.

    A composite key ``rank * 10 + F_norm`` (``F_norm`` in ``[0, 1]``) lets a
    single ``argsort`` along ``dim=1`` sort every front of every seed by every
    objective simultaneously without mixing fronts.
    """
    B, P, M = F.shape
    if P <= 2:
        return torch.full((B, P), INF_CD, device=F.device, dtype=F.dtype)
    f_min = F.min(dim=1, keepdim=True).values
    f_max = F.max(dim=1, keepdim=True).values
    F_norm = (F - f_min) / torch.clamp(f_max - f_min, min=1e-12)
    key = ranks.unsqueeze(-1).to(F.dtype) * 10.0 + F_norm               # (B, P, M)
    order = torch.argsort(key, dim=1, stable=True)                         # (B, P, M)
    F_sorted = torch.gather(F_norm, 1, order)
    R_sorted = torch.gather(ranks.unsqueeze(-1).expand(-1, -1, M), 1, order)

    contrib = torch.zeros((B, P, M), device=F.device, dtype=F.dtype)
    contrib[:, 1:-1, :] = F_sorted[:, 2:, :] - F_sorted[:, :-2, :]
    is_first = torch.ones((B, P, M), dtype=torch.bool, device=F.device)
    is_first[:, 1:, :] = R_sorted[:, 1:, :] != R_sorted[:, :-1, :]
    is_last = torch.ones((B, P, M), dtype=torch.bool, device=F.device)
    is_last[:, :-1, :] = R_sorted[:, :-1, :] != R_sorted[:, 1:, :]
    contrib = torch.where(is_first | is_last, torch.full_like(contrib, INF_CD), contrib)

    cd = torch.zeros((B, P), device=F.device, dtype=F.dtype)
    for m in range(M):  # loop over objectives only (M <= 5), never over individuals
        cd.scatter_add_(1, order[:, :, m], contrib[:, :, m])
    return torch.clamp(cd, max=INF_CD)


def nsga2_sort_key(ranks: Tensor, crowding: Tensor) -> Tensor:
    """Ascending key implementing the crowded-comparison order (rank, -cd)."""
    cd = crowding.to(torch.float64)
    frac = cd / (1.0 + cd)                       # in [0, 1)
    return ranks.to(torch.float64) - 0.999 * frac


def nsga2_select(F: Tensor, n_select: int) -> tuple[Tensor, Tensor, Tensor]:
    """Indices of the best ``n_select`` individuals under the NSGA-II order.

    Returns ``(idx (B, n_select), ranks (B, P), crowding (B, P))``.
    """
    ranks = batched_pareto_ranks(F, stop_after=n_select)
    cd = batched_crowding_distance(F, ranks)
    order = torch.argsort(nsga2_sort_key(ranks, cd), dim=1, stable=True)
    return order[:, :n_select], ranks, cd


# --------------------------------------------------------------------------- reference vectors
def das_dennis(n_partitions: int, n_obj: int) -> Tensor:
    """Das-Dennis simplex-lattice weights (float64), shape ``(C(H+M-1, M-1), M)``."""
    if n_obj == 1:
        return torch.ones((1, 1), dtype=torch.float64)
    combos = itertools.combinations(range(n_partitions + n_obj - 1), n_obj - 1)
    rows = []
    for c in combos:
        prev = -1
        parts = []
        for pos in c:
            parts.append(pos - prev - 1)
            prev = pos
        parts.append(n_partitions + n_obj - 2 - prev)
        rows.append(parts)
    W = torch.tensor(rows, dtype=torch.float64) / float(n_partitions)
    return W


def farthest_point_sampling(points: Tensor, n: int, start: int = 0) -> Tensor:
    """Deterministic farthest-point subsampling of ``points (P, K)`` -> indices ``(n,)``."""
    P = points.shape[0]
    if n >= P:
        return torch.arange(P, device=points.device)
    chosen = torch.empty(n, dtype=torch.long, device=points.device)
    chosen[0] = start
    dist = torch.cdist(points, points[start:start + 1]).squeeze(1)
    for i in range(1, n):
        nxt = torch.argmax(dist)
        chosen[i] = nxt
        dist = torch.minimum(dist, torch.cdist(points, points[nxt:nxt + 1]).squeeze(1))
    return chosen


def uniform_reference_vectors(n: int, n_obj: int, device=None, dtype=torch.float32) -> Tensor:
    """Exactly ``n`` near-uniform weight vectors on the unit simplex.

    ``M = 2`` uses an exact linspace; otherwise a Das-Dennis lattice with at
    least ``n`` points is generated and reduced deterministically with
    farthest-point sampling (corners always kept).
    """
    if n_obj == 2:
        w1 = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        W = torch.stack([w1, 1.0 - w1], dim=1)
    else:
        H = 1
        while math.comb(H + n_obj - 1, n_obj - 1) < n:
            H += 1
        W = das_dennis(H, n_obj)
        if W.shape[0] > n:
            corner = int(torch.argmax(W[:, 0]))
            W = W[farthest_point_sampling(W, n, start=corner)]
    return W.to(device=device, dtype=dtype)


def associate_to_reference(F_norm: Tensor, W: Tensor) -> tuple[Tensor, Tensor]:
    """Associate normalised objectives to reference lines.

    ``F_norm (B, P, M)``, ``W (R, M)`` -> ``(assoc (B, P) long, perp_dist (B, P))``.
    """
    Wn = W / torch.clamp(W.norm(dim=1, keepdim=True), min=1e-12)          # (R, M)
    proj = torch.einsum("bpm,rm->bpr", F_norm, Wn)                          # (B, P, R)
    diff = F_norm.unsqueeze(2) - proj.unsqueeze(-1) * Wn.view(1, 1, -1, W.shape[-1])
    perp = torch.linalg.norm(diff, dim=-1)                                  # (B, P, R)
    dist, assoc = perp.min(dim=-1)
    return assoc, dist
