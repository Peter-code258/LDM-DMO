"""GPU/CPU batched evaluation indicators: IGD, IGD+, exact 2-D HV, QMC Sobol
high-dimensional HV, MIGD, MHV, WSRR and AUC-IGD (Section 6.1).

All calculations promote to ``torch.float64`` for numerical accuracy.
"""
from __future__ import annotations

import torch

Tensor = torch.Tensor


def batched_igd(F: Tensor, PF_true: Tensor) -> Tensor:
    """Inverted Generational Distance.

    Supports ``F: (B, N, M)`` with ``PF_true: (P, M)`` -> ``(B,)``, or
    ``F: (T, B, N, M)`` with ``PF_true: (T, P, M)`` -> ``(T, B)``.
    """
    F64 = F.to(torch.float64)
    P64 = PF_true.to(device=F.device, dtype=torch.float64)
    if F64.dim() == 3:
        P_exp = P64.unsqueeze(0).expand(F64.shape[0], -1, -1)
        d = torch.cdist(P_exp, F64)                                        # (B, P, N)
        return d.min(dim=-1).values.mean(dim=-1)
    if F64.dim() == 4:
        T, B, N, M = F64.shape
        P_exp = P64.unsqueeze(1).expand(T, B, -1, M)
        d = torch.cdist(P_exp, F64)                                        # (T, B, P, N)
        return d.min(dim=-1).values.mean(dim=-1)
    raise ValueError(f"Expected 3-D or 4-D F tensor, got shape {tuple(F.shape)}")


def batched_igd_plus(F: Tensor, PF_true: Tensor) -> Tensor:
    """Weakly Pareto-compliant IGD+: ``d^+(p, f) = ||max(f - p, 0)||_2``."""
    F64 = F.to(torch.float64)
    P64 = PF_true.to(device=F.device, dtype=torch.float64)
    if F64.dim() == 3:
        # (B, P, 1, M) vs (B, 1, N, M)
        diff = torch.clamp(F64.unsqueeze(1) - P64.unsqueeze(0).unsqueeze(2), min=0.0)
        d = torch.linalg.norm(diff, dim=-1)                                # (B, P, N)
        return d.min(dim=-1).values.mean(dim=-1)
    if F64.dim() == 4:
        diff = torch.clamp(F64.unsqueeze(2) - P64.unsqueeze(1).unsqueeze(3), min=0.0)
        d = torch.linalg.norm(diff, dim=-1)                                # (T, B, P, N)
        return d.min(dim=-1).values.mean(dim=-1)
    raise ValueError(f"Expected 3-D or 4-D F tensor, got shape {tuple(F.shape)}")


def _hv_2d_exact(F: Tensor, ref_point: Tensor) -> Tensor:
    """Exact 2-D hypervolume via sorted cumulative-min staircase; ``F (B, N, 2)`` -> ``(B,)``."""
    B, N, _ = F.shape
    F64 = F.to(torch.float64)
    ref = ref_point.to(device=F.device, dtype=torch.float64).reshape(1, 2)

    # Points outside the reference box do not contribute: clamp them to ref
    valid = (F64[..., 0] < ref[:, 0:1]) & (F64[..., 1] < ref[:, 1:2])
    f1 = torch.where(valid, F64[..., 0], ref[:, 0:1].expand(B, N))
    f2 = torch.where(valid, F64[..., 1], ref[:, 1:2].expand(B, N))

    order = torch.argsort(f1, dim=1, stable=True)
    f1_s = torch.gather(f1, 1, order)
    f2_s = torch.gather(f2, 1, order)

    # Non-dominated staircase envelope via cummin of f2
    f2_env = torch.cummin(f2_s, dim=1).values
    f1_next = torch.cat([f1_s[:, 1:], ref[:, 0:1].expand(B, 1)], dim=1)
    widths = torch.clamp(f1_next - f1_s, min=0.0)
    heights = torch.clamp(ref[:, 1:2] - f2_env, min=0.0)
    return (widths * heights).sum(dim=1)


def _hv_mc_sobol(F: Tensor, ref_point: Tensor, ideal_point: Tensor | None = None,
                 n_sobol: int = 10000, seed: int = 2026) -> Tensor:
    """Quasi-Monte Carlo Sobol hypervolume for ``M >= 3``; ``F (B, N, M)`` -> ``(B,)``."""
    B, N, M = F.shape
    dev = F.device
    F64 = F.to(torch.float64)
    ref = ref_point.to(device=dev, dtype=torch.float64).reshape(1, M)
    if ideal_point is None:
        ideal = torch.minimum(F64.amin(dim=(0, 1), keepdim=False).unsqueeze(0), torch.zeros_like(ref))
    else:
        ideal = ideal_point.to(device=dev, dtype=torch.float64).reshape(1, M)

    span = torch.clamp(ref - ideal, min=0.0)
    box_vol = float(torch.prod(span).item())
    if box_vol <= 0.0:
        return torch.zeros(B, device=dev, dtype=torch.float64)

    eng = torch.quasirandom.SobolEngine(dimension=M, scramble=True, seed=seed)
    S = eng.draw(n_sobol).to(device=dev, dtype=torch.float64) * span + ideal  # (K, M)

    # Chunk over Sobol points to bound peak memory
    chunk = 2048
    dom_hits = torch.zeros(B, device=dev, dtype=torch.float64)
    for start in range(0, n_sobol, chunk):
        Sc = S[start : start + chunk]                                      # (C, M)
        # (B, 1, N, M) <= (1, C, 1, M) -> dominated if any n has F[b,n,m] <= Sc[c,m] for all m
        dom = (F64.unsqueeze(1) <= Sc.unsqueeze(0).unsqueeze(2)).all(dim=-1).any(dim=-1)  # (B, C)
        dom_hits += dom.sum(dim=1).to(torch.float64)
    return (dom_hits / float(n_sobol)) * box_vol


def batched_hv(F: Tensor, ref_point: Tensor, ideal_point: Tensor | None = None,
               n_sobol: int = 10000) -> Tensor:
    """Exact 2-D HV or QMC Sobol HV (``M >= 3``). Input ``F: (B, N, M)`` -> ``(B,)``."""
    if F.dim() == 2:
        return batched_hv(F.unsqueeze(0), ref_point, ideal_point, n_sobol)[0]
    if F.shape[-1] == 2:
        return _hv_2d_exact(F, ref_point)
    return _hv_mc_sobol(F, ref_point, ideal_point=ideal_point, n_sobol=n_sobol)


def adaptive_ref_point(PF_true: Tensor, scale: float = 1.1) -> Tensor:
    """Reference point ``1.1 * max(PF_true, dim=0)`` (clamped away from zero)."""
    mx = PF_true.to(torch.float64).max(dim=0).values
    return torch.where(mx > 0, mx * scale, mx + 0.1)


def compute_dynamic_metrics(
    F_env_end: Tensor,
    F_warmstart: Tensor,
    PF_list: list[Tensor],
    igd_per_gen: Tensor,
    baseline_warmstart_igd: Tensor | None = None,
    n_sobol: int = 4096,
) -> dict[str, Tensor]:
    """Compute per-seed MIGD, MIGD+, MHV, AUC-IGD and WSRR over all environments.

    Args:
        F_env_end: ``(T, B, N, M)`` end-of-environment objectives.
        F_warmstart: ``(T, B, N, M)`` gen-0 warm-start objectives.
        PF_list: list of ``T`` true fronts ``(P_t, M)``.
        igd_per_gen: ``(T * tau_t, B)`` full generation-by-generation IGD curve.
        baseline_warmstart_igd: optional ``(T, B)`` warm-start IGD of DNSGA-II for WSRR.
    """
    T, B, _, _ = F_env_end.shape
    dev = F_env_end.device
    igd_end = torch.empty((T, B), device=dev, dtype=torch.float64)
    igdp_end = torch.empty((T, B), device=dev, dtype=torch.float64)
    hv_end = torch.empty((T, B), device=dev, dtype=torch.float64)
    igd_warm = torch.empty((T, B), device=dev, dtype=torch.float64)

    for t_idx in range(T):
        pf = PF_list[t_idx]
        igd_end[t_idx] = batched_igd(F_env_end[t_idx], pf)
        igdp_end[t_idx] = batched_igd_plus(F_env_end[t_idx], pf)
        igd_warm[t_idx] = batched_igd(F_warmstart[t_idx], pf)
        ref = adaptive_ref_point(pf)
        ideal = torch.minimum(pf.to(torch.float64).min(dim=0).values, torch.zeros_like(ref))
        hv_end[t_idx] = batched_hv(F_env_end[t_idx], ref, ideal_point=ideal, n_sobol=n_sobol)

    # Exclude env 0 (static init) from dynamic averages when T > 1
    sl = slice(1, T) if T > 1 else slice(0, T)
    migd = igd_end[sl].mean(dim=0)
    migd_plus = igdp_end[sl].mean(dim=0)
    mhv = hv_end[sl].mean(dim=0)
    auc_igd = igd_per_gen.to(torch.float64).mean(dim=0)

    if baseline_warmstart_igd is not None:
        base_w = baseline_warmstart_igd.to(device=dev, dtype=torch.float64)[sl].clamp(min=1e-12)
        wsrr = ((base_w - igd_warm[sl]) / base_w).mean(dim=0)
    else:
        wsrr = torch.zeros(B, device=dev, dtype=torch.float64)

    return {
        "igd_env": igd_end,
        "igd_warm": igd_warm,
        "hv_env": hv_end,
        "migd": migd,
        "migd_plus": migd_plus,
        "mhv": mhv,
        "auc_igd": auc_igd,
        "wsrr": wsrr,
    }
