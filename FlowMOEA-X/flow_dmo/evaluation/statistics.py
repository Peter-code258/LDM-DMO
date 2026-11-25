"""Non-parametric paired Wilcoxon signed-rank tests, Holm step-down multiple-
comparison correction, effect size r and W/T/L summaries (Section 6.2.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PairwiseComparison:
    control: str
    competitor: str
    control_mean: float
    control_std: float
    competitor_mean: float
    competitor_std: float
    p_raw: float
    p_holm: float
    effect_size_r: float
    symbol: str           # "+" (control better), "=" (tie), "-" (control worse)


def holm_step_down(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values preserving original order and monotonicity."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    adj_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for i in range(m):
        val = min(1.0, (m - i) * p_sorted[i])
        running_max = max(running_max, val)
        adj_sorted[i] = running_max
    adj = np.empty(m, dtype=np.float64)
    adj[order] = adj_sorted
    return adj


def paired_wilcoxon(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    higher_is_better: bool = False,
) -> tuple[float, float, int]:
    """Two-sided paired Wilcoxon signed-rank test handling all-zero differences safely.

    Returns ``(p_value, effect_size_r, direction)`` where ``direction`` is
    ``+1`` if ``x`` is better than ``y``, ``-1`` if worse, ``0`` if identical.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    diff = xa - ya
    n = len(diff)
    if n == 0 or np.allclose(diff, 0.0, atol=1e-15):
        return 1.0, 0.0, 0

    try:
        res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        p_val = float(res.pvalue)
        stat = float(res.statistic)
    except ValueError:
        return 1.0, 0.0, 0

    if np.isnan(p_val):
        p_val = 1.0

    # Normal-approximation Z for effect size r = |Z| / sqrt(n)
    nz = int(np.sum(np.abs(diff) > 1e-15))
    mean_w = nz * (nz + 1) / 4.0
    std_w = np.sqrt(nz * (nz + 1) * (2 * nz + 1) / 24.0) if nz > 0 else 1.0
    z_score = abs(stat - mean_w) / max(std_w, 1e-12)
    r = float(min(1.0, z_score / np.sqrt(max(n, 1))))

    med_diff = float(np.median(diff))
    if abs(med_diff) < 1e-15:
        med_diff = float(np.mean(diff))
    if higher_is_better:
        direction = 1 if med_diff > 0 else (-1 if med_diff < 0 else 0)
    else:
        direction = 1 if med_diff < 0 else (-1 if med_diff > 0 else 0)
    return p_val, r, direction


def compare_against_control(
    scores_by_algo: dict[str, np.ndarray | list[float]],
    control: str = "FlowMOEA-X",
    higher_is_better: bool = False,
    alpha: float = 0.05,
) -> list[PairwiseComparison]:
    """Compare ``control`` against every competitor with Holm step-down adjustment."""
    if control not in scores_by_algo:
        raise KeyError(f"Control algorithm {control!r} not in scores")
    ctrl_arr = np.asarray(scores_by_algo[control], dtype=np.float64)
    competitors = [a for a in scores_by_algo if a != control]
    raw_p: list[float] = []
    rs: list[float] = []
    dirs: list[int] = []

    for comp in competitors:
        p, r, d = paired_wilcoxon(ctrl_arr, scores_by_algo[comp], higher_is_better=higher_is_better)
        raw_p.append(p)
        rs.append(r)
        dirs.append(d)

    adj_p = holm_step_down(raw_p)
    out: list[PairwiseComparison] = []
    for comp, p_r, p_h, r, d in zip(competitors, raw_p, adj_p, rs, dirs):
        arr = np.asarray(scores_by_algo[comp], dtype=np.float64)
        if p_h < alpha and d > 0:
            sym = "+"
        elif p_h < alpha and d < 0:
            sym = "-"
        else:
            sym = "="
        out.append(
            PairwiseComparison(
                control=control,
                competitor=comp,
                control_mean=float(ctrl_arr.mean()),
                control_std=float(ctrl_arr.std(ddof=1)) if len(ctrl_arr) > 1 else 0.0,
                competitor_mean=float(arr.mean()),
                competitor_std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                p_raw=float(p_r),
                p_holm=float(p_h),
                effect_size_r=float(r),
                symbol=sym,
            )
        )
    return out


def summarize_wtl(comparisons: list[PairwiseComparison]) -> dict[str, dict[str, int]]:
    """Aggregate Win (+), Tie (=), Loss (-) counts per competitor across configs."""
    summary: dict[str, dict[str, int]] = {}
    for c in comparisons:
        bucket = summary.setdefault(c.competitor, {"W": 0, "T": 0, "L": 0})
        if c.symbol == "+":
            bucket["W"] += 1
        elif c.symbol == "-":
            bucket["L"] += 1
        else:
            bucket["T"] += 1
    return summary
