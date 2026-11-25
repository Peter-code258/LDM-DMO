"""Friedman rank-sum test, Iman-Davenport F_F statistic and Nemenyi post-hoc
Critical Difference (CD) at alpha = 0.05 (Section 6.2.2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Exact Studentised range critical values q_{0.05} / sqrt(2) for k = 2..12 (Demsar 2006)
_NEMENYI_Q_ALPHA_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
}


@dataclass
class FriedmanResult:
    algorithms: list[str]
    avg_ranks: dict[str, float]
    n_configs: int
    chi2_stat: float
    p_value: float
    iman_davenport_f: float
    iman_davenport_p: float
    nemenyi_cd: float


def nemenyi_q_alpha(k: int, alpha: float = 0.05) -> float:
    """Return Nemenyi critical value ``q_alpha = studentized_range.ppf(1-alpha, k, inf) / sqrt(2)``."""
    if alpha == 0.05 and k in _NEMENYI_Q_ALPHA_005:
        return _NEMENYI_Q_ALPHA_005[k]
    return float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0))


def friedman_nemenyi_test(
    score_matrix: np.ndarray,
    algorithms: list[str],
    higher_is_better: bool = False,
    alpha: float = 0.05,
) -> FriedmanResult:
    """Run Friedman rank test and compute Nemenyi CD.

    Args:
        score_matrix: shape ``(N_configs, K)``, where row ``i`` is a problem/config
            and column ``j`` is algorithm ``j``'s mean score.
        algorithms: list of ``K`` algorithm names.
    """
    mat = np.asarray(score_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] != len(algorithms):
        raise ValueError("score_matrix must have shape (N_configs, len(algorithms))")
    N, K = mat.shape
    if K < 2 or N < 1:
        raise ValueError("Need at least 2 algorithms and 1 configuration")

    # Rank each row (1 = best)
    ranks = np.empty_like(mat)
    for i in range(N):
        row = -mat[i] if higher_is_better else mat[i]
        ranks[i] = stats.rankdata(row, method="average")

    avg_r = ranks.mean(axis=0)
    # Friedman chi-square statistic
    chi2 = (12.0 * N / (K * (K + 1.0))) * (np.sum(avg_r ** 2) - K * ((K + 1.0) ** 2) / 4.0)
    chi2 = float(max(0.0, chi2))
    p_chi2 = float(stats.chi2.sf(chi2, df=K - 1))

    # Iman-Davenport F_F statistic
    denom = max(N * (K - 1.0) - chi2, 1e-12)
    f_stat = float((N - 1.0) * chi2 / denom) if N > 1 else 0.0
    p_f = float(stats.f.sf(f_stat, dfn=K - 1, dfd=(K - 1) * (N - 1))) if N > 1 else 1.0

    q_a = nemenyi_q_alpha(K, alpha=alpha)
    cd = float(q_a * np.sqrt(K * (K + 1.0) / (6.0 * N)))

    return FriedmanResult(
        algorithms=list(algorithms),
        avg_ranks={a: float(r) for a, r in zip(algorithms, avg_r)},
        n_configs=N,
        chi2_stat=chi2,
        p_value=p_chi2,
        iman_davenport_f=f_stat,
        iman_davenport_p=p_f,
        nemenyi_cd=cd,
    )
