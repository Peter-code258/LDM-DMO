"""Publication-grade plotting helpers for FlowMOEA-X (Section 8).

All functions render with the non-interactive ``Agg`` backend and return
base64-encoded PNG data URIs (and optionally write to disk).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flow_dmo.evaluation.friedman import FriedmanResult

_PALETTE = [
    "#0ea5e9",  # FlowMOEA-X (cyan)
    "#6366f1",  # FlowMOEA-X-Pro (indigo)
    "#ef4444",  # DNSGA2 (red)
    "#f59e0b",  # LDT_v32 (amber)
    "#10b981",  # PPS (emerald)
    "#8b5cf6",  # AR (violet)
    "#ec4899",  # GMM (pink)
]


def _fig_to_data_uri(fig: plt.Figure, save_path: str | Path | None = None, dpi: int = 160) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(buf.getvalue())
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def plot_nemenyi_cd(result: FriedmanResult, title: str = "Nemenyi Critical Difference (alpha = 0.05)",
                    save_path: str | Path | None = None) -> str:
    """Classic Demsar-style Critical Difference (CD) diagram."""
    items = sorted(result.avg_ranks.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    ranks = [v for _, v in items]
    k = len(names)

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.52 * k + 1.6)))
    ax.set_xlim(0.7, k + 0.3)
    ax.set_ylim(-0.6, k + 1.4)
    ax.axis("off")

    # Top rank axis (1 on right -> best, or 1 on left -> clearer reading)
    y_axis = k + 0.7
    ax.plot([1, k], [y_axis, y_axis], color="#1e293b", lw=1.6)
    for r in range(1, k + 1):
        ax.plot([r, r], [y_axis - 0.12, y_axis + 0.12], color="#1e293b", lw=1.2)
        ax.text(r, y_axis + 0.25, str(r), ha="center", va="bottom", fontsize=9, color="#334155")

    # CD bar
    cd = result.nemenyi_cd
    cd_start = 1.0
    cd_end = min(float(k), 1.0 + cd)
    ax.plot([cd_start, cd_end], [y_axis + 0.65, y_axis + 0.65], color="#dc2626", lw=2.5)
    ax.plot([cd_start, cd_start], [y_axis + 0.55, y_axis + 0.75], color="#dc2626", lw=2.0)
    ax.plot([cd_end, cd_end], [y_axis + 0.55, y_axis + 0.75], color="#dc2626", lw=2.0)
    ax.text((cd_start + cd_end) / 2, y_axis + 0.80, f"CD = {cd:.3f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#dc2626")

    # Algorithm markers & labels
    for i, (name, rk) in enumerate(items):
        y = k - 0.2 - i * 0.78
        col = _PALETTE[i % len(_PALETTE)]
        ax.plot([rk, rk], [y_axis, y], color=col, lw=1.4, ls="--", alpha=0.8)
        ax.scatter([rk], [y_axis], color=col, s=55, zorder=5)
        ax.text(rk + 0.08, y, f"{name} ({rk:.2f})", ha="left", va="center", fontsize=9.5,
                fontweight="bold" if i == 0 else "normal", color="#0f172a")

    # Clique bars for non-significant groups (|r_j - r_i| < CD)
    clique_y = y_axis - 0.28
    drawn = []
    for i in range(k):
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in drawn):
            drawn.append((i, j))
            ax.plot([ranks[i], ranks[j]], [clique_y, clique_y], color="#0f172a", lw=3.2, alpha=0.85)
            clique_y -= 0.16

    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12, color="#0f172a")
    return _fig_to_data_uri(fig, save_path=save_path)


def plot_recovery_curves(
    problem_name: str,
    igd_curves: dict[str, np.ndarray],
    tau_t: int,
    save_path: str | Path | None = None,
) -> str:
    """Per-generation log-IGD recovery curves with 25th-75th percentile IQR bands."""
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    n_gens = 0
    for idx, (algo, mat) in enumerate(igd_curves.items()):
        arr = np.asarray(mat, dtype=np.float64)  # (G, B)
        n_gens = max(n_gens, arr.shape[0])
        gens = np.arange(arr.shape[0])
        med = np.median(arr, axis=1)
        q25 = np.percentile(arr, 25, axis=1)
        q75 = np.percentile(arr, 75, axis=1)
        col = _PALETTE[idx % len(_PALETTE)]
        lw = 2.3 if "FlowMOEA" in algo else 1.4
        ax.plot(gens, med, label=algo, color=col, lw=lw)
        ax.fill_between(gens, q25, q75, color=col, alpha=0.14)

    for g in range(tau_t, n_gens, tau_t):
        ax.axvline(g, color="#94a3b8", ls=":", lw=0.8, alpha=0.6)

    ax.set_yscale("log")
    ax.set_xlabel("Generation (vertical lines = environment switches)", fontsize=9.5)
    ax.set_ylabel("IGD (log scale, median + IQR)", fontsize=9.5)
    ax.set_title(f"{problem_name} — Per-Generation IGD Recovery Dynamics", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.25)
    ax.legend(frameon=True, fontsize=8.5, ncol=min(4, max(1, len(igd_curves))))
    return _fig_to_data_uri(fig, save_path=save_path)


def plot_latency_comparison(latencies_ms: dict[str, float], save_path: str | Path | None = None) -> str:
    """Horizontal bar chart comparing mean per-switch transfer response latency (ms)."""
    names = list(latencies_ms.keys())
    vals = [float(latencies_ms[n]) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, max(2.8, 0.45 * len(names) + 1.2)))
    y = np.arange(len(names))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]
    bars = ax.barh(y, vals, color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Environment-Switch Response Time (ms / batch)", fontsize=9.5)
    ax.set_title("Per-Switch Transfer Latency Comparison", fontsize=11, fontweight="bold")
    ax.grid(axis="x", ls="--", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f} ms",
                va="center", fontsize=8.5, color="#1e293b")
    return _fig_to_data_uri(fig, save_path=save_path)


def plot_kalman_alpha_ablation(alpha_history: list[np.ndarray], problem_name: str,
                               save_path: str | Path | None = None) -> str:
    """Visualise per-dimension Kalman step sizes ``alpha_vec`` across environment switches."""
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    if not alpha_history:
        ax.text(0.5, 0.5, "No alpha history recorded", ha="center", va="center")
        return _fig_to_data_uri(fig, save_path=save_path)
    # Stack (T_sw, B, D) -> mean over seeds (T_sw, D)
    arr = np.stack([np.asarray(a, dtype=np.float64).mean(axis=0) for a in alpha_history], axis=0)
    steps = np.arange(1, arr.shape[0] + 1)
    ax.plot(steps, arr[:, 0], label="Position variable x_1 (alpha_0)", color="#0ea5e9", lw=2.4, marker="o")
    if arr.shape[1] > 1:
        dist_mean = arr[:, 1:].mean(axis=1)
        ax.plot(steps, dist_mean, label="Distance variables x_{2:D} (mean alpha)", color="#ef4444", lw=2.4, marker="s")
    ax.axhline(1.0, color="#64748b", ls="--", lw=1.2, label="Legacy scalar alpha = 1.0 (v3.1)")
    ax.set_xlabel("Environment Switch Index k", fontsize=9.5)
    ax.set_ylabel("Effective Momentum Step Size alpha", fontsize=9.5)
    ax.set_title(f"{problem_name} — Per-Dimension Kalman Step-Size Decoupling (v3.1 vs v4.0)",
                 fontsize=10.5, fontweight="bold")
    ax.set_ylim(-0.05, 1.35)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=8.5)
    return _fig_to_data_uri(fig, save_path=save_path)
