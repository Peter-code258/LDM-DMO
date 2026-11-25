"""Visualisation, HTML report compilation and Streamlit dashboard (Section 8)."""
from flow_dmo.visualization.plots import (
    plot_kalman_alpha_ablation,
    plot_latency_comparison,
    plot_nemenyi_cd,
    plot_recovery_curves,
)
from flow_dmo.visualization.report_builder import build_html_report

__all__ = [
    "build_html_report",
    "plot_kalman_alpha_ablation",
    "plot_latency_comparison",
    "plot_nemenyi_cd",
    "plot_recovery_curves",
]
