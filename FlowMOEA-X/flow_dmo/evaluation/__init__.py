"""Evaluation metrics and non-parametric statistical tests (Section 6)."""
from flow_dmo.evaluation.friedman import FriedmanResult, friedman_nemenyi_test, nemenyi_q_alpha
from flow_dmo.evaluation.gpu_metrics import (
    adaptive_ref_point,
    batched_hv,
    batched_igd,
    batched_igd_plus,
    compute_dynamic_metrics,
)
from flow_dmo.evaluation.statistics import (
    PairwiseComparison,
    compare_against_control,
    holm_step_down,
    paired_wilcoxon,
    summarize_wtl,
)

__all__ = [
    "FriedmanResult",
    "PairwiseComparison",
    "adaptive_ref_point",
    "batched_hv",
    "batched_igd",
    "batched_igd_plus",
    "compare_against_control",
    "compute_dynamic_metrics",
    "friedman_nemenyi_test",
    "holm_step_down",
    "nemenyi_q_alpha",
    "paired_wilcoxon",
    "summarize_wtl",
]
