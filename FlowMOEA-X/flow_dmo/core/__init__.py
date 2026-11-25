"""Core tensor infrastructure: operators, dominance, kernels, FE ledger, driver."""
from flow_dmo.core.dominance import (
    batched_crowding_distance,
    batched_gather,
    batched_pareto_ranks,
    das_dennis,
    nsga2_sort_key,
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

__all__ = [
    "BudgetExceededError",
    "FELedger",
    "TensorMOEAD",
    "TensorNSGA2",
    "TensorNSGA3",
    "batched_crowding_distance",
    "batched_gather",
    "batched_pareto_ranks",
    "batched_pm",
    "batched_sbx",
    "das_dennis",
    "denormalize",
    "make_kernel",
    "normalize",
    "nsga2_sort_key",
    "orthogonal_lhs",
    "reflect_clamp",
    "uniform_reference_vectors",
]
