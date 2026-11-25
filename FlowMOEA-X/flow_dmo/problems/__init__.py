"""Dynamic benchmark registry: FDA1-4, dMOP1-3, CEC 2018 DF1-14, Microgrid-DEED."""
from __future__ import annotations

from flow_dmo.problems.base import DynamicProblem
from flow_dmo.problems.cec2018_df import (
    DF1,
    DF2,
    DF3,
    DF4,
    DF5,
    DF6,
    DF7,
    DF8,
    DF9,
    DF10,
    DF11,
    DF12,
    DF13,
    DF14,
    DF_PROBLEMS,
)
from flow_dmo.problems.dmop_suite import dMOP1, dMOP2, dMOP3
from flow_dmo.problems.fda_suite import FDA1, FDA2, FDA3, FDA4
from flow_dmo.problems.real_world_deed import DynamicMicrogridDEED

PROBLEM_REGISTRY: dict[str, type[DynamicProblem]] = {
    "FDA1": FDA1,
    "FDA2": FDA2,
    "FDA3": FDA3,
    "FDA4": FDA4,
    "DMOP1": dMOP1,
    "DMOP2": dMOP2,
    "DMOP3": dMOP3,
    **DF_PROBLEMS,
    "MICROGRID-DEED": DynamicMicrogridDEED,
    "DEED": DynamicMicrogridDEED,
}


def make_problem(name: str, **kwargs) -> DynamicProblem:
    """Instantiate a dynamic benchmark by name (case-insensitive)."""
    key = name.strip().upper()
    if key not in PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem {name!r}; available: {sorted(PROBLEM_REGISTRY)}")
    cls = PROBLEM_REGISTRY[key]
    if cls is DynamicMicrogridDEED:
        kwargs.pop("n_var", None)
    return cls(**kwargs)


ALL_ANALYTICAL_PROBLEMS = [
    "FDA1", "FDA2", "FDA3", "FDA4",
    "dMOP1", "dMOP2", "dMOP3",
    *[f"DF{i}" for i in range(1, 15)],
]

__all__ = [
    "ALL_ANALYTICAL_PROBLEMS",
    "DF1", "DF2", "DF3", "DF4", "DF5", "DF6", "DF7",
    "DF8", "DF9", "DF10", "DF11", "DF12", "DF13", "DF14",
    "DynamicMicrogridDEED",
    "DynamicProblem",
    "FDA1", "FDA2", "FDA3", "FDA4",
    "PROBLEM_REGISTRY",
    "dMOP1", "dMOP2", "dMOP3",
    "make_problem",
]
