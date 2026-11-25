"""FlowMOEA-X (package ``flow_dmo``).

Fully-tensorised dynamic multi-objective optimisation framework. The core
public classes are exported lazily (PEP 562) so that light-weight sub-modules
(e.g. ``flow_dmo.core.tensor_ops``) can be imported without pulling in the
whole algorithm stack.
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "4.0.0"

_LAZY_EXPORTS = {
    "BatchedDynamicDriver": "flow_dmo.core.driver",
    "FlowMOEAX": "flow_dmo.algorithms.flow_moea_x",
    "DynamicProblem": "flow_dmo.problems.base",
    "FELedger": "flow_dmo.core.fe_ledger",
    "ExperimentConfig": "flow_dmo.config",
    "get_device": "flow_dmo.config",
}

__all__ = sorted(_LAZY_EXPORTS) + ["__version__"]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'flow_dmo' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
