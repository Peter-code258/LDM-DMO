"""Algorithm registry: FlowMOEA-X (Lite/Pro) + 5 fully-tensorised baselines."""
from __future__ import annotations

from flow_dmo.algorithms.base import BatchedDynamicMOEA
from flow_dmo.algorithms.baselines import TensorAR, TensorDNSGA2, TensorGMM, TensorLDT_v32, TensorPPS
from flow_dmo.algorithms.flow_moea_x import FlowMOEAX
from flow_dmo.config import TransportConfig


def make_algorithm(
    name: str,
    pop_size: int,
    n_var: int,
    n_obj: int,
    device="cpu",
    transport_cfg: TransportConfig | None = None,
) -> BatchedDynamicMOEA:
    key = name.strip().lower().replace("_", "-")
    if key in ("flowmoea-x", "flowmoea-x-lite", "flowmoeax"):
        return FlowMOEAX(pop_size, n_var, n_obj, device=device, transport_cfg=transport_cfg, mode="lite", name=name)
    if key in ("flowmoea-x-pro", "flowmoeax-pro", "pro"):
        return FlowMOEAX(pop_size, n_var, n_obj, device=device, transport_cfg=transport_cfg, mode="pro", name=name)
    if key in ("dnsga2", "dnsga-ii", "tensordnsga2"):
        return TensorDNSGA2(pop_size, n_var, n_obj, device=device)
    if key in ("ldt-v32", "ldtv32", "tensorldt-v32", "ldt"):
        return TensorLDT_v32(pop_size, n_var, n_obj, device=device)
    if key in ("pps", "tensorpps"):
        return TensorPPS(pop_size, n_var, n_obj, device=device)
    if key in ("ar", "tensorar"):
        return TensorAR(pop_size, n_var, n_obj, device=device)
    if key in ("gmm", "tensorgmm"):
        return TensorGMM(pop_size, n_var, n_obj, device=device)
    raise ValueError(f"Unknown algorithm {name!r}")


__all__ = [
    "BatchedDynamicMOEA",
    "FlowMOEAX",
    "TensorAR",
    "TensorDNSGA2",
    "TensorGMM",
    "TensorLDT_v32",
    "TensorPPS",
    "make_algorithm",
]
