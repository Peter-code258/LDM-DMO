"""Shared pytest fixtures: fixed seed 2026, device selection and CPU/GPU
consistency helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flow_dmo.config import get_device, make_generator  # noqa: E402

SEED = 2026


@pytest.fixture
def seed() -> int:
    return SEED


@pytest.fixture
def device() -> torch.device:
    return get_device("auto")


@pytest.fixture
def generator(device) -> torch.Generator:
    return make_generator(SEED, device)


def available_devices() -> list[torch.device]:
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def assert_finite_in_unit_box(Z: torch.Tensor) -> None:
    assert torch.isfinite(Z).all(), "tensor contains NaN/Inf"
    assert float(Z.min()) >= 0.0 and float(Z.max()) <= 1.0, "tensor left [0, 1]"


def assert_cpu_gpu_consistent(fn, *args, atol: float = 1e-5) -> None:
    """Run ``fn(device, *args)`` on CPU and (if present) CUDA and compare outputs.

    ``fn`` must be deterministic given its arguments (no device RNG involved).
    """
    ref = fn(torch.device("cpu"), *args)
    if not torch.cuda.is_available():
        return
    out = fn(torch.device("cuda"), *args)
    ref_t = ref if isinstance(ref, (tuple, list)) else (ref,)
    out_t = out if isinstance(out, (tuple, list)) else (out,)
    for a, b in zip(ref_t, out_t):
        assert torch.allclose(a.cpu().double(), b.cpu().double(), atol=atol)
