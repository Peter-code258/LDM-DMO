"""Strongly-typed experiment configuration (dataclasses) with YAML round-trip.

Also hosts the device / dtype resolution helpers mandated by engineering rule
#2 (device agnosticism: ``cuda`` -> ``mps`` -> ``cpu``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def get_device(preference: str | torch.device | None = "auto") -> torch.device:
    """Resolve a torch device. ``"auto"`` probes cuda -> mps -> cpu."""
    if isinstance(preference, torch.device):
        return preference
    if preference is None or str(preference).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(str(preference))


def resolve_dtype(name: str | torch.dtype) -> torch.dtype:
    """Map a dtype name (``"float32"``) to the torch dtype."""
    if isinstance(name, torch.dtype):
        return name
    key = str(name).replace("torch.", "").lower()
    if key not in _DTYPES:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(_DTYPES)}")
    return _DTYPES[key]


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Seed-bound PRNG on the requested device (no global implicit RNG)."""
    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(int(seed))
    return gen


@dataclass
class TransportConfig:
    """Hyper-parameters of the distribution-transfer layer (Section 4)."""

    mode: str = "lite"                 # "lite" (Sliced-OT) | "pro" (Flow Matching)
    n_slices: int = 32                 # L random projection directions
    sinkhorn_eps: float = 0.01
    sinkhorn_iters: int = 4
    tangent_noise: float = 0.16
    align_strength: float = 0.10
    oversample_factor: int = 4         # K in K * N_trans
    krr_lambda: float = 1e-3
    lcb_beta: float = 0.25
    fm_hidden: int = 64
    fm_steps: int = 8
    fm_lr: float = 1e-2
    use_kalman: bool = True
    use_sliced_ot: bool = True
    use_krr: bool = True
    eps_change: float = 1e-4


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    problems: list[str] = field(default_factory=lambda: ["FDA1", "DF1", "DF3", "dMOP2"])
    algorithms: list[str] = field(
        default_factory=lambda: ["FlowMOEA-X", "DNSGA2", "LDT_v32", "PPS", "AR", "GMM"]
    )
    n_seeds: int = 30                  # B
    pop_size: int = 100                # N (bi-objective default; 150 for M=3)
    n_var: int = 10                    # D
    tau_t: int = 10                    # generations per environment
    n_t: int = 10                      # severity of change
    n_envs: int = 20                   # number of environments
    kernel: str = "nsga2"              # nsga2 | nsga3 | moead
    seed: int = 2026
    device: str = "auto"
    dtype: str = "float32"
    strict_fe_budget: bool = True
    results_dir: str = "results"
    eta_c: float = 15.0
    eta_m: float = 20.0
    prob_c: float = 0.9
    transport: TransportConfig = field(default_factory=TransportConfig)

    # ------------------------------------------------------------------ helpers
    @property
    def torch_device(self) -> torch.device:
        return get_device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return resolve_dtype(self.dtype)

    def pop_size_for(self, n_obj: int) -> int:
        """Spec default: N=100 for M=2, N=150 for M>=3 (unless overridden)."""
        if n_obj >= 3 and self.pop_size == 100:
            return 150
        return self.pop_size

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        data = dict(data or {})
        transport = data.pop("transport", None)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown ExperimentConfig keys: {sorted(unknown)}")
        cfg = cls(**data)
        if transport is not None:
            t_known = {f.name for f in dataclasses.fields(TransportConfig)}
            bad = set(transport) - t_known
            if bad:
                raise ValueError(f"Unknown TransportConfig keys: {sorted(bad)}")
            cfg.transport = TransportConfig(**transport)
        return cfg

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path_or_text: str | Path) -> ExperimentConfig:
        candidate = Path(str(path_or_text))
        if len(str(path_or_text)) < 1024 and candidate.exists():
            text = candidate.read_text(encoding="utf-8")
        else:
            text = str(path_or_text)
        return cls.from_dict(yaml.safe_load(text) or {})


def quick_verify_config() -> ExperimentConfig:
    """Configuration used by ``python main.py --quick-verify`` (B=5 smoke run)."""
    return ExperimentConfig(
        problems=["FDA1", "DF1", "DF3", "dMOP2", "DF10"],
        algorithms=["FlowMOEA-X", "FlowMOEA-X-Pro", "DNSGA2", "LDT_v32", "PPS", "AR", "GMM"],
        n_seeds=5,
        pop_size=60,
        n_var=10,
        tau_t=10,
        n_t=10,
        n_envs=10,
        device="auto",
    )
