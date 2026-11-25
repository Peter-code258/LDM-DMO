# FlowMOEA-X (v4.0)

**Fully-Tensorised PyTorch 2.x Dynamic Multi-Objective Optimisation with Continuous Manifold Optimal Transport**

FlowMOEA-X (`flow_dmo`) replaces single-seed Python object loops and isotropic Gaussian resampling with a 3-D batched tensor engine (`X: (B, N, D)`, `F: (B, N, M)`), reference-aligned Sliced Wasserstein Optimal Transport, per-dimension heterogeneous Kalman momentum, optional 8-step Conditional Flow Matching ODE, and zero-extra-FE closed-form Kernel Ridge Regression (KRR) pre-screening.

## Quick Start

```bash
# 1. Run full pytest acceptance suite
pytest -q

# 2. Run B=5 smoke verification and compile self-contained HTML report
python main.py --quick-verify

# 3. Launch the interactive Streamlit Dual-Space Manifold Workbench
python main.py --dashboard
# or:
streamlit run flow_dmo/visualization/streamlit_app.py
```

## Architecture Overview

| Layer | Package | Key Components |
| --- | --- | --- |
| Core Tensor Engine | `flow_dmo/core/` | `tensor_ops.py` (SBX, PM, LHS, reflective clamp), `dominance.py` (3-D batched NDS, crowding, Das-Dennis), `kernels.py` (`TensorNSGA2`, `TensorNSGA3`, `TensorMOEAD`), `fe_ledger.py` (strict zero-probe-leak FE ledger), `driver.py` (`BatchedDynamicDriver`) |
| Manifold Transport | `flow_dmo/transport/` | `change_detector.py` (embedded rank-0 sensors), `kalman_momentum.py` (per-dim `alpha_vec`), `sliced_ot.py` (reference-aligned Sliced-OT), `flow_matching.py` (CFM Midpoint ODE), `portfolio_gate.py` (3-regime negative-transfer circuit breaker) |
| Surrogate Layer | `flow_dmo/surrogate/` | `krr_prescreener.py` (Cholesky RBF-KRR + LCB 4x pre-screening, 0 extra real FEs) |
| Benchmarks | `flow_dmo/problems/` | `FDA1–4` (configurable `M in {2,3,5}`), authoritative `dMOP1–3`, full CEC 2018 `DF1–14`, `DynamicMicrogridDEED` (24-h microgrid dispatch) |
| Evaluation & Stats | `flow_dmo/evaluation/` | `gpu_metrics.py` (IGD, IGD+, exact 2-D HV, QMC Sobol HV, MIGD, MHV, WSRR), `statistics.py` (paired Wilcoxon + Holm + effect size `r`), `friedman.py` (Friedman + Nemenyi CD) |
| Visualisation | `flow_dmo/visualization/` | `plots.py`, `report_builder.py` (self-contained HTML report), `streamlit_app.py` |

## Mathematical Notation

| Symbol | Shape / Domain | Meaning |
| --- | --- | --- |
| $Z$ | $(B, N, D) \in [0, 1]^D$ | Normalised 3-D batched population tensor across $B$ seeds |
| $F$ | $(B, N, M)$ | Objective values evaluated via `FELedger` |
| $D_k$ | $(B,)$ | Normalised embedded-sensor objective drift intensity |
| $\boldsymbol{\alpha}_k$ | $(B, D)$ | Per-dimension Kalman momentum step vector (static / monotonic / reversal) |
| $\tilde{F}_{\text{cand}}$ | $(B, 4N_{\text{trans}}, M)$ | KRR Lower-Confidence-Bound surrogate predictions (0 real FEs) |
