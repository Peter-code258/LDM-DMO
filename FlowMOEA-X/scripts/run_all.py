"""Cross-platform fully automated end-to-end pipeline orchestrator for FlowMOEA-X.

Stages:
  1. Environment & hardware diagnostics (Python, PyTorch, CUDA/MPS/CPU)
  2. Full pytest acceptance test suite (optional skip via ``--skip-tests``)
  3. Batched tensor benchmark execution + Wilcoxon/Holm + Friedman/Nemenyi + HTML report
  4. Strict FE ledger audit & console leaderboard summary
  5. Optional browser report open / Streamlit workbench launch
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from flow_dmo.config import ExperimentConfig, get_device  # noqa: E402
from main import run_benchmark_pipeline  # noqa: E402


def _banner(stage: str, title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n[{stage}] {title}\n{line}", flush=True)


def stage_env_check(device_pref: str) -> torch.device:
    _banner("Stage 1/4", "Environment & Hardware Check")
    dev = get_device(device_pref)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device : {torch.cuda.get_device_name(0)}")
    print(f"  Active dev  : {dev}")
    return dev


def stage_pytest() -> None:
    _banner("Stage 2/4", "Running Full Pytest Acceptance Suite")
    t0 = time.perf_counter()
    rc = subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    if rc != 0:
        raise SystemExit(f"[FAIL] pytest exited with code {rc} after {elapsed:.2f}s")
    print(f"  [PASS] All acceptance tests passed in {elapsed:.2f}s")


def stage_run_experiments(cfg: ExperimentConfig) -> Path:
    _banner("Stage 3/4", f"Running Batched Experiments -> {cfg.results_dir}")
    t0 = time.perf_counter()
    report_path = run_benchmark_pipeline(cfg)
    elapsed = time.perf_counter() - t0
    print(f"  [DONE] Benchmark + statistical analysis + HTML report completed in {elapsed:.2f}s")
    return report_path


def stage_audit_and_leaderboard(report_path: Path) -> None:
    _banner("Stage 4/4", "Strict FE Ledger Audit & Friedman Leaderboard")
    json_path = report_path.with_suffix(".json")
    if not json_path.exists():
        raise SystemExit(f"[FAIL] Expected JSON sidecar not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # 1. Verify strict FE ledger across all runs
    ledger_rows = data.get("ledger_rows", [])
    bad = [r for r in ledger_rows if not r.get("all_envs_exact", False)]
    if bad:
        raise SystemExit(f"[FAIL] FE budget audit failed on {len(bad)} runs!")
    print(f"  [PASS] Strict FE Ledger verified across all {len(ledger_rows)} (problem, algorithm) runs (0 probe leakage)")

    # 2. Print Friedman average ranks + Wilcoxon W/T/L leaderboard
    friedman = data.get("friedman", {})
    avg_ranks = friedman.get("avg_ranks", {})
    wtl = data.get("wtl", {})
    control = data.get("meta", {}).get("control", "FlowMOEA-X")
    print(f"\n  Friedman chi2 = {friedman.get('chi2_stat', 0.0):.3f} "
          f"(p = {friedman.get('p_value', 1.0):.4g}) | "
          f"Nemenyi CD = {friedman.get('nemenyi_cd', 0.0):.3f}\n")
    print(f"  {'Rank':<6} {'Algorithm':<18} {'Avg Rank':<10} {'W/T/L vs ' + control}")
    print("  " + "-" * 56)
    for idx, (algo, rk) in enumerate(sorted(avg_ranks.items(), key=lambda kv: kv[1]), start=1):
        if algo == control:
            wtl_s = "Control"
        else:
            w = wtl.get(algo, {})
            wtl_s = f"{w.get('W', 0)} / {w.get('T', 0)} / {w.get('L', 0)}"
        print(f"  #{idx:<5} {algo:<18} {rk:<10.3f} {wtl_s}")

    print(f"\n  HTML Report : {report_path.resolve()}")
    print(f"  CSV Metrics : {(report_path.parent / 'summary_metrics.csv').resolve()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FlowMOEA-X Full Automated Pipeline")
    parser.add_argument("--mode", choices=["quick", "standard", "paper"], default="quick",
                        help="Preset experiment scale (default: quick)")
    parser.add_argument("--config", type=str, default=None, help="Custom YAML config path (overrides --mode)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto | cpu | cuda | mps")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--skip-tests", action="store_true", help="Skip Stage 2 pytest suite")
    parser.add_argument("--open-report", action="store_true", help="Open compiled HTML report in browser")
    parser.add_argument("--launch-dashboard", action="store_true", help="Launch Streamlit workbench after completion")
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    stage_env_check(args.device)

    if not args.skip_tests:
        stage_pytest()
    else:
        print("\n[Stage 2/4] Skipped pytest (--skip-tests)")

    cfg_path = Path(args.config) if args.config else (ROOT / "configs" / f"{args.mode}.yaml")
    cfg = ExperimentConfig.from_yaml(cfg_path)
    cfg.device = args.device
    if args.output_dir:
        cfg.results_dir = args.output_dir
    else:
        cfg.results_dir = str((ROOT / cfg.results_dir).resolve())

    report_path = stage_run_experiments(cfg)
    stage_audit_and_leaderboard(report_path)

    total_s = time.perf_counter() - t_start
    print(f"\n[FlowMOEA-X] Full automated pipeline finished successfully in {total_s:.2f}s.\n")

    if args.open_report:
        webbrowser.open(report_path.resolve().as_uri())

    if args.launch_dashboard:
        app = ROOT / "flow_dmo" / "visualization" / "streamlit_app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)], cwd=str(ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
