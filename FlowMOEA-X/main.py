"""One-click FlowMOEA-X experiment pipeline & CLI entry point (Section 2 & 10).

Examples:
    python main.py --quick-verify
    python main.py --config path/to/exp.yaml
    python main.py --dashboard
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from flow_dmo.algorithms import make_algorithm
from flow_dmo.config import ExperimentConfig, get_device, quick_verify_config
from flow_dmo.core.driver import BatchedDynamicDriver
from flow_dmo.evaluation import (
    compare_against_control,
    compute_dynamic_metrics,
    friedman_nemenyi_test,
    summarize_wtl,
)
from flow_dmo.problems import make_problem
from flow_dmo.visualization import (
    build_html_report,
    plot_kalman_alpha_ablation,
    plot_latency_comparison,
    plot_nemenyi_cd,
    plot_recovery_curves,
)


def run_benchmark_pipeline(cfg: ExperimentConfig) -> Path:
    """Execute batched experiments, run Wilcoxon/Friedman tests and compile HTML report."""
    device = get_device(cfg.device)
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    control = "FlowMOEA-X" if "FlowMOEA-X" in cfg.algorithms else cfg.algorithms[0]
    summary_rows: list[dict] = []
    ledger_rows: list[dict] = []
    all_comparisons = []
    friedman_rows: list[list[float]] = []
    latency_acc: dict[str, list[float]] = {a: [] for a in cfg.algorithms}
    recovery_charts: dict[str, str] = {}
    kalman_chart_uri = ""

    print(f"[FlowMOEA-X] Running {len(cfg.problems)} problems x {len(cfg.algorithms)} algorithms "
          f"(B={cfg.n_seeds}, N={cfg.pop_size}, tau_t={cfg.tau_t}, n_envs={cfg.n_envs}) on {device}")

    for prob_name in cfg.problems:
        prob = make_problem(prob_name, n_var=cfg.n_var, tau_t=cfg.tau_t, n_t=cfg.n_t,
                            device=device, dtype=cfg.torch_dtype)
        pop_n = cfg.pop_size_for(prob.n_obj)
        trajs = {}
        for algo_name in cfg.algorithms:
            algo = make_algorithm(algo_name, pop_size=pop_n, n_var=prob.n_var, n_obj=prob.n_obj,
                                  device=device, transport_cfg=cfg.transport)
            driver = BatchedDynamicDriver(
                problem=prob,
                algorithm=algo,
                kernel=cfg.kernel,
                n_seeds=cfg.n_seeds,
                pop_size=pop_n,
                tau_t=cfg.tau_t,
                n_envs=cfg.n_envs,
                seed=cfg.seed,
                strict_fe_budget=cfg.strict_fe_budget,
                eps_change=cfg.transport.eps_change,
            )
            trajs[algo_name] = driver.run()

        # Use DNSGA2 warm-start IGD as WSRR baseline when present
        base_warm = None
        if "DNSGA2" in trajs:
            base_m = compute_dynamic_metrics(
                trajs["DNSGA2"].F_env_end, trajs["DNSGA2"].F_warmstart,
                trajs["DNSGA2"].PF_true, trajs["DNSGA2"].igd_per_gen, n_sobol=2048,
            )
            base_warm = base_m["igd_warm"]

        migd_by_algo: dict[str, np.ndarray] = {}
        metrics_by_algo: dict[str, dict] = {}
        igd_curves: dict[str, np.ndarray] = {}

        for algo_name in cfg.algorithms:
            tr = trajs[algo_name]
            m = compute_dynamic_metrics(
                tr.F_env_end, tr.F_warmstart, tr.PF_true, tr.igd_per_gen,
                baseline_warmstart_igd=base_warm, n_sobol=2048,
            )
            metrics_by_algo[algo_name] = m
            migd_by_algo[algo_name] = m["migd"].cpu().numpy()
            igd_curves[algo_name] = tr.igd_per_gen.cpu().numpy()
            mean_ms = float(np.mean(tr.transfer_times_ms)) if tr.transfer_times_ms else 0.0
            latency_acc[algo_name].append(mean_ms)

            ls = dict(tr.ledger_summary)
            ls["problem"] = prob_name
            ls["algorithm"] = algo_name
            ledger_rows.append(ls)

            if not kalman_chart_uri and algo_name == control and tr.alpha_history:
                kalman_chart_uri = plot_kalman_alpha_ablation(
                    [a.numpy() for a in tr.alpha_history], problem_name=prob_name
                )

        # Pairwise Wilcoxon + Holm vs control
        comps = compare_against_control(migd_by_algo, control=control, higher_is_better=False)
        all_comparisons.extend(comps)
        sym_map = {c.competitor: c.symbol for c in comps}

        friedman_rows.append([float(migd_by_algo[a].mean()) for a in cfg.algorithms])
        recovery_charts[prob_name] = plot_recovery_curves(prob_name, igd_curves, tau_t=cfg.tau_t)

        for algo_name in cfg.algorithms:
            m = metrics_by_algo[algo_name]
            migd = m["migd"].cpu().numpy()
            mhv = m["mhv"].cpu().numpy()
            wsrr = m["wsrr"].cpu().numpy()
            tr = trajs[algo_name]
            summary_rows.append({
                "problem": prob_name,
                "algorithm": algo_name,
                "migd_mean": float(migd.mean()),
                "migd_std": float(migd.std(ddof=1)) if len(migd) > 1 else 0.0,
                "mhv_mean": float(mhv.mean()),
                "mhv_std": float(mhv.std(ddof=1)) if len(mhv) > 1 else 0.0,
                "wsrr_mean": float(wsrr.mean()),
                "transfer_ms": float(np.mean(tr.transfer_times_ms)) if tr.transfer_times_ms else 0.0,
                "symbol": "" if algo_name == control else sym_map.get(algo_name, "="),
            })

        print(f"  [OK] {prob_name}: {control} MIGD={migd_by_algo[control].mean():.4f}")

    # Save tabular CSV summary
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_metrics.csv", index=False)

    # Global Friedman + Nemenyi CD
    fr_res = friedman_nemenyi_test(np.asarray(friedman_rows), cfg.algorithms, higher_is_better=False)
    wtl = summarize_wtl(all_comparisons)
    mean_lat = {a: float(np.mean(v)) if v else 0.0 for a, v in latency_acc.items()}

    report_data = {
        "meta": {
            "device": str(device),
            "control": control,
            "n_seeds": cfg.n_seeds,
            "pop_size": cfg.pop_size,
            "tau_t": cfg.tau_t,
            "n_t": cfg.n_t,
            "n_envs": cfg.n_envs,
            "kernel": cfg.kernel,
        },
        "friedman": {
            "avg_ranks": fr_res.avg_ranks,
            "chi2_stat": fr_res.chi2_stat,
            "p_value": fr_res.p_value,
            "iman_davenport_f": fr_res.iman_davenport_f,
            "nemenyi_cd": fr_res.nemenyi_cd,
        },
        "wtl": wtl,
        "summary_rows": summary_rows,
        "ledger_rows": ledger_rows,
        "charts": {
            "nemenyi_cd": plot_nemenyi_cd(fr_res),
            "latency": plot_latency_comparison(mean_lat),
            "kalman_ablation": kalman_chart_uri,
            "recovery": recovery_charts,
        },
    }
    report_path = build_html_report(report_data, output_path=out_dir / "flow_moea_experiment_report.html")
    print(f"[FlowMOEA-X] Report written to: {report_path.resolve()}")
    return report_path


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FlowMOEA-X v4.0 Experiment Runner")
    parser.add_argument("--quick-verify", action="store_true", help="Run B=5 smoke verification and compile HTML report")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML ExperimentConfig")
    parser.add_argument("--output-dir", type=str, default=None, help="Override results output directory")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest acceptance suite before experiments")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dual-space manifold workbench")
    args = parser.parse_args(argv)

    if args.dashboard:
        app_path = Path(__file__).resolve().parent / "flow_dmo" / "visualization" / "streamlit_app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])

    if args.run_tests:
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q"])
        if rc != 0:
            return rc

    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
    elif args.quick_verify:
        cfg = quick_verify_config()
    else:
        cfg = ExperimentConfig()

    if args.output_dir:
        cfg.results_dir = args.output_dir

    run_benchmark_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
