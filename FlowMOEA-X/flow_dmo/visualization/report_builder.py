"""Self-contained HTML experiment report compiler (Section 8.1).

Generates a single portable HTML file ``results/flow_moea_experiment_report.html``
with inline base64 figures, interactive table filtering, and four sections:
  1. Global Overview (Friedman avg ranks, Nemenyi CD, Wilcoxon W/T/L, latency)
  2. Per-Dimension Kalman Step-Size Ablation (v3.1 vs v4.0)
  3. Per-Generation IGD Recovery Curves
  4. Strict Function-Evaluation Ledger Audit Table
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_html_report(
    report_data: dict[str, Any],
    output_path: str | Path = "results/flow_moea_experiment_report.html",
) -> Path:
    """Compile ``report_data`` into a self-contained single-file HTML report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = report_data.get("meta", {})
    friedman = report_data.get("friedman", {})
    wtl = report_data.get("wtl", {})
    rows = report_data.get("summary_rows", [])
    ledger_rows = report_data.get("ledger_rows", [])
    charts = report_data.get("charts", {})
    recovery_charts: dict[str, str] = charts.get("recovery", {})

    # Build Friedman rank table rows
    avg_ranks = friedman.get("avg_ranks", {})
    rank_rows_html = ""
    for rank_idx, (algo, rk) in enumerate(sorted(avg_ranks.items(), key=lambda kv: kv[1]), start=1):
        badge = '<span class="badge best">#1</span>' if rank_idx == 1 else f"#{rank_idx}"
        w_info = wtl.get(algo, {})
        wtl_str = (
            "Control"
            if algo == meta.get("control", "FlowMOEA-X")
            else f"{w_info.get('W', 0)} / {w_info.get('T', 0)} / {w_info.get('L', 0)}"
        )
        rank_rows_html += (
            f"<tr><td>{badge}</td><td><b>{html.escape(algo)}</b></td>"
            f"<td>{rk:.3f}</td><td>{wtl_str}</td></tr>\n"
        )

    # Build per-problem summary rows (filterable by JS)
    detail_rows_html = ""
    for r in rows:
        sym = r.get("symbol", "")
        sym_cls = {"+": "win", "-": "loss", "=": "tie"}.get(sym, "")
        detail_rows_html += (
            f"<tr data-problem='{html.escape(str(r['problem']))}' data-algo='{html.escape(str(r['algorithm']))}'>"
            f"<td>{html.escape(str(r['problem']))}</td>"
            f"<td><b>{html.escape(str(r['algorithm']))}</b></td>"
            f"<td>{r['migd_mean']:.4f} &plusmn; {r['migd_std']:.4f}</td>"
            f"<td>{r['mhv_mean']:.4f} &plusmn; {r['mhv_std']:.4f}</td>"
            f"<td>{r['wsrr_mean'] * 100:.1f}%</td>"
            f"<td>{r['transfer_ms']:.2f} ms</td>"
            f"<td class='{sym_cls}'>{html.escape(sym or 'ctrl')}</td>"
            f"</tr>\n"
        )

    # Build FE ledger audit rows
    ledger_rows_html = ""
    for lr in ledger_rows:
        ok = "PASS" if lr.get("all_envs_exact") else "CHECK"
        ok_cls = "win" if lr.get("all_envs_exact") else "loss"
        ledger_rows_html += (
            f"<tr><td>{html.escape(str(lr['problem']))}</td>"
            f"<td>{html.escape(str(lr['algorithm']))}</td>"
            f"<td>{lr['sensor_fe']}</td>"
            f"<td>{lr['warmstart_fe']}</td>"
            f"<td>{lr['evolution_fe']}</td>"
            f"<td><b>{lr['total_fe_per_seed']}</b></td>"
            f"<td class='{ok_cls}'>{ok}</td></tr>\n"
        )

    # Recovery charts gallery
    recovery_html = ""
    for prob_name, uri in recovery_charts.items():
        recovery_html += (
            f'<div class="card"><h3>{html.escape(prob_name)}</h3>'
            f'<img src="{uri}" alt="{html.escape(prob_name)} recovery"/></div>\n'
        )

    cd_img = f'<img src="{charts["nemenyi_cd"]}" alt="Nemenyi CD"/>' if charts.get("nemenyi_cd") else ""
    lat_img = f'<img src="{charts["latency"]}" alt="Latency"/>' if charts.get("latency") else ""
    kal_img = f'<img src="{charts["kalman_ablation"]}" alt="Kalman Ablation"/>' if charts.get("kalman_ablation") else ""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FlowMOEA-X v4.0 Experiment Report</title>
<style>
  :root {{ --bg:#f8fafc; --card:#ffffff; --text:#0f172a; --muted:#475569; --accent:#0ea5e9; --border:#e2e8f0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background:var(--bg); color:var(--text); margin:0; padding:24px; }}
  .wrap {{ max-width:1240px; margin:0 auto; }}
  header {{ background:linear-gradient(135deg,#0f172a,#1e293b); color:#fff; padding:26px 32px; border-radius:14px; margin-bottom:24px; }}
  header h1 {{ margin:0 0 8px 0; font-size:24px; }}
  header p {{ margin:0; color:#cbd5e1; font-size:14px; }}
  h2 {{ margin:28px 0 12px 0; font-size:19px; border-left:4px solid var(--accent); padding-left:10px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(460px, 1fr)); gap:18px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; box-shadow:0 1px 3px rgba(15,23,42,.05); }}
  .card img {{ width:100%; height:auto; border-radius:6px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:left; }}
  th {{ background:#f1f5f9; font-weight:600; color:#334155; }}
  .badge.best {{ background:#0ea5e9; color:#fff; padding:2px 8px; border-radius:999px; font-weight:700; font-size:12px; }}
  .win {{ color:#059669; font-weight:700; }}
  .loss {{ color:#dc2626; font-weight:700; }}
  .tie {{ color:#64748b; }}
  .filter-bar {{ margin-bottom:10px; display:flex; gap:10px; }}
  .filter-bar input {{ padding:6px 10px; border:1px solid var(--border); border-radius:6px; width:260px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>FlowMOEA-X v4.0 — 全张量动态多目标优化实验总报告</h1>
    <p>Generated: {timestamp} &nbsp;|&nbsp; Device: {html.escape(str(meta.get('device', 'cpu')))} &nbsp;|&nbsp;
       Seeds (B): {meta.get('n_seeds', '-')} &nbsp;|&nbsp; Pop (N): {meta.get('pop_size', '-')} &nbsp;|&nbsp;
       &tau;_t = {meta.get('tau_t', '-')}, n_t = {meta.get('n_t', '-')}, Envs = {meta.get('n_envs', '-')}</p>
  </header>

  <h2>1. 全局总览看板（Friedman 秩检验 · Nemenyi CD · Wilcoxon W/T/L · 耗时）</h2>
  <div class="grid">
    <div class="card">
      <h3>Friedman Average Ranks &amp; Wilcoxon W/T/L (Control: {html.escape(str(meta.get('control', 'FlowMOEA-X')))})</h3>
      <p style="font-size:13px;color:#475569;">
        Friedman &chi;&sup2; = {friedman.get('chi2_stat', 0.0):.3f} (p = {friedman.get('p_value', 1.0):.4g}) &nbsp;|&nbsp;
        Iman-Davenport F = {friedman.get('iman_davenport_f', 0.0):.3f} &nbsp;|&nbsp;
        Nemenyi CD(&alpha;=0.05) = {friedman.get('nemenyi_cd', 0.0):.3f}
      </p>
      <table>
        <thead><tr><th>Rank</th><th>Algorithm</th><th>Avg Rank</th><th>W / T / L vs Control</th></tr></thead>
        <tbody>{rank_rows_html}</tbody>
      </table>
    </div>
    <div class="card"><h3>Nemenyi Critical Difference Diagram</h3>{cd_img}</div>
  </div>
  <div class="card" style="margin-top:18px;">
    <h3>Per-Problem Benchmark Results (MIGD &darr; · MHV &uarr; · WSRR &uarr;)</h3>
    <div class="filter-bar"><input id="tblFilter" type="text" placeholder="Filter by problem or algorithm..."/></div>
    <table id="summaryTbl">
      <thead><tr><th>Problem</th><th>Algorithm</th><th>MIGD (mean &plusmn; std)</th><th>MHV (mean &plusmn; std)</th><th>WSRR</th><th>Transfer Time</th><th>Sig.</th></tr></thead>
      <tbody>{detail_rows_html}</tbody>
    </table>
  </div>

  <h2>2. 分维卡尔曼动量步长消融专题（v3.1 标量 vs v4.0 分维解耦）</h2>
  <div class="grid">
    <div class="card">{kal_img}</div>
    <div class="card">{lat_img}</div>
  </div>

  <h2>3. 逐代 IGD 恢复动力学曲线库</h2>
  <div class="grid">{recovery_html}</div>

  <h2>4. 严格函数评估预算账本审计表（Zero Probe Leakage）</h2>
  <div class="card">
    <table>
      <thead><tr><th>Problem</th><th>Algorithm</th><th>Sensor FE</th><th>Warmstart FE</th><th>Evolution FE</th><th>Total FE / Seed</th><th>Strict Audit</th></tr></thead>
      <tbody>{ledger_rows_html}</tbody>
    </table>
  </div>
</div>
<script>
document.getElementById('tblFilter').addEventListener('input', function(e) {{
  var q = e.target.value.toLowerCase();
  document.querySelectorAll('#summaryTbl tbody tr').forEach(function(tr) {{
    tr.style.display = tr.innerText.toLowerCase().indexOf(q) >= 0 ? '' : 'none';
  }});
}});
</script>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    # Also persist machine-readable JSON sidecar (without large base64 strings)
    json_sidecar = out.with_suffix(".json")
    clean = {k: v for k, v in report_data.items() if k != "charts"}
    json_sidecar.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
