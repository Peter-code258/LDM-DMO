"""Compile the V4.0 McKinsey/Bain-styled PhD-level undergraduate thesis Markdown
into a single self-contained, print-ready executive HTML report with embedded
300-DPI base64 charts and KaTeX math rendering.
"""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS_DIR = ROOT / "thesis_v4"
MD_PATH = THESIS_DIR / "FlowMOEA-X_V4_本科毕业论文_麦肯锡咨询风格.md"
HTML_PATH = THESIS_DIR / "FlowMOEA-X_V4_本科毕业论文_麦肯锡咨询风格.html"


def _embed_img(rel_path: str) -> str:
    p = (THESIS_DIR / rel_path).resolve()
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _inline_md(text: str) -> str:
    # Protect LaTeX $...$ spans from formatting substitutions
    math_tokens: list[str] = []

    def _stash_math(m: re.Match) -> str:
        math_tokens.append(m.group(0))
        return f"@@MATH{len(math_tokens) - 1}@@"

    text = re.sub(r"\$\$[^$]+\$\$|\$[^$\n]+\$", _stash_math, text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)

    for idx, tok in enumerate(math_tokens):
        text = text.replace(f"@@MATH{idx}@@", tok)
    return text


def md_to_html_body(md_text: str) -> tuple[str, list[tuple[str, str]]]:
    lines = md_text.splitlines()
    out: list[str] = []
    toc: list[tuple[str, str]] = []
    i = 0
    n = len(lines)
    sec_id = 0

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            out.append("<hr class='divider'/>")
            i += 1
            continue

        # Display math block $$...$$ (single line or multi-line)
        if stripped.startswith("$$"):
            block = [stripped]
            if not (stripped.endswith("$$") and len(stripped) > 2):
                i += 1
                while i < n and not lines[i].strip().endswith("$$"):
                    block.append(lines[i])
                    i += 1
                if i < n:
                    block.append(lines[i])
            out.append("<div class='math-block'>" + "\n".join(block) + "</div>")
            i += 1
            continue

        # Headings
        m_h = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m_h:
            lvl = len(m_h.group(1))
            title = m_h.group(2).strip()
            sec_id += 1
            anchor = f"sec-{sec_id}"
            if lvl == 2:
                toc.append((anchor, title))
            if lvl == 1:
                # Skip top H1 in body since we render a dedicated executive cover header
                i += 1
                continue
            out.append(f"<h{lvl} id='{anchor}'>{_inline_md(title)}</h{lvl}>")
            i += 1
            continue

        # Image line: ![alt](path)
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m_img:
            alt, path = m_img.group(1), m_img.group(2)
            uri = _embed_img(path)
            caption = ""
            if i + 2 < n and not lines[i + 1].strip() and lines[i + 2].strip().startswith("*图"):
                caption = lines[i + 2].strip().strip("*")
                i += 2
            elif i + 1 < n and lines[i + 1].strip().startswith("*图"):
                caption = lines[i + 1].strip().strip("*")
                i += 1
            cap_html = f"<figcaption>{_inline_md(caption)}</figcaption>" if caption else f"<figcaption>{html.escape(alt)}</figcaption>"
            out.append(f"<figure class='chart-card'><img src='{uri}' alt='{html.escape(alt)}'/>{cap_html}</figure>")
            i += 1
            continue

        # Blockquote (McKinsey Key Takeaway / Callout box)
        if stripped.startswith(">"):
            bq = []
            while i < n and lines[i].strip().startswith(">"):
                bq.append(lines[i].strip()[1:].strip())
                i += 1
            content = "<br/>".join(_inline_md(x) for x in bq)
            out.append(f"<div class='callout'>{content}</div>")
            continue

        # Markdown Table
        if stripped.startswith("|") and i + 1 < n and lines[i + 1].strip().startswith("|") and "---" in lines[i + 1]:
            hdr = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            ths = "".join(f"<th>{_inline_md(c)}</th>" for c in hdr)
            trs = ""
            for r in rows:
                tds = "".join(f"<td>{_inline_md(c)}</td>" for c in r)
                trs += f"<tr>{tds}</tr>"
            out.append(f"<div class='tbl-wrap'><table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div>")
            continue

        # Unordered list
        if stripped.startswith("* ") or stripped.startswith("- "):
            items = []
            while i < n and (lines[i].strip().startswith("* ") or lines[i].strip().startswith("- ")):
                items.append(_inline_md(lines[i].strip()[2:]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{it}</li>" for it in items) + "</ul>")
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(_inline_md(re.sub(r"^\d+\.\s+", "", lines[i].strip())))
                i += 1
            out.append("<ol>" + "".join(f"<li>{it}</li>" for it in items) + "</ol>")
            continue

        # Paragraph
        para = [stripped]
        i += 1
        while (
            i < n
            and lines[i].strip()
            and not lines[i].strip().startswith(("#", "|", ">", "* ", "- ", "![", "$$", "---"))
            and not re.match(r"^\d+\.\s+", lines[i].strip())
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline_md(' '.join(para))}</p>")

    return "\n".join(out), toc


def compile_thesis() -> Path:
    md_text = MD_PATH.read_text(encoding="utf-8")
    body_html, toc = md_to_html_body(md_text)
    nav_html = "".join(
        f"<a href='#{anc}'>{html.escape(title)}</a>" for anc, title in toc
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>FlowMOEA-X v4.0 — 本科毕业论文（麦肯锡/贝恩战略咨询报告版）</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
  :root {{
    --navy: #0F2942;
    --blue: #0077B6;
    --cyan: #00B4D8;
    --coral: #D64545;
    --slate: #475569;
    --bg: #F4F7FA;
    --card: #FFFFFF;
    --border: #DCE4EE;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Microsoft YaHei", "PingFang SC", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: #1E293B;
    line-height: 1.75;
    font-size: 15px;
  }}
  .layout {{ display: flex; max-width: 1440px; margin: 0 auto; }}
  .sidebar {{
    width: 270px;
    flex-shrink: 0;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    background: var(--navy);
    color: #E2E8F0;
    padding: 26px 18px;
  }}
  .sidebar .brand {{ font-size: 13px; letter-spacing: 1.6px; text-transform: uppercase; color: var(--cyan); font-weight: 700; margin-bottom: 6px; }}
  .sidebar .sub {{ font-size: 15px; font-weight: 700; color: #FFF; margin-bottom: 18px; line-height: 1.4; }}
  .sidebar a {{
    display: block;
    color: #CBD5E1;
    text-decoration: none;
    font-size: 12.8px;
    padding: 7px 10px;
    border-radius: 6px;
    margin-bottom: 4px;
    border-left: 3px solid transparent;
    transition: all 0.15s;
  }}
  .sidebar a:hover {{ background: rgba(255,255,255,0.08); color: #FFF; border-left-color: var(--cyan); }}
  .main {{ flex: 1; padding: 30px 44px 70px 44px; min-width: 0; }}
  .cover {{
    background: linear-gradient(135deg, #091B2C 0%, #0F2942 58%, #005B8F 100%);
    color: #FFFFFF;
    border-radius: 14px;
    padding: 38px 42px 32px 42px;
    margin-bottom: 26px;
    box-shadow: 0 10px 28px rgba(15,41,66,0.16);
  }}
  .cover .eyebrow {{ font-size: 12px; letter-spacing: 2px; text-transform: uppercase; color: #7DD3FC; font-weight: 700; }}
  .cover h1 {{ margin: 10px 0 8px 0; font-size: 27px; line-height: 1.32; font-weight: 800; }}
  .cover .en-title {{ font-size: 13.5px; color: #CBD5E1; font-style: italic; margin-bottom: 22px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
  .kpi {{
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 10px;
    padding: 14px 16px;
  }}
  .kpi .val {{ font-size: 24px; font-weight: 800; color: #38BDF8; line-height: 1.15; }}
  .kpi .lbl {{ font-size: 12px; color: #E2E8F0; margin-top: 4px; }}
  .paper {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 36px 46px;
    box-shadow: 0 4px 16px rgba(15,41,66,0.04);
  }}
  h2 {{
    color: var(--navy);
    font-size: 21px;
    font-weight: 800;
    margin: 38px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2.5px solid var(--navy);
  }}
  h3 {{ color: var(--blue); font-size: 17px; font-weight: 700; margin: 26px 0 10px 0; }}
  h4 {{ color: var(--navy); font-size: 15px; font-weight: 700; margin: 18px 0 8px 0; }}
  p {{ margin: 10px 0; text-align: justify; }}
  .callout {{
    background: linear-gradient(90deg, #EDF6FC 0%, #F8FBFE 100%);
    border-left: 5px solid var(--blue);
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0 20px 0;
    color: var(--navy);
    font-size: 14.3px;
  }}
  .chart-card {{
    margin: 22px 0;
    padding: 16px 18px 12px 18px;
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(15,41,66,0.05);
    text-align: center;
  }}
  .chart-card img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  .chart-card figcaption {{
    margin-top: 10px;
    font-size: 13px;
    color: var(--slate);
    text-align: left;
    line-height: 1.55;
    padding-top: 8px;
    border-top: 1px dashed var(--border);
  }}
  .tbl-wrap {{ overflow-x: auto; margin: 16px 0 22px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: var(--navy); color: #FFFFFF; padding: 9px 10px; text-align: left; font-weight: 600; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  tr:nth-child(even) td {{ background: #F8FAFC; }}
  code {{ background: #EFF6FF; color: #0369A1; padding: 1px 5px; border-radius: 4px; font-size: 13px; }}
  .math-block {{ margin: 14px 0; padding: 10px 14px; background: #F8FAFC; border-radius: 8px; overflow-x: auto; text-align: center; }}
  hr.divider {{ border: none; border-top: 1px solid var(--border); margin: 28px 0; }}
  ul, ol {{ padding-left: 22px; margin: 10px 0; }}
  li {{ margin: 5px 0; }}
  @media print {{
    .sidebar {{ display: none; }}
    .main {{ padding: 0; }}
    .paper {{ border: none; box-shadow: none; padding: 10px; }}
    .chart-card {{ break-inside: avoid; }}
    h2 {{ break-after: avoid; }}
  }}
  @media (max-width: 960px) {{
    .sidebar {{ display: none; }}
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .main {{ padding: 16px; }}
    .paper {{ padding: 20px; }}
  }}
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">FlowMOEA-X · v4.0</div>
    <div class="sub">动态多目标优化本科毕业论文<br/>麦肯锡/贝恩战略咨询版</div>
    {nav_html}
  </aside>
  <main class="main">
    <section class="cover">
      <div class="eyebrow">UNDERGRADUATE THESIS · PHD-LEVEL RESEARCH RIGOR × MCKINSEY / BAIN EXECUTIVE EDITION · V4.0</div>
      <h1>面向动态多目标优化的连续流形最优传输与全张量进化架构（FlowMOEA-X v4.0）</h1>
      <div class="en-title">FlowMOEA-X: A Fully-Tensorised Dynamic Multi-Objective Optimization Architecture with Continuous Manifold Optimal Transport, Per-Dimension Kalman Momentum, and Zero-FE Surrogate Pre-Screening</div>
      <div class="kpi-grid">
        <div class="kpi"><div class="val">#1 (2.46)</div><div class="lbl">13 基准 Friedman 全局平均秩<br/>(p = 8.42e-6, 9W/4T/0L vs DNSGA2)</div></div>
        <div class="kpi"><div class="val">9.09×</div><div class="lbl">PyTorch 三维张量批处理加速<br/>(B=30 并发，0.035 s / 种子)</div></div>
        <div class="kpi"><div class="val">-66.1%</div><div class="lbl">高频波纹 DF5 误差降幅<br/>(三维 FDA4 降幅 -53.1%)</div></div>
        <div class="kpi"><div class="val">0 FE 泄漏</div><div class="lbl">FELedger 严格预算 100% 审计<br/>(134/134 自动化验收测试通过)</div></div>
      </div>
    </section>
    <article class="paper">
      {body_html}
    </article>
  </main>
</div>
</body>
</html>
"""
    HTML_PATH.write_text(page, encoding="utf-8")
    print(f"[OK] Compiled self-contained executive thesis HTML: {HTML_PATH.resolve()} ({HTML_PATH.stat().st_size:,} bytes)")
    return HTML_PATH


if __name__ == "__main__":
    compile_thesis()
