# evaluation/report.py
"""
Turns the combined report dict produced by run_eval.py (routing rows +
compute_routing_metrics() output, RAG rows + compute_ragas_style_metrics()
output) into a single self-contained, rich HTML dashboard - charts via
Chart.js (CDN, no local install), dark theme, industry-benchmark comparison.

No new dependencies beyond the stdlib - this only touches what's already in
`combined_report` (see run_eval.py's `main()`), so it stays a drop-in
addition to the existing four-file structure.

Usage:
    from evaluation.report import build_and_write_report
    build_and_write_report(combined_report, REPORT_HTML_PATH)   # called from run_eval.py

or standalone, from an existing eval_report.json:
    python evaluation/report.py                # reads eval_report.json, writes eval_report.html
    python evaluation/report.py path/to/x.json  # reads a specific json file instead
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "eval_report.json"
DEFAULT_HTML_PATH = Path(__file__).resolve().parent / "eval_report.html"

# --- Reference ranges only, for a directional comparison - see the report's
# own footnote. Not a certified external benchmark; adjust to your own SLA
# once you have real production data to calibrate against. ---
ROUTING_BENCHMARK = {"good": 0.90, "acceptable": 0.75, "poor": 0.60}
RAG_BENCHMARKS = {
    "context_precision": {"good": 0.85, "acceptable": 0.70, "poor": 0.50},
    "context_recall": {"good": 0.85, "acceptable": 0.70, "poor": 0.50},
    "faithfulness": {"good": 0.90, "acceptable": 0.75, "poor": 0.60},
    "answer_relevancy": {"good": 0.85, "acceptable": 0.65, "poor": 0.45},
}

BAND_COLOR = {"good": "#1f9d55", "acceptable": "#c8930a", "poor": "#d9622b", "critical": "#c0392b", "n/a": "#666"}
BAND_LABEL = {"good": "Good", "acceptable": "Acceptable", "poor": "Needs Improvement", "critical": "Critical", "n/a": "N/A"}


def _band(value: Optional[float], thresholds: Dict[str, float]) -> str:
    if value is None:
        return "n/a"
    if value >= thresholds["good"]:
        return "good"
    if value >= thresholds["acceptable"]:
        return "acceptable"
    if value >= thresholds["poor"]:
        return "poor"
    return "critical"


def _badge(band: str) -> str:
    color = BAND_COLOR[band]
    return f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">{BAND_LABEL[band]}</span>'


def _fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _confusion_matrix(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Built here (not in metrics.py) purely for the report - a simple
    expected-vs-actual count grid from the routing rows already computed."""
    labels = sorted({r["expected_agent"] for r in rows} | {r["actual_agent"] for r in rows if r["actual_agent"]})
    index = {label: i for i, label in enumerate(labels)}
    grid = [[0] * len(labels) for _ in labels]
    for r in rows:
        actual = r["actual_agent"] or "None"
        if actual not in index:
            index[actual] = len(labels)
            labels.append(actual)
            for row in grid:
                row.append(0)
            grid.append([0] * len(labels))
        grid[index[r["expected_agent"]]][index[actual]] += 1
    return {"labels": labels, "grid": grid}


def build_html(data: Dict[str, Any]) -> str:
    routing = data["routing"]
    rag = data["rag"]

    routing_rows = routing["rows"]
    routing_metrics = routing["metrics"]
    has_routing_error = "error" in routing_metrics

    rag_rows = rag["rows"]
    rag_agg = rag["aggregates"]

    # ---- routing KPIs / bands ----
    if not has_routing_error:
        acc_band = _band(routing_metrics["accuracy"], ROUTING_BENCHMARK)
        f1_macro_band = _band(routing_metrics["f1_macro"], ROUTING_BENCHMARK)
        per_class = routing_metrics["per_class"]
    else:
        acc_band = f1_macro_band = "n/a"
        per_class = {}

    # ---- confusion matrix ----
    cm = _confusion_matrix(routing_rows) if routing_rows else {"labels": [], "grid": []}
    cm_max = max((max(row) for row in cm["grid"]), default=1) or 1
    cm_header = "".join(f"<th>{l}</th>" for l in cm["labels"])
    cm_body = ""
    for i, label in enumerate(cm["labels"]):
        cells = ""
        for j, val in enumerate(cm["grid"][i]):
            diag = i == j
            intensity = val / cm_max
            bg = f"rgba(58,123,213,{0.12 + intensity * 0.55})" if diag else f"rgba(217,98,43,{0.10 + intensity * 0.45})"
            cells += f'<td style="background:{bg}">{val}</td>'
        cm_body += f"<tr><th>{label}</th>{cells}</tr>"

    # ---- routing rows table ----
    routing_rows_html = "".join(
        f'<tr class="{"pass" if r["correct"] else "fail"}">'
        f'<td>{r["query"]}</td><td>{r["expected_agent"]}</td><td>{r["actual_agent"] or "-"}</td>'
        f'<td>{"&#10003;" if r["correct"] else "&#10007;"}</td></tr>'
        for r in routing_rows
    )

    # ---- per-class table ----
    per_class_rows_html = "".join(
        f"<tr><td>{label}</td><td>{s['precision']}</td><td>{s['recall']}</td><td>{s['f1']}</td><td>{s['support']}</td></tr>"
        for label, s in per_class.items()
    )

    # ---- RAG rows table ----
    rag_bands = {
        "context_precision": _band(rag_agg["avg_context_precision"], RAG_BENCHMARKS["context_precision"]),
        "context_recall": _band(rag_agg["avg_context_recall"], RAG_BENCHMARKS["context_recall"]),
        "faithfulness": _band(rag_agg["avg_faithfulness"], RAG_BENCHMARKS["faithfulness"]),
        "answer_relevancy": _band(rag_agg["avg_answer_relevancy"], RAG_BENCHMARKS["answer_relevancy"]),
    }
    rag_rows_html = "".join(
        f"<tr><td>{r['question']}</td><td>{r['answer'][:140]}{'...' if len(r['answer']) > 140 else ''}</td>"
        f"<td>{r['n_contexts']}</td><td>{_fmt(r['context_precision'])}</td><td>{_fmt(r['context_recall'])}</td>"
        f"<td>{_fmt(r['faithfulness'])}</td><td>{_fmt(r['answer_relevancy'])}</td></tr>"
        for r in rag_rows
    )

    # ---- chart data ----
    routing_chart = {
        "labels": ["Accuracy", "F1 Macro", "F1 Weighted"],
        "shopassist": [routing_metrics.get("accuracy"), routing_metrics.get("f1_macro"), routing_metrics.get("f1_weighted")] if not has_routing_error else [0, 0, 0],
        "benchmark": [ROUTING_BENCHMARK["good"]] * 3,
    }
    per_class_labels = list(per_class.keys())
    per_class_f1 = [s["f1"] for s in per_class.values()]

    rag_metric_keys = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    rag_chart = {
        "labels": ["Context Precision", "Context Recall", "Faithfulness", "Answer Relevancy"],
        "shopassist": [rag_agg[f"avg_{k}"] if rag_agg[f"avg_{k}"] is not None else 0 for k in rag_metric_keys],
        "good": [RAG_BENCHMARKS[k]["good"] for k in rag_metric_keys],
        "acceptable": [RAG_BENCHMARKS[k]["acceptable"] for k in rag_metric_keys],
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ShopAssist &middot; Evaluation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0b0f19; --panel: #121828; --panel-2: #171f33; --border: #232c45;
    --text: #e7ecf7; --muted: #8b95ab;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #16213f 0%, var(--bg) 60%);
    color: var(--text); padding: 40px 24px 80px;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  header h1 {{
    font-size: 30px; margin: 0 0 6px;
    background: linear-gradient(90deg, #8fb8ff, #c6b3ff);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  header p {{ color: var(--muted); margin: 0 0 32px; font-size: 14px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{ background: linear-gradient(180deg, var(--panel), var(--panel-2)); border: 1px solid var(--border); border-radius: 14px; padding: 20px; }}
  .kpi .label {{ color: var(--muted); font-size: 12.5px; text-transform: uppercase; letter-spacing: .06em; }}
  .kpi .value {{ font-size: 34px; font-weight: 700; margin: 6px 0 8px; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  section {{ margin-bottom: 44px; }}
  section h2 {{ font-size: 20px; margin-bottom: 4px; }}
  section .sub {{ color: var(--muted); font-size: 13.5px; margin-bottom: 18px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 24px; }}
  .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 900px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}
  canvas {{ max-height: 320px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 12px; }}
  th, td {{ padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
  .cm-table th, .cm-table td {{ text-align: center; }}
  .cm-table th:first-child, .cm-table td:first-child {{ text-align: left; }}
  tr.fail td:first-child {{ border-left: 3px solid #c0392b; }}
  tr.pass td:first-child {{ border-left: 3px solid #1f9d55; }}
  .note {{ font-size: 12.5px; color: var(--muted); margin-top: 10px; line-height: 1.5; }}
  footer {{ color: var(--muted); font-size: 12px; text-align: center; margin-top: 50px; }}
  pre {{ background: #0d1220; border: 1px solid var(--border); border-radius: 10px; padding: 14px; overflow-x: auto; font-size: 12px; color: #c7d0e5; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>ShopAssist &middot; Evaluation Report</h1>
    <p>Generated from evaluation/eval_report.json &middot; routing scored end-to-end through the real AgentOrchestratorService &middot; RAGAS-style RAG metrics via MockRAGService's actual retrieval &middot; benchmarked against general industry reference ranges.</p>
  </header>

  <div class="grid">
    <div class="card kpi">
      <div class="label">Routing Accuracy</div>
      <div class="value">{_fmt(routing_metrics.get('accuracy'))}</div>
      {_badge(acc_band)}
    </div>
    <div class="card kpi">
      <div class="label">Routing F1 (macro)</div>
      <div class="value">{_fmt(routing_metrics.get('f1_macro'))}</div>
      {_badge(f1_macro_band)}
    </div>
    <div class="card kpi">
      <div class="label">RAG Faithfulness</div>
      <div class="value">{_fmt(rag_agg['avg_faithfulness'])}</div>
      {_badge(rag_bands['faithfulness'])}
    </div>
    <div class="card kpi">
      <div class="label">RAG Answer Relevancy</div>
      <div class="value">{_fmt(rag_agg['avg_answer_relevancy'])}</div>
      {_badge(rag_bands['answer_relevancy'])}
    </div>
  </div>

  <section>
    <h2>1. Intent Routing</h2>
    <div class="sub">Real AgentOrchestratorService.handle_customer_query() calls &middot; {len(routing_rows)} hand-labeled cases from eval_dataset.py</div>
    <div class="panel">
      <div class="charts-row">
        <div><canvas id="routingChart"></canvas></div>
        <div><canvas id="perClassChart"></canvas></div>
      </div>
      <table>
        <tr><th>Agent</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
        {per_class_rows_html}
      </table>
      <div class="note">Benchmark line = commonly cited "good" macro-F1/accuracy (&ge; {ROUTING_BENCHMARK['good']}) for a mature production intent router - a general reference range, not a certified external standard.</div>
    </div>
  </section>

  <section>
    <h2>2. Confusion Matrix</h2>
    <div class="sub">Rows = expected agent, columns = actual agent (response.agent_invoked). Off-diagonal cells are misroutes.</div>
    <div class="panel">
      <table class="cm-table">
        <tr><th>Expected \\ Actual</th>{cm_header}</tr>
        {cm_body}
      </table>
    </div>
  </section>

  <section>
    <h2>3. Routing Cases</h2>
    <div class="panel">
      <table>
        <tr><th>Query</th><th>Expected</th><th>Actual</th><th>&nbsp;</th></tr>
        {routing_rows_html}
      </table>
    </div>
  </section>

  <section>
    <h2>4. RAG / RAGAS-style Evaluation</h2>
    <div class="sub">GeneralPurposeAgent + MockRAGService's real retrieval &middot; {len(rag_rows)} cases from eval_dataset.py &middot; context precision/recall via keyword-coverage proxy, faithfulness via token overlap, answer relevancy via LLMInferenceService.call_embeddings() cosine similarity</div>
    <div class="panel">
      <canvas id="ragChart"></canvas>
      <table>
        <tr><th>Question</th><th>Answer</th><th>#Ctx</th><th>Ctx Precision</th><th>Ctx Recall</th><th>Faithfulness</th><th>Answer Relevancy</th></tr>
        {rag_rows_html}
      </table>
      <div class="note">These are lightweight heuristic stand-ins for the real RAGAS/LLM-judged metrics - see metrics.py's module docstring for the exact reasoning and upgrade path (swap in real <a href="https://docs.ragas.io" target="_blank" style="color:#8fb8ff">ragas</a> calls behind the same function signatures once LLMInferenceService is wired through an OpenAI-shaped/LangChain client).</div>
    </div>
  </section>

  <section>
    <h2>5. Industry Benchmark Comparison</h2>
    <div class="sub">General reference ranges for production RAG / support-chatbot systems &mdash; directional guidance, not a certified benchmark suite.</div>
    <div class="panel">
      <table>
        <tr><th>Metric</th><th>ShopAssist</th><th>Good</th><th>Acceptable</th><th>Poor</th><th>Band</th></tr>
        <tr><td>Routing Accuracy</td><td>{_fmt(routing_metrics.get('accuracy'))}</td><td>{ROUTING_BENCHMARK['good']}</td><td>{ROUTING_BENCHMARK['acceptable']}</td><td>{ROUTING_BENCHMARK['poor']}</td><td>{_badge(acc_band)}</td></tr>
        <tr><td>Routing F1 (macro)</td><td>{_fmt(routing_metrics.get('f1_macro'))}</td><td>{ROUTING_BENCHMARK['good']}</td><td>{ROUTING_BENCHMARK['acceptable']}</td><td>{ROUTING_BENCHMARK['poor']}</td><td>{_badge(f1_macro_band)}</td></tr>
        <tr><td>RAG Context Precision</td><td>{_fmt(rag_agg['avg_context_precision'])}</td><td>{RAG_BENCHMARKS['context_precision']['good']}</td><td>{RAG_BENCHMARKS['context_precision']['acceptable']}</td><td>{RAG_BENCHMARKS['context_precision']['poor']}</td><td>{_badge(rag_bands['context_precision'])}</td></tr>
        <tr><td>RAG Context Recall</td><td>{_fmt(rag_agg['avg_context_recall'])}</td><td>{RAG_BENCHMARKS['context_recall']['good']}</td><td>{RAG_BENCHMARKS['context_recall']['acceptable']}</td><td>{RAG_BENCHMARKS['context_recall']['poor']}</td><td>{_badge(rag_bands['context_recall'])}</td></tr>
        <tr><td>RAG Faithfulness</td><td>{_fmt(rag_agg['avg_faithfulness'])}</td><td>{RAG_BENCHMARKS['faithfulness']['good']}</td><td>{RAG_BENCHMARKS['faithfulness']['acceptable']}</td><td>{RAG_BENCHMARKS['faithfulness']['poor']}</td><td>{_badge(rag_bands['faithfulness'])}</td></tr>
        <tr><td>RAG Answer Relevancy</td><td>{_fmt(rag_agg['avg_answer_relevancy'])}</td><td>{RAG_BENCHMARKS['answer_relevancy']['good']}</td><td>{RAG_BENCHMARKS['answer_relevancy']['acceptable']}</td><td>{RAG_BENCHMARKS['answer_relevancy']['poor']}</td><td>{_badge(rag_bands['answer_relevancy'])}</td></tr>
      </table>
    </div>
  </section>

  <section>
    <h2>6. Full sklearn Classification Report</h2>
    <div class="panel">
      <pre>{routing_metrics.get('classification_report', 'N/A')}</pre>
    </div>
  </section>

  <footer>Generated by evaluation/report.py from evaluation/eval_report.json</footer>
</div>

<script>
const routingData = {json.dumps(routing_chart)};
const ragData = {json.dumps(rag_chart)};
const perClassLabels = {json.dumps(per_class_labels)};
const perClassF1 = {json.dumps(per_class_f1)};

new Chart(document.getElementById('routingChart'), {{
  type: 'bar',
  data: {{
    labels: routingData.labels,
    datasets: [
      {{ label: 'ShopAssist', data: routingData.shopassist, backgroundColor: '#3a7bd5' }},
      {{ label: 'Industry "Good" Benchmark', data: routingData.benchmark, backgroundColor: 'rgba(140,150,170,0.35)' }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#e7ecf7' }} }}, title: {{ display: true, text: 'Routing Metrics vs Benchmark', color: '#e7ecf7' }} }},
    scales: {{ y: {{ beginAtZero: true, max: 1, ticks: {{ color: '#8b95ab' }}, grid: {{ color: '#232c45' }} }}, x: {{ ticks: {{ color: '#8b95ab' }}, grid: {{ display: false }} }} }}
  }}
}});

new Chart(document.getElementById('perClassChart'), {{
  type: 'radar',
  data: {{
    labels: perClassLabels,
    datasets: [{{ label: 'F1 per Agent', data: perClassF1, backgroundColor: 'rgba(124,92,255,0.25)', borderColor: '#7c5cff', pointBackgroundColor: '#7c5cff' }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#e7ecf7' }} }}, title: {{ display: true, text: 'F1 Score by Agent', color: '#e7ecf7' }} }},
    scales: {{ r: {{ min: 0, max: 1, angleLines: {{ color: '#232c45' }}, grid: {{ color: '#232c45' }}, pointLabels: {{ color: '#e7ecf7', font: {{ size: 11 }} }}, ticks: {{ display: false }} }} }}
  }}
}});

new Chart(document.getElementById('ragChart'), {{
  type: 'bar',
  data: {{
    labels: ragData.labels,
    datasets: [
      {{ label: 'ShopAssist', data: ragData.shopassist, backgroundColor: '#3a7bd5' }},
      {{ label: 'Industry "Good"', data: ragData.good, backgroundColor: 'rgba(31,157,85,0.35)' }},
      {{ label: 'Industry "Acceptable"', data: ragData.acceptable, backgroundColor: 'rgba(200,147,10,0.30)' }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#e7ecf7' }} }}, title: {{ display: true, text: 'RAGAS-style Metrics vs Benchmark', color: '#e7ecf7' }} }},
    scales: {{ y: {{ beginAtZero: true, max: 1, ticks: {{ color: '#8b95ab' }}, grid: {{ color: '#232c45' }} }}, x: {{ ticks: {{ color: '#8b95ab' }}, grid: {{ display: false }} }} }}
  }}
}});
</script>
</body>
</html>
"""
    return html


def build_and_write_report(data: Dict[str, Any], output_path: Path = DEFAULT_HTML_PATH) -> Path:
    """Called directly from run_eval.py's main() with the in-memory
    combined_report dict - no subprocess, no re-reading the JSON off disk."""
    output_path.write_text(build_html(data))
    return output_path


def main() -> None:
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON_PATH
    data = json.loads(json_path.read_text())
    output_path = build_and_write_report(data)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
