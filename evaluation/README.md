# ShopAssist Evaluation

Five files, no separate metrics/report subfolders:

```
evaluation/
├── eval_dataset.py       # hand-labeled ROUTING_CASES / RAG_CASES
├── metrics.py             # sklearn F1/precision/recall + RAGAS-style RAG metrics
├── report.py              # NEW: turns eval_report.json into a rich standalone HTML dashboard
├── run_eval.py             # wires up the real orchestrator/agents, runs both evals, writes JSON + HTML
├── requirements-eval.txt    # scikit-learn + numpy on top of the project's own requirements.txt
├── eval_report.json         # generated - the raw combined report
└── eval_report.html         # generated - the rich dashboard, open directly in a browser
```

## Run it

```bash
pip install -r requirements.txt -r evaluation/requirements-eval.txt
python evaluation/run_eval.py
```

This does everything in one command: rebuilds the local dev DB, ingests
sample docs into RAG, runs every case in `eval_dataset.py` through the real
`AgentOrchestratorService`, scores it, writes `eval_report.json`, and
writes `eval_report.html` right after (via `report.build_and_write_report()`
- no subprocess, called inline at the end of `run_eval.py`'s `main()`).

Then just open `evaluation/eval_report.html` in a browser - it's a
self-contained file (Chart.js loads from a CDN), no local server needed.

## report.py, if you want to regenerate the HTML without re-running eval

```bash
python evaluation/report.py                      # reads evaluation/eval_report.json
python evaluation/report.py path/to/other.json    # or point it at a specific file
```

Same function (`build_and_write_report`) either way - `run_eval.py` just
calls it directly instead of shelling out.

## What's in the report

- Routing accuracy / F1 (macro + weighted) + per-class precision/recall/F1
- A confusion matrix (expected vs. actual agent), built from the routing
  rows already in `eval_report.json` - not a separate metric, just a
  different view of the same data
- Full case-by-case tables for both routing and RAG cases
- The four RAGAS-style metrics (context precision/recall, faithfulness,
  answer relevancy) with per-case and averaged scores
- A comparison against general industry reference ranges (defined inline
  at the top of `report.py` - not a certified external benchmark, just
  directional guidance; adjust to your own SLA once you have real
  production data)

## Extending the dataset

Add cases directly to `ROUTING_CASES` / `RAG_CASES` in `eval_dataset.py` -
nothing else needs to change. `run_eval.py` and `report.py` both just
iterate over whatever's in those lists.
