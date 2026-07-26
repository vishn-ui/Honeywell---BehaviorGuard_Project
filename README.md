# BehaviorGuard
### AI-Powered Behavioral Anomaly Detection for Cybersecurity — Working Prototype

An end-to-end pipeline that learns what "normal" access behavior looks like for a
user, service account, or device, flags deviations in near real time, classifies
*which* attack pattern an anomaly resembles, explains *why* it was flagged, and
surfaces everything in an analyst dashboard.

## Quick start

```bash
bash run_all.sh                       # generates data, trains the model, scores alerts
cd dashboard && streamlit run app.py  # opens the analyst console in your browser
```

Each stage can also be run individually — see `report/REPORT.md` section 5.

## Project layout

```
behaviorguard/
├── data/
│   └── generate_data.py     # Deliverable 1: synthetic data generator + attack taxonomy
├── src/
│   ├── features.py          # Deliverable 2: per-entity baseline profiling model
│   ├── dataset.py           # sliding-window sequence construction + chronological split
│   ├── model.py             # Deliverable 3: LSTM sequence-aware detector
│   ├── train.py             # training + Deliverable 3/4 evaluation (imbalance-aware)
│   ├── explain.py           # Deliverable 5: explainability / feature attribution
│   └── infer.py             # scores events, writes the ranked alert queue
├── dashboard/
│   └── app.py                # Deliverable 6: Streamlit analyst-facing dashboard
├── report/
│   ├── generate_report.py    # builds REPORT.md from the live metrics.json
│   └── REPORT.md             # Deliverable 7: assumptions, metrics, limitations
├── models/                   # trained checkpoint + metrics.json (created by train.py)
└── run_all.sh                 # one-command pipeline runner
```

## How each problem-statement requirement is addressed

| Requirement | Where |
|---|---|
| Sequential/behavioral data, not static snapshots | `dataset.py` builds sliding-window sequences per entity |
| Extreme class imbalance | `train.py` uses inverse-frequency class weights; `generate_data.py` targets ~2% anomaly rate |
| Concept drift | Chronological (not random) train/val/test split in `dataset.py` |
| Explainability | `explain.py` — deviation-feature attribution, plain-language phrasing |
| Cold-start | `features.py`'s `BaselineProfiler` falls back to population-level priors under 5 prior events |

## Notes for the hackathon demo

- Default run generates 300 entities × 45 days (~65k events, ~2% anomalous) — trains
  in well under a minute on CPU.
- `models/metrics.json` holds the exact numbers embedded in `report/REPORT.md` so the
  report and the dashboard's "Model evaluation details" panel always agree.
- To make the demo more dramatic, lower `--anomaly_rate` further or increase
  `--n_entities` / `--days` in `run_all.sh`.
