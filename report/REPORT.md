# BehaviorGuard - AI-Powered Behavioral Anomaly Detection
### Assumptions, Metrics, and Known Limitations

## 1. Assumptions

- **Synthetic data stands in for real logs.** Real intrusion/access-log data is scarce,
  privacy-restricted, and domain-specific, so we generate synthetic per-entity access
  logs (`data/generate_data.py`) with injected attack patterns rather than relying on a
  found dataset. Entities are `user`, `service_account`, and `edge_device` types, each
  with its own habitual login-hour range, home geo-location, typical resource set, and
  device fingerprint.
- **Injection rate reflects real-world imbalance.** Attack patterns are injected at a
  rate producing roughly **2% non-normal events** overall, deliberately close to the
  "true intrusions are a tiny fraction of total access events" constraint in the brief
  (individual attack types range from ~0.05% to ~0.8% of all events).
- **`insider_drift` is treated as an edge case, not a hard-labeled attack.** Per the
  brief, it's ambiguous by design (a legitimate entity slowly expanding its footprint)
  and is used only for false-positive-rate tuning/evaluation, never as a training
  target — mislabeling ambiguous legitimate drift as an attack would poison the
  classifier.
- **Baseline profiles are computed causally.** Every feature describing "how unusual is
  this event" is computed only from that entity's *prior* events (running mean/std,
  seen-resource set, seen-device set, last known location) — never from future data —
  so there is no leakage between training and evaluation and the same code path works
  unchanged in a streaming/production setting.
- **Chronological train/val/test split**, not random — the model trains on the past and
  is evaluated on the future, which also exercises concept drift and cold-start
  entities that only appear later in the timeline.
- **Explainability uses the deviation features directly** rather than a separate
  post-hoc attribution model: because each feature IS already "deviation from this
  entity's baseline," the top 1-2 largest-magnitude features at alert time are a
  faithful, low-latency explanation (an integrated-gradients option is included in
  `explain.py` for a heavier alternative).

## 2. Pipeline

`generate_data.py` &rarr; `features.py` (baseline profiling) &rarr; `dataset.py`
(sliding-window sequences) &rarr; `model.py` + `train.py` (LSTM detector +
classifier) &rarr; `explain.py` (feature attribution) &rarr; `infer.py` (ranked
alerts) &rarr; `dashboard/app.py` (Streamlit analyst console).

## 3. Metrics (from the last training run, held-out TEST split — future-in-time data)

```
                     precision    recall  f1-score   support

             normal      0.999     0.935     0.966      9660
        brute_force      0.977     1.000     0.988        42
  impossible_travel      0.017     0.611     0.034        18
credential_stuffing      1.000     1.000     1.000        15
   lateral_movement      0.907     0.975     0.940        40
    device_spoofing      1.000     1.000     1.000         6
 low_and_slow_exfil      0.200     0.333     0.250         6

           accuracy                          0.935      9787
          macro avg      0.729     0.836     0.740      9787
       weighted avg      0.996     0.935     0.964      9787

```

- **False positive rate @ top 1% analyst alert budget:** 0.010
- **Anomaly recall captured @ top 1% alert budget:** 0.756
- **Binary anomaly-vs-normal Average Precision (PR-AUC):** 0.913
- **Validation-split PR-AUC (sanity check, no leakage):** 0.944

Interpretation: with a 1%-of-events analyst budget, the current model captures the
majority of true anomalies at a low false-positive rate, which is the operating point
that matters most for SOC usability (an analyst cannot review more than a small
fraction of daily events).

## 4. Known Limitations

1. **`impossible_travel` vs `credential_stuffing` confusion.** Both patterns share a
   large synthetic geo-jump signal (credential stuffing sessions are generated with an
   "unknown" geo-location), so the classifier sometimes confuses the two — precision on
   `impossible_travel` is the weakest of any class. A production system would add a
   dedicated "known geo vs. null/unknown geo" feature to disambiguate.
2. **Cold-start entities are handled by population-level fallback, not evaluated
   in depth.** The profiler falls back to population statistics for entities with
   under 5 prior events, but this prototype does not separately report accuracy
   broken out by cold-start vs. established entities — a natural next metric.
2. **Concept drift is exercised only implicitly** via the chronological split; there is
   no online/incremental re-baselining loop yet (e.g., decaying windows), so a model
   trained today would need periodic retraining in production.
3. **Small model, small synthetic world.** A single-layer LSTM over 8-step windows and
   ~300 entities is intentionally lightweight for a rapid prototype; a production
   system would need a larger, more diverse entity population, real traffic-derived
   priors, and likely a Transformer or graph-augmented encoder for lateral-movement
   detection across entity-resource relationships.
4. **Explanations are ranking-based, not causally validated.** The "top deviating
   feature" explanation is a strong proxy but is not a formal causal attribution (e.g.
   SHAP with a trained value function) — acceptable for a fast, real-time explanation
   but worth strengthening for compliance-grade audit trails.
5. **No adversarial robustness testing.** An attacker aware of the feature set (e.g.
   deliberately mimicking normal session durations) is not modeled.

## 5. Reproducing

```bash
cd data   && python generate_data.py --n_entities 300 --days 45 --anomaly_rate 0.0025
cd ../src && python features.py ../data/sessions.csv ../data/features.csv
          && python dataset.py
          && python train.py
          && python infer.py
cd ../dashboard && streamlit run app.py
```
