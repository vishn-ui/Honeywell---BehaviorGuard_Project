"""
BehaviorGuard - Analyst Dashboard
=====================================
Deliverable 6: "Analyst-facing dashboard - ranked alert queue, risk
score, contributing factors, entity history view."

Run with:
    streamlit run app.py
(from inside the dashboard/ folder, after data + model have been built)
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="BehaviorGuard - Analyst Console", layout="wide")


@st.cache_data
def load_data():
    alerts = pd.read_csv(DATA_DIR / "alerts.csv", parse_dates=["timestamp"])
    sessions = pd.read_csv(DATA_DIR / "sessions.csv", parse_dates=["timestamp"])
    metrics = {}
    mpath = MODEL_DIR / "metrics.json"
    if mpath.exists():
        with open(mpath) as f:
            metrics = json.load(f)
    return alerts, sessions, metrics


alerts, sessions, metrics = load_data()

st.title("BehaviorGuard")
st.caption("AI-Powered Behavioral Anomaly Detection - Analyst Console")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Alert queue controls")
budget_pct = st.sidebar.slider("Analyst alert budget (top % of events by risk)", 0.1, 10.0, 1.0, 0.1)
type_filter = st.sidebar.multiselect(
    "Filter by predicted type", sorted(alerts["predicted_type"].unique()),
)
entity_search = st.sidebar.text_input("Search entity_id contains")

n_budget = max(1, int(len(alerts) * budget_pct / 100))
queue = alerts.sort_values("risk_score", ascending=False).head(n_budget)
if type_filter:
    queue = queue[queue["predicted_type"].isin(type_filter)]
if entity_search:
    queue = queue[queue["entity_id"].str.contains(entity_search, case=False, na=False)]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events scored (test window)", f"{len(alerts):,}")
c2.metric("Alert budget size", f"{n_budget:,}  ({budget_pct:.1f}%)")
true_positive_rate = (queue["true_label"] != "normal").mean() if len(queue) else 0.0
c3.metric("Precision within budget", f"{true_positive_rate * 100:.1f}%",
          help="Share of alerts in the current budget that are true anomalies "
               "(demo-only, uses ground truth from the synthetic generator).")
test_metrics = metrics.get("test", {})
c4.metric("Model PR-AUC (test)", f"{test_metrics.get('pr_auc', 0):.3f}")

st.divider()

# ---------------------------------------------------------------------------
# Ranked alert queue + type breakdown
# ---------------------------------------------------------------------------
left, right = st.columns([2.2, 1])

with left:
    st.subheader("Ranked alert queue")
    display_cols = ["rank", "entity_id", "timestamp", "risk_score",
                     "predicted_type", "predicted_confidence", "explanation"]
    st.dataframe(
        queue[display_cols].style.format({"risk_score": "{:.3f}", "predicted_confidence": "{:.3f}"}),
        use_container_width=True, height=430, hide_index=True,
    )

with right:
    st.subheader("Alerts by type")
    type_counts = queue["predicted_type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]
    fig = px.bar(type_counts, x="count", y="type", orientation="h", color="type")
    fig.update_layout(showlegend=False, height=430, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Entity drill-down
# ---------------------------------------------------------------------------
st.subheader("Entity drill-down")
options = queue["entity_id"].tolist() if len(queue) else alerts["entity_id"].tolist()
if options:
    selected_entity = st.selectbox("Select an entity from the alert queue", options)

    ent_alerts = alerts[alerts["entity_id"] == selected_entity].sort_values("timestamp")
    ent_history = sessions[sessions["entity_id"] == selected_entity].sort_values("timestamp")

    colA, colB = st.columns([1, 1.4])
    with colA:
        st.markdown(f"**Entity:** `{selected_entity}`")
        top_alert = ent_alerts.sort_values("risk_score", ascending=False).iloc[0]
        st.markdown(f"**Highest risk score:** {top_alert['risk_score']:.3f}")
        st.markdown(f"**Predicted type:** {top_alert['predicted_type']}")
        st.info(f"Why flagged: {top_alert['explanation']}")
        st.markdown(f"**Total historical sessions:** {len(ent_history)}")
        st.markdown("**Distinct resources accessed:** "
                     f"{ent_history['resource_accessed'].nunique()}")
        st.markdown("**Distinct devices seen:** "
                     f"{ent_history['device_fingerprint'].nunique()}")

    with colB:
        st.markdown("**Session-duration history (anomalous points highlighted)**")
        hist_plot = ent_history.copy()
        hist_plot["is_anomaly"] = hist_plot["label"] != "normal"
        fig2 = px.scatter(
            hist_plot, x="timestamp", y="session_duration", color="is_anomaly",
            color_discrete_map={True: "#C0392B", False: "#2C5F8A"},
            hover_data=["resource_accessed", "geo_city", "label"],
        )
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Recent session history**")
    st.dataframe(
        ent_history[["timestamp", "resource_accessed", "geo_city", "auth_method",
                      "auth_success", "device_fingerprint", "label"]].tail(25),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No alerts in the current filter/budget selection.")

st.divider()
with st.expander("Model evaluation details (from training run)"):
    if test_metrics:
        st.text(test_metrics.get("report", "No report available."))
        st.markdown(
            f"- False positive rate @ 1% alert budget: **{test_metrics.get('fp_rate_at_budget', 0):.3f}**\n"
            f"- Anomaly recall captured @ 1% alert budget: **{test_metrics.get('recall_at_budget', 0):.3f}**\n"
            f"- Binary anomaly-vs-normal PR-AUC: **{test_metrics.get('pr_auc', 0):.3f}**"
        )
    else:
        st.write("Run `train.py` to populate evaluation metrics.")
