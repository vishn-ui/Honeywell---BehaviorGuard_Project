"""
BehaviorGuard - Inference / Alert Generation
=================================================
Runs the trained model over the (chronological) test split -- standing
in for "new events arriving in near real time" -- and writes a ranked
alert queue with explanations for the dashboard.

Ground-truth labels are carried through ONLY as `true_label` for
evaluation/demo purposes; a real deployment would not have this column
at scoring time.
"""
import numpy as np
import pandas as pd
import torch

from dataset import SEQ_FEATURE_COLUMNS, TRAIN_CLASSES
from explain import explain_event
from model import BehaviorLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    d = np.load("../data/sequences.npz")
    X, y = d["X"], d["y"]
    test_idx = d["test_idx"]
    meta = pd.read_csv("../data/sequence_meta.csv", parse_dates=["timestamp"])

    model = BehaviorLSTM(n_features=X.shape[2], n_classes=len(TRAIN_CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load("../models/behavior_lstm.pt", map_location=DEVICE))
    model.eval()

    xb = torch.tensor(X[test_idx], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        class_logits, risk_logit = model(xb)
        probs = torch.softmax(class_logits, dim=1).cpu().numpy()
        risk_score = torch.sigmoid(risk_logit).cpu().numpy()
        pred_class_idx = probs.argmax(axis=1)

    rows = []
    for j, i in enumerate(test_idx):
        last_step = X[i, -1, :]  # most recent event's feature vector
        fvec = dict(zip(SEQ_FEATURE_COLUMNS, last_step.tolist()))
        explanation = explain_event(fvec)
        m = meta.iloc[i]
        rows.append({
            "session_id": m["session_id"],
            "entity_id": m["entity_id"],
            "timestamp": m["timestamp"],
            "risk_score": float(risk_score[j]),
            "predicted_type": TRAIN_CLASSES[pred_class_idx[j]],
            "predicted_confidence": float(probs[j, pred_class_idx[j]]),
            "explanation": explanation,
            "true_label": m["raw_label"],  # evaluation-only column
        })

    alerts = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    alerts.insert(0, "rank", np.arange(1, len(alerts) + 1))
    alerts.to_csv("../data/alerts.csv", index=False)
    print(f"Wrote {len(alerts)} scored events -> ../data/alerts.csv")
    print(alerts.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
