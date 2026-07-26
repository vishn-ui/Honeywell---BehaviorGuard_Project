"""
BehaviorGuard - Training
============================
Trains the LSTM detector+classifier on the chronological train split,
selects the best checkpoint on val loss, and reports the evaluation
metrics called out in the assignment's Evaluation Criteria:
  - Detection accuracy on imbalanced labels (macro-F1, per-class P/R)
  - Correct anomaly-type classification (confusion matrix)
  - False positive rate at a realistic analyst alert budget (top 1%)
"""
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from dataset import TRAIN_CLASSES
from model import BehaviorLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(npz_path="../data/sequences.npz"):
    d = np.load(npz_path)
    return d["X"], d["y"], d["train_idx"], d["val_idx"], d["test_idx"]


def make_loader(X, y, idx, batch_size=256, shuffle=False):
    xt = torch.tensor(X[idx], dtype=torch.float32)
    yt = torch.tensor(y[idx], dtype=torch.long)
    is_anom = torch.tensor((y[idx] != 0).astype(np.float32))
    ds = TensorDataset(xt, yt, is_anom)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(y_train, n_classes):
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1
    weights = counts.sum() / (n_classes * counts)
    # normal class weight capped so it doesn't get suppressed to near-zero
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, optimizer, class_weights, train=True):
    model.train() if train else model.eval()
    ce = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    bce = nn.BCEWithLogitsLoss()
    total_loss, n = 0.0, 0
    all_logits, all_y, all_risk = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb, ab in loader:
            xb, yb, ab = xb.to(DEVICE), yb.to(DEVICE), ab.to(DEVICE)
            class_logits, risk_logit = model(xb)
            loss = ce(class_logits, yb) + 0.5 * bce(risk_logit, ab)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            all_logits.append(class_logits.detach().cpu())
            all_y.append(yb.detach().cpu())
            all_risk.append(risk_logit.detach().cpu())
    return total_loss / n, torch.cat(all_logits), torch.cat(all_y), torch.cat(all_risk)


def evaluate(model, loader, class_weights, split_name, alert_budget=0.01):
    loss, logits, y_true, risk_logits = run_epoch(model, loader, None, class_weights, train=False)
    y_pred = logits.argmax(dim=1).numpy()
    y_true_np = y_true.numpy()
    risk_score = torch.sigmoid(risk_logits).numpy()

    print(f"\n=== {split_name} ===  loss={loss:.4f}")
    report = classification_report(
        y_true_np, y_pred, labels=list(range(len(TRAIN_CLASSES))),
        target_names=TRAIN_CLASSES, zero_division=0, digits=3,
    )
    print(report)

    cm = confusion_matrix(y_true_np, y_pred, labels=list(range(len(TRAIN_CLASSES))))

    # False positive rate at a realistic analyst alert budget (top 1% of events)
    n = len(risk_score)
    budget_n = max(1, int(n * alert_budget))
    top_idx = np.argsort(-risk_score)[:budget_n]
    is_true_anomaly = (y_true_np != 0)
    fp_at_budget = 1 - is_true_anomaly[top_idx].mean()
    recall_at_budget = is_true_anomaly[top_idx].sum() / max(is_true_anomaly.sum(), 1)

    binary_ap = average_precision_score(is_true_anomaly.astype(int), risk_score)

    print(f"Alert budget = top {alert_budget * 100:.1f}% ({budget_n} events)")
    print(f"  False positive rate @ budget : {fp_at_budget:.3f}")
    print(f"  Anomaly recall captured @ budget : {recall_at_budget:.3f}")
    print(f"  Binary anomaly-vs-normal Average Precision (PR-AUC): {binary_ap:.3f}")

    return {
        "loss": loss, "report": report, "confusion_matrix": cm.tolist(),
        "fp_rate_at_budget": float(fp_at_budget),
        "recall_at_budget": float(recall_at_budget),
        "pr_auc": float(binary_ap),
    }


def main():
    X, y, train_idx, val_idx, test_idx = load_split()
    n_features = X.shape[2]
    n_classes = len(TRAIN_CLASSES)

    train_loader = make_loader(X, y, train_idx, shuffle=True)
    val_loader = make_loader(X, y, val_idx, shuffle=False)
    test_loader = make_loader(X, y, test_idx, shuffle=False)

    class_weights = compute_class_weights(y[train_idx], n_classes)
    print("Class weights:", dict(zip(TRAIN_CLASSES, class_weights.tolist())))

    model = BehaviorLSTM(n_features=n_features, n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

    best_val_loss = float("inf")
    n_epochs = 12
    for epoch in range(1, n_epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, optimizer, class_weights, train=True)
        val_loss, val_logits, val_y, val_risk = run_epoch(model, val_loader, None, class_weights, train=False)
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "../models/behavior_lstm.pt")

    model.load_state_dict(torch.load("../models/behavior_lstm.pt"))

    val_metrics = evaluate(model, val_loader, class_weights, "VALIDATION")
    test_metrics = evaluate(model, test_loader, class_weights, "TEST (held-out, future in time)")

    with open("../models/metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics, "classes": TRAIN_CLASSES}, f, indent=2)
    print("\nSaved model -> ../models/behavior_lstm.pt")
    print("Saved metrics -> ../models/metrics.json")


if __name__ == "__main__":
    main()
