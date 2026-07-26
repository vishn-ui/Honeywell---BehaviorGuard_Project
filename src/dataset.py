"""
BehaviorGuard - Sequence Construction
=========================================
Deliverable: "Sequential and behavioural data - access events over time,
not static snapshots."

Turns the per-event feature table into fixed-length sliding-window
sequences per entity (front-padded for entities with short history --
this is how cold-start entities are represented instead of being dropped).

Split strategy: CHRONOLOGICAL by the window's final timestamp (train on
the past, evaluate on the future) -- this mirrors real deployment and
also exercises concept drift, rather than a random/leaky split.

insider_drift is deliberately excluded from the *hard* attack-class
label used for training (it's an edge case for false-positive tuning,
not a certain attack; see problem statement) but its true label is kept
in `raw_label` for evaluation.
"""
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS

SEQ_LEN = 8

TRAIN_CLASSES = [
    "normal", "brute_force", "impossible_travel", "credential_stuffing",
    "lateral_movement", "device_spoofing", "low_and_slow_exfil",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRAIN_CLASSES)}


def training_label(raw_label):
    if raw_label == "insider_drift":
        return "normal"  # edge case: not a hard-labeled attack during training
    return raw_label


SEQ_FEATURE_COLUMNS = [
    "duration_z", "hour_novelty", "is_new_resource", "is_new_device",
    "geo_velocity_kmh_log", "geo_jump_km_log", "recent_fail_rate", "auth_failed",
]


def build_sequences(feats: pd.DataFrame, seq_len=SEQ_LEN):
    feats = feats.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    F = len(SEQ_FEATURE_COLUMNS)

    X, y, meta = [], [], []
    for eid, group in feats.groupby("entity_id", sort=False):
        arr = group[SEQ_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        n = len(group)
        raw_labels = group["label"].tolist()
        timestamps = group["timestamp"].tolist()
        session_ids = group["session_id"].tolist()
        prior_counts = group["prior_event_count"].tolist()

        for i in range(n):
            lo = max(0, i - seq_len + 1)
            window = arr[lo:i + 1]
            pad_len = seq_len - window.shape[0]
            if pad_len > 0:
                window = np.vstack([np.zeros((pad_len, F), dtype=np.float32), window])
            X.append(window)
            tr_label = training_label(raw_labels[i])
            y.append(CLASS_TO_IDX[tr_label])
            meta.append({
                "entity_id": eid, "session_id": session_ids[i], "timestamp": timestamps[i],
                "raw_label": raw_labels[i], "train_label": tr_label,
                "prior_event_count": prior_counts[i],
            })

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    meta = pd.DataFrame(meta)
    return X, y, meta


def chronological_split(meta, train_frac=0.7, val_frac=0.15):
    ts_sorted = meta["timestamp"].sort_values()
    n = len(ts_sorted)
    train_cut = ts_sorted.iloc[int(n * train_frac)]
    val_cut = ts_sorted.iloc[int(n * (train_frac + val_frac))]
    train_idx = meta.index[meta["timestamp"] <= train_cut].to_numpy()
    val_idx = meta.index[(meta["timestamp"] > train_cut) & (meta["timestamp"] <= val_cut)].to_numpy()
    test_idx = meta.index[meta["timestamp"] > val_cut].to_numpy()
    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    feats = pd.read_csv("../data/features.csv", parse_dates=["timestamp"])
    X, y, meta = build_sequences(feats)
    print("X shape:", X.shape, "y shape:", y.shape)
    print(meta["train_label"].value_counts())
    train_idx, val_idx, test_idx = chronological_split(meta)
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    np.savez_compressed("../data/sequences.npz", X=X, y=y,
                         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    meta.to_csv("../data/sequence_meta.csv", index=False)
    print("Saved ../data/sequences.npz and ../data/sequence_meta.csv")
