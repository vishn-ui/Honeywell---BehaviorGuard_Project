"""
BehaviorGuard - Synthetic Behavioral Access-Log Generator
===========================================================
Generates per-entity (user / service_account / edge_device) access-log
sessions with realistic "normal" behavioral baselines, then injects the
attack taxonomy defined in the problem statement at controlled rates.

Ground-truth labels are written to a separate column so the modeling
code can choose to hide them at inference time (as required by the
"cold-start / near-real-time" framing of the problem).

Usage:
    python generate_data.py --n_entities 300 --days 45 --out ../data/sessions.csv
"""
import argparse
import json
import math
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
CITIES = [
    ("New York", 40.7128, -74.0060), ("London", 51.5074, -0.1278),
    ("Mumbai", 19.0760, 72.8777), ("Singapore", 1.3521, 103.8198),
    ("Sydney", -33.8688, 151.2093), ("Tokyo", 35.6762, 139.6503),
    ("Frankfurt", 50.1109, 8.6821), ("Sao Paulo", -23.5505, -46.6333),
    ("Cape Town", -33.9249, 18.4241), ("Toronto", 43.6532, -79.3832),
    ("Dubai", 25.2048, 55.2708), ("Bengaluru", 12.9716, 77.5946),
]

RESOURCE_POOL = [f"resource_{i:03d}" for i in range(1, 61)]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]
OS_LIST = ["Windows11", "Windows10", "macOS14", "Ubuntu22", "iOS17", "Android14", "FirmwareV3"]
ENTITY_TYPES = ["user", "service_account", "edge_device"]
ENTITY_TYPE_WEIGHTS = [0.65, 0.15, 0.20]

LABELS = [
    "normal", "brute_force", "impossible_travel", "credential_stuffing",
    "lateral_movement", "device_spoofing", "low_and_slow_exfil", "insider_drift",
]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def random_mac():
    return ":".join(f"{random.randint(0, 255):02X}" for _ in range(6))


def new_device_fingerprint():
    return f"{random.choice(OS_LIST)}-{random_mac()}"


# ---------------------------------------------------------------------------
# Entity profile construction
# ---------------------------------------------------------------------------
def build_entities(n_entities):
    entities = []
    for i in range(n_entities):
        etype = random.choices(ENTITY_TYPES, weights=ENTITY_TYPE_WEIGHTS)[0]
        home_city = random.choice(CITIES)
        n_typical = random.randint(3, 8)
        typical_resources = set(random.sample(RESOURCE_POOL, n_typical))
        if etype == "user":
            login_start, login_end = sorted([random.randint(6, 11), random.randint(15, 22)])
            sessions_per_day_lambda = random.uniform(1.5, 4.0)
        elif etype == "service_account":
            login_start, login_end = 0, 23  # runs any time
            sessions_per_day_lambda = random.uniform(3.0, 10.0)
        else:  # edge_device
            login_start, login_end = 0, 23
            sessions_per_day_lambda = random.uniform(5.0, 15.0)

        entities.append({
            "entity_id": f"{etype}_{i:04d}",
            "entity_type": etype,
            "home_city": home_city[0],
            "home_lat": home_city[1],
            "home_lon": home_city[2],
            "typical_resources": typical_resources,
            "typical_device": new_device_fingerprint(),
            "typical_auth": random.choice(AUTH_METHODS),
            "login_start": login_start,
            "login_end": login_end,
            "sessions_per_day_lambda": sessions_per_day_lambda,
            "session_duration_mean": random.uniform(180, 1800),
        })
    return entities


# ---------------------------------------------------------------------------
# Normal session generation
# ---------------------------------------------------------------------------
def gen_normal_session(entity, ts):
    lat = entity["home_lat"] + np.random.normal(0, 0.05)
    lon = entity["home_lon"] + np.random.normal(0, 0.05)
    resource = random.choice(list(entity["typical_resources"]))
    duration = max(10, np.random.normal(entity["session_duration_mean"], entity["session_duration_mean"] * 0.3))
    return {
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "timestamp": ts,
        "source_ip": fake.ipv4_public(),
        "geo_city": entity["home_city"],
        "geo_lat": lat,
        "geo_lon": lon,
        "resource_accessed": resource,
        "auth_method": entity["typical_auth"],
        "auth_success": True,
        "session_duration": duration,
        "device_fingerprint": entity["typical_device"],
        "label": "normal",
    }


def gen_timestamps_for_entity(entity, days, start_date):
    timestamps = []
    for d in range(days):
        day = start_date + timedelta(days=d)
        n_sessions = np.random.poisson(entity["sessions_per_day_lambda"])
        for _ in range(n_sessions):
            if entity["entity_type"] == "user":
                hour = int(np.clip(np.random.normal(
                    (entity["login_start"] + entity["login_end"]) / 2, 1.5), 0, 23))
            else:
                hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = day.replace(hour=hour, minute=minute, second=second)
            timestamps.append(ts)
    timestamps.sort()
    return timestamps


# ---------------------------------------------------------------------------
# Attack pattern injectors -- each returns a list of session dicts
# ---------------------------------------------------------------------------
def inject_brute_force(entity, ts):
    src_ip = fake.ipv4_public()
    n_attempts = random.randint(6, 25)
    rows = []
    base = ts
    for k in range(n_attempts):
        t = base + timedelta(seconds=random.randint(1, 15) * k)
        rows.append({
            "entity_id": entity["entity_id"], "entity_type": entity["entity_type"],
            "timestamp": t, "source_ip": src_ip,
            "geo_city": entity["home_city"], "geo_lat": entity["home_lat"], "geo_lon": entity["home_lon"],
            "resource_accessed": entity["typical_auth"] + "_login",
            "auth_method": entity["typical_auth"], "auth_success": (k == n_attempts - 1),
            "session_duration": random.uniform(2, 8),
            "device_fingerprint": entity["typical_device"],
            "label": "brute_force",
        })
    return rows


def inject_impossible_travel(entity, ts):
    far_city = random.choice([c for c in CITIES if c[0] != entity["home_city"]])
    dist = haversine_km(entity["home_lat"], entity["home_lon"], far_city[1], far_city[2])
    gap_minutes = max(5, dist / 900 * 60 * random.uniform(0.05, 0.3))  # implausible speed
    t2 = ts + timedelta(minutes=gap_minutes)
    s1 = gen_normal_session(entity, ts)
    s1["label"] = "impossible_travel"
    s2 = gen_normal_session(entity, t2)
    s2["geo_city"], s2["geo_lat"], s2["geo_lon"] = far_city
    s2["source_ip"] = fake.ipv4_public()
    s2["label"] = "impossible_travel"
    return [s1, s2]


def inject_credential_stuffing(entities_subset, ts):
    src_ips = [fake.ipv4_public() for _ in range(max(1, len(entities_subset) // 8))]
    rows = []
    for ent in entities_subset:
        t = ts + timedelta(seconds=random.randint(0, 120))
        rows.append({
            "entity_id": ent["entity_id"], "entity_type": ent["entity_type"],
            "timestamp": t, "source_ip": random.choice(src_ips),
            "geo_city": "unknown", "geo_lat": 0.0, "geo_lon": 0.0,
            "resource_accessed": ent["typical_auth"] + "_login",
            "auth_method": ent["typical_auth"],
            "auth_success": random.random() < 0.08,
            "session_duration": random.uniform(2, 6),
            "device_fingerprint": new_device_fingerprint(),
            "label": "credential_stuffing",
        })
    return rows


def inject_lateral_movement(entity, ts):
    n_hops = random.randint(5, 12)
    unseen = [r for r in RESOURCE_POOL if r not in entity["typical_resources"]]
    hop_resources = random.sample(unseen, min(n_hops, len(unseen)))
    rows = []
    for k, res in enumerate(hop_resources):
        t = ts + timedelta(minutes=random.randint(1, 4) * k)
        s = gen_normal_session(entity, t)
        s["resource_accessed"] = res
        s["label"] = "lateral_movement"
        rows.append(s)
    return rows


def inject_device_spoofing(entity, ts):
    s = gen_normal_session(entity, ts)
    spoofed = entity["typical_device"].split("-")[0] + "-MISMATCH-" + random_mac()
    # deliberately different OS family too
    other_os = random.choice([o for o in OS_LIST if o != entity["typical_device"].split("-")[0]])
    s["device_fingerprint"] = f"{other_os}-{random_mac()}"
    s["label"] = "device_spoofing"
    return [s]


def inject_low_and_slow(entity, ts, days_span=10):
    unseen = [r for r in RESOURCE_POOL if r not in entity["typical_resources"]]
    n_events = random.randint(6, 14)
    rows = []
    for k in range(n_events):
        t = ts + timedelta(days=random.uniform(0, days_span), hours=random.uniform(0, 5))  # off-hours
        s = gen_normal_session(entity, t)
        s["resource_accessed"] = random.choice(unseen)
        s["session_duration"] = random.uniform(600, 2400)
        s["label"] = "low_and_slow_exfil"
        rows.append(s)
    return rows


def inject_insider_drift(entity, ts, days_span=20):
    """Edge case: legitimate-looking but slowly-expanding footprint. Label kept
    separate for FP-tuning/evaluation only -- NOT used as a hard attack class."""
    unseen = [r for r in RESOURCE_POOL if r not in entity["typical_resources"]]
    n_events = random.randint(4, 8)
    rows = []
    for k in range(n_events):
        t = ts + timedelta(days=(k / max(n_events - 1, 1)) * days_span)
        s = gen_normal_session(entity, t)
        s["resource_accessed"] = random.choice(unseen) if unseen else s["resource_accessed"]
        s["label"] = "insider_drift"
        rows.append(s)
    return rows


# ---------------------------------------------------------------------------
# Main generation driver
# ---------------------------------------------------------------------------
def generate(n_entities=300, days=45, anomaly_rate=0.02, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    entities = build_entities(n_entities)
    ent_by_id = {e["entity_id"]: e for e in entities}
    start_date = datetime(2026, 5, 1)

    all_rows = []
    for entity in entities:
        for ts in gen_timestamps_for_entity(entity, days, start_date):
            all_rows.append(gen_normal_session(entity, ts))

    n_normal = len(all_rows)
    print(f"Generated {n_normal} normal sessions across {n_entities} entities")

    # ---- injections ----
    injectors_single = [inject_impossible_travel, inject_brute_force,
                         inject_lateral_movement, inject_device_spoofing]
    n_inject_events = int(n_normal * anomaly_rate)

    for _ in range(n_inject_events):
        entity = random.choice(entities)
        ts = start_date + timedelta(days=random.uniform(2, days - 2), hours=random.uniform(0, 24))
        fn = random.choice(injectors_single)
        all_rows.extend(fn(entity, ts))

    # credential stuffing (multi-entity bursts)
    n_stuffing_events = max(1, int(n_inject_events * 0.08))
    for _ in range(n_stuffing_events):
        subset = random.sample(entities, k=min(15, len(entities)))
        ts = start_date + timedelta(days=random.uniform(2, days - 2), hours=random.uniform(0, 24))
        all_rows.extend(inject_credential_stuffing(subset, ts))

    # low-and-slow exfiltration campaigns
    n_exfil_campaigns = max(1, int(n_inject_events * 0.05))
    for _ in range(n_exfil_campaigns):
        entity = random.choice(entities)
        ts = start_date + timedelta(days=random.uniform(2, days - 12))
        all_rows.extend(inject_low_and_slow(entity, ts))

    # insider drift (edge case)
    n_drift_campaigns = max(1, int(n_inject_events * 0.05))
    for _ in range(n_drift_campaigns):
        entity = random.choice(entities)
        ts = start_date + timedelta(days=random.uniform(2, days - 22))
        all_rows.extend(inject_insider_drift(entity, ts))

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df["session_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]

    label_counts = df["label"].value_counts()
    print("\nLabel distribution:")
    print(label_counts)
    print(f"\nAnomaly rate (non-normal): {(1 - label_counts['normal'] / len(df)) * 100:.2f}%")

    entities_meta = []
    for e in entities:
        entities_meta.append({**{k: v for k, v in e.items() if k != "typical_resources"},
                               "typical_resources": json.dumps(sorted(e["typical_resources"]))})
    entities_df = pd.DataFrame(entities_meta)

    return df, entities_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_entities", type=int, default=300)
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--anomaly_rate", type=float, default=0.02)
    ap.add_argument("--out_sessions", type=str, default="sessions.csv")
    ap.add_argument("--out_entities", type=str, default="entities.csv")
    args = ap.parse_args()

    sessions_df, entities_df = generate(args.n_entities, args.days, args.anomaly_rate)
    sessions_df.to_csv(args.out_sessions, index=False)
    entities_df.to_csv(args.out_entities, index=False)
    print(f"\nSaved {len(sessions_df)} sessions -> {args.out_sessions}")
    print(f"Saved {len(entities_df)} entity profiles -> {args.out_entities}")
