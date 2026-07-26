"""
BehaviorGuard - Baseline Profiling + Feature Engineering
===========================================================
Deliverable 2: "Baseline profiling model - per-entity 'normal' behaviour
representation (statistical profile)".

For every session we compute features that describe HOW MUCH that session
deviates from the entity's own historical baseline, using only data that
came *before* it in time (causal / no leakage). This is what lets the
downstream sequence model reason about deviation rather than raw values,
and it's also what solves cold-start: an entity with little/no history
automatically falls back to a population-level (peer-group) baseline.
"""
import math

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class BaselineProfiler:
    """Maintains a running (causal) per-entity behavioral baseline.

    For each entity we track:
      - session_duration: running mean/std
      - typical hour-of-day: running mean/std (circular-ish, simplified)
      - resource set seen so far
      - device fingerprints seen so far
      - last known geo location + timestamp (for velocity / impossible travel)
      - failed-auth rate (rolling)

    Population-level statistics act as the cold-start fallback for any
    entity with fewer than `cold_start_min_events` prior events.
    """

    def __init__(self, cold_start_min_events=5):
        self.cold_start_min_events = cold_start_min_events
        self.pop_duration_mean = None
        self.pop_duration_std = None

    def fit_population_prior(self, df):
        self.pop_duration_mean = df["session_duration"].mean()
        self.pop_duration_std = df["session_duration"].std() + 1e-6

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causal pass over sessions (must already be sorted by entity_id, timestamp)."""
        assert self.pop_duration_mean is not None, "call fit_population_prior first"

        records = []
        state = {}  # entity_id -> running stats dict

        for row in df.itertuples(index=False):
            eid = row.entity_id
            st = state.get(eid)
            if st is None:
                st = {
                    "n": 0, "dur_mean": self.pop_duration_mean, "dur_M2": 0.0,
                    "hour_counts": np.zeros(24), "resources": set(), "devices": set(),
                    "last_lat": None, "last_lon": None, "last_ts": None,
                    "n_fail": 0, "n_total": 0,
                }
                state[eid] = st

            is_cold_start = st["n"] < self.cold_start_min_events
            dur_std = math.sqrt(st["dur_M2"] / st["n"]) if st["n"] > 1 else self.pop_duration_std
            dur_std = max(dur_std, 1e-3)

            hour = pd.Timestamp(row.timestamp).hour
            hour_novelty = 1.0 - (st["hour_counts"][hour] / max(st["hour_counts"].sum(), 1)) if st["n"] > 0 else 1.0

            is_new_resource = float(row.resource_accessed not in st["resources"]) if st["n"] > 0 else 1.0
            is_new_device = float(row.device_fingerprint not in st["devices"]) if st["n"] > 0 else 1.0

            if st["last_lat"] is not None:
                dist_km = haversine_km(st["last_lat"], st["last_lon"], row.geo_lat, row.geo_lon)
                dt_hours = max((pd.Timestamp(row.timestamp) - st["last_ts"]).total_seconds() / 3600.0, 1e-3)
                velocity_kmh = dist_km / dt_hours
            else:
                dist_km, velocity_kmh = 0.0, 0.0

            fail_rate_so_far = (st["n_fail"] / st["n_total"]) if st["n_total"] > 0 else 0.0
            duration_z = (row.session_duration - st["dur_mean"]) / dur_std

            records.append({
                "session_id": row.session_id, "entity_id": eid, "timestamp": row.timestamp,
                "entity_type": row.entity_type, "label": row.label,
                "is_cold_start": is_cold_start,
                "prior_event_count": st["n"],
                "duration_z": np.clip(duration_z, -8, 8),
                "hour_novelty": hour_novelty,
                "is_new_resource": is_new_resource,
                "is_new_device": is_new_device,
                "geo_velocity_kmh": min(velocity_kmh, 20000.0),
                "geo_jump_km": dist_km,
                "recent_fail_rate": fail_rate_so_far,
                "auth_failed": float(not row.auth_success),
                "hour_of_day": hour,
            })

            # ---- update running state (after computing features, still causal) ----
            st["n"] += 1
            delta = row.session_duration - st["dur_mean"]
            st["dur_mean"] += delta / st["n"]
            st["dur_M2"] += delta * (row.session_duration - st["dur_mean"])
            st["hour_counts"][hour] += 1
            st["resources"].add(row.resource_accessed)
            st["devices"].add(row.device_fingerprint)
            st["last_lat"], st["last_lon"], st["last_ts"] = row.geo_lat, row.geo_lon, pd.Timestamp(row.timestamp)
            st["n_total"] += 1
            if not row.auth_success:
                st["n_fail"] += 1

        return pd.DataFrame(records)


FEATURE_COLUMNS = [
    "duration_z", "hour_novelty", "is_new_resource", "is_new_device",
    "geo_velocity_kmh", "geo_jump_km", "recent_fail_rate", "auth_failed",
]


def build_feature_table(sessions_csv, out_csv):
    df = pd.read_csv(sessions_csv, parse_dates=["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    profiler = BaselineProfiler(cold_start_min_events=5)
    profiler.fit_population_prior(df)
    feats = profiler.transform(df)

    # log-scale the heavy-tailed velocity/jump features for the model
    feats["geo_velocity_kmh_log"] = np.log1p(feats["geo_velocity_kmh"])
    feats["geo_jump_km_log"] = np.log1p(feats["geo_jump_km"])

    feats.to_csv(out_csv, index=False)
    print(f"Wrote {len(feats)} feature rows -> {out_csv}")
    return feats


if __name__ == "__main__":
    import sys
    sessions_csv = sys.argv[1] if len(sys.argv) > 1 else "sessions.csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "features.csv"
    build_feature_table(sessions_csv, out_csv)
