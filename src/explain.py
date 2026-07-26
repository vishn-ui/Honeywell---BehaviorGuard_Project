"""
BehaviorGuard - Explainability Layer
=========================================
Deliverable 5: "feature attribution per alert (e.g. 'flagged due to
geo-velocity + new device fingerprint')."

Approach: for each flagged session we already have its baseline-deviation
feature vector (from features.py) -- these features are *already*
z-scores / novelty flags relative to the entity's own history, so the
single largest-magnitude features ARE the explanation, no separate
attribution model needed. This keeps the explainability layer fast
enough for near-real-time scoring and fully transparent to a SOC analyst.

(For a heavier-weight alternative, `gradient_attribution()` below shows
how to get integrated-gradients-style attribution straight from the
trained LSTM if a numeric SHAP-style score is preferred.)
"""
import numpy as np
import torch

FEATURE_LABELS = {
    "duration_z": "unusual session duration",
    "hour_novelty": "atypical time-of-day",
    "is_new_resource": "access to a never-before-seen resource",
    "is_new_device": "new/unrecognized device fingerprint",
    "geo_velocity_kmh_log": "implausible geo-velocity (impossible travel)",
    "geo_jump_km_log": "large geographic jump from last known location",
    "recent_fail_rate": "elevated recent authentication failure rate",
    "auth_failed": "failed authentication attempt",
}
FEATURE_ORDER = list(FEATURE_LABELS.keys())

# rough "typical" scale per feature, used only to rank contribution magnitude
FEATURE_SCALE = {
    "duration_z": 1.0, "hour_novelty": 1.0, "is_new_resource": 1.0,
    "is_new_device": 1.0, "geo_velocity_kmh_log": 2.0, "geo_jump_km_log": 2.0,
    "recent_fail_rate": 1.0, "auth_failed": 1.0,
}


def explain_event(feature_vector: dict, top_k=2):
    """feature_vector: dict of {feature_name: raw_value} for ONE event
    (the last timestep of its sequence window)."""
    contributions = []
    for feat in FEATURE_ORDER:
        val = feature_vector.get(feat, 0.0)
        magnitude = abs(val) / FEATURE_SCALE[feat]
        contributions.append((feat, magnitude, val))
    contributions.sort(key=lambda t: -t[1])
    top = contributions[:top_k]
    phrases = [FEATURE_LABELS[f] for f, mag, val in top if mag > 0.15]
    if not phrases:
        return "no single feature stands out; flagged on combined weak signals"
    return "flagged due to " + " + ".join(phrases)


def gradient_attribution(model, x_tensor, device):
    """Optional: integrated-gradients-style saliency straight from the LSTM,
    for when a numeric per-feature attribution score (not just ranking) is
    needed. x_tensor: (1, T, F) tensor for a single sequence."""
    model.eval()
    x_tensor = x_tensor.clone().to(device).requires_grad_(True)
    baseline = torch.zeros_like(x_tensor)
    steps = 20
    grads = torch.zeros_like(x_tensor)
    for alpha in np.linspace(0, 1, steps):
        interp = baseline + alpha * (x_tensor - baseline)
        interp.requires_grad_(True)
        class_logits, risk_logit = model(interp)
        risk_logit.sum().backward()
        grads += interp.grad
    avg_grads = grads / steps
    attributions = (x_tensor - baseline) * avg_grads
    # sum over time, keep per-feature
    return attributions.sum(dim=1).detach().cpu().numpy()[0]  # (F,)
