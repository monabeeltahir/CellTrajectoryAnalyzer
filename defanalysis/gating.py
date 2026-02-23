import numpy as np



def percentile_clip(values, low=0.1, high=99.9):
    """
    Clip array to percentile range [low, high].
    Returns filtered array.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return values

    lo = np.percentile(values, low)
    hi = np.percentile(values, high)

    return values[(values >= lo) & (values <= hi)]


def compute_gate(ctrl_slopes, exp_slopes, sensitivity=99.9, correct_baseline_drift=False):
    """
    Noise is always measured from control.
    Gate anchor is either ctrl median (static) or exp median (dynamic drift-corrected).
    """
    ctrl_median = float(np.median(ctrl_slopes))
    ctrl_p = float(np.percentile(ctrl_slopes, sensitivity))
    noise_width = ctrl_p - ctrl_median

    exp_median = float(np.median(exp_slopes))
    baseline_anchor = exp_median if correct_baseline_drift else ctrl_median

    gate_threshold = float(baseline_anchor + noise_width)

    metrics = {
        "ctrl_median": ctrl_median,
        "ctrl_p": ctrl_p,                 # p(sensitivity)
        "noise_width": float(noise_width),
        "exp_median": exp_median,
        "baseline_anchor": float(baseline_anchor),
        "gate_threshold": gate_threshold,
        "sensitivity": float(sensitivity),
        "mode_name": "Dynamic (Drift Corrected)" if correct_baseline_drift else "Static (Absolute Control)",
    }
    return gate_threshold, metrics

def classify_tracks(exp_track_stats, gate_threshold, feature_key="slope_inverted"):
    """
    Adds is_deflected and gate_used for each track using chosen feature.
    Returns rows suitable for CSV (no x/y arrays)
    """
    rows = []
    for t in exp_track_stats:
        val = t.get(feature_key, None)
        if val is None or not np.isfinite(val):
            # if missing, mark not deflected (or you can mark as None)
            is_deflected = False
            val_out = np.nan
        else:
            is_deflected = bool(val > gate_threshold)
            val_out = float(val)

        rows.append({
            "id": t["id"],
            feature_key: val_out,
            "slope_inverted": float(t.get("slope_inverted", np.nan)),
            "slope_raw": float(t.get("slope_raw", np.nan)),
            "is_deflected": is_deflected,
            "gate_used": float(gate_threshold),
            "gate_feature": feature_key,
        })
    return rows

