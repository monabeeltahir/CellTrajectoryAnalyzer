import numpy as np
from scipy.signal import savgol_filter

def _curvature_smoothed_derivative_raw(x, y, window_length=9, polyorder=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 5:
        return x, y, y.copy(), np.array([], dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # make valid odd window
    w = min(window_length, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if w < 5:
        y_s = y.copy()
    else:
        p = min(polyorder, w - 1)
        y_s = savgol_filter(y, window_length=w, polyorder=p)

    dx = np.gradient(x)
    dy = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2)**1.5 + 1e-12
    kappa = np.abs(dx * ddy - dy * ddx) / denom

    return x, y, y_s, kappa


def _fit_curvature_smoothed(x, y, window_length=9, polyorder=3):
    x_s, y_raw, y_smooth, kappa = _curvature_smoothed_derivative_raw(
        x, y,
        window_length=window_length,
        polyorder=polyorder
    )

    kappa = np.asarray(kappa, dtype=float)
    kappa = kappa[np.isfinite(kappa)]

    if kappa.size == 0:
        raise ValueError("no_finite_curvature")

    return {
        "mean": float(np.mean(kappa)),
        "median": float(np.median(kappa)),
        "max": float(np.max(kappa)),
        "std": float(np.std(kappa)),
        "p90": float(np.percentile(kappa, 90)),
    }