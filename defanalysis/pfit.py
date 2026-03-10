import numpy as np
from .utility import _r2_score, _rmse

# -----------------------------
# Quadratic fit (NEW): y = ax^2 + bx + c
# -----------------------------
def _fit_quadratic(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError("too_few_points")

    a2, b1, c0 = np.polyfit(x, y, 2)
    y_pred = a2 * x**2 + b1 * x + c0

    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    r2 = 1 - ss_res / ss_tot

    return {
        "a2": float(a2),
        "b1": float(b1),
        "c0": float(c0),
        "r2": float(r2),
        "rmse": _rmse(y, y_pred),
    }
