import numpy as np
from .utility import _r2_score
# -----------------------------
# Quadratic fit (NEW): y = ax^2 + bx + c
# -----------------------------
def _fit_quadratic(x, y, deg=2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # np.polyfit returns [a, b, c] for deg=2
    a2, b1, c0 = map(float, np.polyfit(x, y, deg=deg))
    y_pred = (a2 * x**2) + (b1 * x) + c0
    r2 = _r2_score(y, y_pred)

    return {
        "a2": a2,
        "b1": b1,
        "c0": c0,
        "r2": float(r2),
    }
