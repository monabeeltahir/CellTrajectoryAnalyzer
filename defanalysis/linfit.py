from scipy.stats import linregress
import numpy as np

# -----------------------------
# Linear fit wrapped as a "plugin" too (recommended)
# -----------------------------
def _fit_linear(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slope, intercept, r_val, _, _ = linregress(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r_val**2),
    }