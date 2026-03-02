import numpy as np

# -----------------------------
# Utility: R^2 for any model y_pred
# -----------------------------
def _r2_score(y, y_pred):
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot