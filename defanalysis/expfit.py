
import numpy as np
from scipy.optimize import curve_fit
from utility import _r2_score


def _exp_saturating(x, y0, A, a, x0):
    return y0 + A * (1.0 - np.exp(-a * (x - x0)))

def _fit_exp_saturating(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x0 = float(x.min())

    # initial guesses
    y0_guess = float(np.median(y[:max(3, len(y)//10)]))
    y_end = float(np.median(y[-max(3, len(y)//10):]))
    A_guess = y_end - y0_guess
    if abs(A_guess) < 1e-6:
        A_guess = 1.0

    y_target = y0_guess + 0.632 * A_guess
    idx = int(np.argmin(np.abs(y - y_target)))
    dx = float(x[idx] - x0)
    a_guess = 1.0 / max(dx, 1.0)

    def f(x_in, y0, A, a):
        return _exp_saturating(x_in, y0, A, a, x0)

    bounds = ([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(
        f, x, y,
        p0=[y0_guess, A_guess, a_guess],
        bounds=bounds,
        maxfev=20000
    )
    y0_fit, A_fit, a_fit = map(float, popt)

    y_inf = y0_fit + A_fit
    tau_px = (1.0 / a_fit) if a_fit > 0 else float("inf")

    y_pred = f(x, y0_fit, A_fit, a_fit)
    r2 = _r2_score(y, y_pred)

    return {
        "y0": y0_fit,
        "A": A_fit,
        "a": a_fit,
        "x0": x0,
        "y_inf": y_inf,
        "tau_px": tau_px,
        "r2": float(r2),
    }