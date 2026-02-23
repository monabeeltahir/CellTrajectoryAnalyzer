import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pandas as pd
import os



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
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    return {
        "y0": y0_fit,
        "A": A_fit,
        "a": a_fit,
        "x0": x0,
        "y_inf": y_inf,
        "tau_px": tau_px,
        "r2_exp": float(r2),
    }

def compute_track_stats(df, min_track_length=10, fit_exponential=False, min_x_span_px=30):
    """
    df must have columns: id, frame, center_x, center_y
    Returns list of dicts with:
      id, slope_raw, slope_inverted, intercept, x_vals, y_vals
    """
    track_stats = []

    for obj_id, track in df.groupby("id"):
        if len(track) < min_track_length:
            continue

        track = track.sort_values("frame")
        x = track["center_x"].to_numpy()
        y = track["center_y"].to_numpy()

        slope, intercept, r_val, _, _ = linregress(x, y)
        slope_inverted = -1.0 * slope
       
        exp_ok = False
        exp_reason = "not_run"

        y0_fit = A_fit = x0_fit = None
        y_inf = a = tau_px = r2_exp = None

        if fit_exponential:
            try:
                ef = _fit_exp_saturating(x, y)  # should return dict with: y0, A, x0, y_inf, a, tau_px, r2_exp
                # basic sanity checks
                if not np.isfinite(ef["y0"]) or not np.isfinite(ef["A"]) or not np.isfinite(ef["a"]) or not np.isfinite(ef["x0"]):
                    raise ValueError("non_finite_parameters")

                exp_ok = True
                exp_reason = "ok"

                y0_fit = float(ef["y0"])
                A_fit  = float(ef["A"])
                x0_fit = float(ef["x0"])

                y_inf  = float(ef["y_inf"])
                a      = float(ef["a"])
                tau_px = float(ef["tau_px"])
                r2_exp = float(ef["r2_exp"])

            except Exception as e:
                exp_ok = False
                exp_reason = f"fit_failed:{type(e).__name__}"
        
        
        # if fit_exponential and (x.max() - x.min()) >= min_x_span_px:
        #     try:
        #         ef = _fit_exp_saturating(x, y)
        #         exp_ok = True
        #         y0_fit = ef["y0"]
        #         A_fit  = ef["A"]
                
        #         x0_fit = ef["x0"]
        #         y_inf = ef["y_inf"]
        #         a = ef["a"]
        #         tau_px = ef["tau_px"]
        #         r2_exp = ef["r2_exp"]
        #     except Exception:
        #         exp_ok = False

        
        track_stats.append({
            "id": obj_id,
            "x_vals": x,
            "y_vals": y,

            # linear fields you already have...
            "slope_raw": float(slope),
            "slope_inverted": float(slope_inverted),
            "intercept": float(intercept),
            "r2_lin": float(r_val**2),

            # exponential fields
            "exp_ok": exp_ok,
            "exp_reason": exp_reason,
            "y0_exp": y0_fit,
            "A_exp": A_fit,
            "x0_exp": x0_fit,
            "y_inf": y_inf,
            "a": a,
            "tau_px": tau_px,
            "r2_exp": r2_exp,
            })

    return track_stats

def extract_slopes(track_stats):
    return np.array([d["slope_inverted"] for d in track_stats], dtype=float)




def save_track_parameters(track_stats, out_path):
    """
    Save scalar track parameters (no x/y arrays) to CSV.
    """
    rows = []

    for t in track_stats:
        rows.append({
            "id": t.get("id"),

            # linear
            "slope_raw": float(t.get("slope_raw", np.nan)),
            "slope_inverted": float(t.get("slope_inverted", np.nan)),
            "intercept": float(t.get("intercept", np.nan)),
            "r2_lin": float(t.get("r2_lin", np.nan)),

            # exponential
            "exp_ok": bool(t.get("exp_ok", False)),
            "y0_exp": float(t.get("y0_exp", np.nan)),
            "A_exp": float(t.get("A_exp", np.nan)),
            "a": float(t.get("a", np.nan)),
            "tau_px": float(t.get("tau_px", np.nan)),
            "y_inf": float(t.get("y_inf", np.nan)),
            "r2_exp": float(t.get("r2_exp", np.nan)),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
