import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
from .expfit import _fit_exp_saturating
from .pfit import _fit_quadratic
from .linfit import _fit_linear
from .curaturesmothed import _fit_curvature_smoothed

# -----------------------------
# Registry: add new fits here later
# Each fitter returns dict of params including r2
# -----------------------------
FIT_REGISTRY = {
    "lin": _fit_linear,
    "exp": _fit_exp_saturating,
    "quad": _fit_quadratic,
    "curvature": _fit_curvature_smoothed,
}


def compute_track_stats(
    df,
    min_track_length=10,
    fits=("lin",),                 # e.g. ("lin","exp","quad")
    min_x_span_px=30,
        ):
    """
    df must have columns: id, frame, center_x, center_y

    fits: tuple/list of fit names from FIT_REGISTRY:
      - "lin"
      - "exp"
      - "quad"

    Returns list of dicts with:
      id, x_vals, y_vals, and per-fit scalar outputs stored with prefixes.
    """
    track_stats = []

    for obj_id, track in df.groupby("id"):
        if len(track) < min_track_length:
            continue

        track = track.sort_values("frame")
        x = track["center_x"].to_numpy(dtype=float)
        y = track["center_y"].to_numpy(dtype=float)

        # basic sanity check (optional)
        x_span = float(np.max(x) - np.min(x))

        out = {
            "id": obj_id,
            "x_vals": x,
            "y_vals": y,
        }

        for fit_name in fits:
            if fit_name not in FIT_REGISTRY:
                out[f"{fit_name}_ok"] = False
                out[f"{fit_name}_reason"] = "unknown_fit"
                continue

            # Example: skip exp if x-span too short (keep rule generic)
            if fit_name in ("exp", "quad") and x_span < min_x_span_px:
                out[f"{fit_name}_ok"] = False
                out[f"{fit_name}_reason"] = f"x_span_too_small<{min_x_span_px}"
                continue

            try:
                params = FIT_REGISTRY[fit_name](x, y)

                # require finite numbers for all params except maybe inf tau
                # (you can relax this if you want)
                for k, v in params.items():
                    if v is None:
                        continue
                    if isinstance(v, (int, float)) and (not np.isfinite(v)):
                        # allow tau_px to be inf
                        if not (fit_name == "exp" and k == "tau_px" and v == float("inf")):
                            raise ValueError(f"non_finite:{k}")

                out[f"{fit_name}_ok"] = True
                out[f"{fit_name}_reason"] = "ok"

                # store params with prefix
                for k, v in params.items():
                    out[f"{fit_name}_{k}"] = v

            except Exception as e:
                out[f"{fit_name}_ok"] = False
                out[f"{fit_name}_reason"] = f"fit_failed:{type(e).__name__}"

        # Convenience fields (keep your old ones if you like)
        # slope_inverted is meaningful only if linear fit ran
        if out.get("lin_ok", False):
            out["slope_raw"] = float(out["lin_slope"])
            out["slope_inverted"] = float(-1.0 * out["lin_slope"])
            out["intercept"] = float(out["lin_intercept"])
            out["r2_lin"] = float(out["lin_r2"])
            out["rmse_lin"] = float(out.get("lin_rmse", np.nan))
        else:
            out["slope_raw"] = np.nan
            out["slope_inverted"] = np.nan
            out["intercept"] = np.nan
            out["r2_lin"] = np.nan
            out["rmse_lin"] = np.nan

        # Backward-compatible exp fields (optional, if you want same keys)
        if out.get("exp_ok", False):
            out["y0_exp"] = out.get("exp_y0", np.nan)
            out["A_exp"]  = out.get("exp_A", np.nan)
            out["x0_exp"] = out.get("exp_x0", np.nan)
            out["y_inf"]  = out.get("exp_y_inf", np.nan)
            out["a"]      = out.get("exp_a", np.nan)
            out["tau_px"] = out.get("exp_tau_px", np.nan)
            out["r2_exp"] = out.get("exp_r2", np.nan)
        else:
            out["y0_exp"] = np.nan
            out["A_exp"]  = np.nan
            out["x0_exp"] = np.nan
            out["y_inf"]  = np.nan
            out["a"]      = np.nan
            out["tau_px"] = np.nan
            out["r2_exp"] = np.nan
    #Quadratic parameters
        if out.get("quad_ok", False):
            out["a2_quad"] = out.get("quad_a2", np.nan)
            out["b1_quad"] = out.get("quad_b1", np.nan)
            out["c0_quad"] = out.get("quad_c0", np.nan)
            out["r2_quad"] = out.get("quad_r2", np.nan)
            out["rmse_quad"] = float(out.get("quad_rmse", np.nan))
        else:
            out["a2_quad"] = np.nan
            out["b1_quad"] = np.nan
            out["c0_quad"] = np.nan
            out["r2_quad"] = np.nan
            out["rmse_quad"] = np.nan
        # Curvature based estimation
        if out.get("curvature_ok", False):
            out["curvature_mean"] = float(out.get("curvature_mean", np.nan))
            out["curvature_median"] = float(out.get("curvature_median", np.nan))
            out["curvature_max"] = float(out.get("curvature_max", np.nan))
            out["curvature_std"] = float(out.get("curvature_std", np.nan))
        else:
            out["curvature_mean"] = np.nan
            out["curvature_median"] = np.nan
            out["curvature_max"] = np.nan
            out["curvature_std"] = np.nan

        track_stats.append(out)

    return track_stats

def extract_slopes(track_stats):
    return np.array([d["slope_inverted"] for d in track_stats], dtype=float)




def save_track_parameters(track_stats, out_path):
    """
    Save scalar track parameters (no x/y arrays) to CSV.
    Automatically includes any new fit outputs you add later.
    """
    rows = []
    for t in track_stats:
        row = {}
        for k, v in t.items():
            if k in ("x_vals", "y_vals"):
                continue
            # skip non-scalars
            if isinstance(v, (list, tuple, np.ndarray)):
                continue
            row[k] = v
        rows.append(row)

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
