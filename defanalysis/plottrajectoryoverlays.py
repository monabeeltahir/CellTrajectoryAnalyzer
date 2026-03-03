import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- predictors ----------
def _pred_exp(x, y0, A, a, x0):
    return y0 + A * (1.0 - np.exp(-a * (x - x0)))

def _pred_lin(x, m, c):
    return m * x + c

def _pred_quad(x, a2, b1, c0):
    return a2 * x**2 + b1 * x + c0


# ---------- generic helpers ----------
def _finite(arr):
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a)]

def _get_valid_tracks(track_stats, ok_key, r2_key=None, r2_min=None):
    valid = [t for t in track_stats if t.get(ok_key, False)]
    if r2_key is not None and r2_min is not None:
        valid = [t for t in valid if (t.get(r2_key) is not None and np.isfinite(t[r2_key]) and t[r2_key] >= r2_min)]
    return valid


def plot_fit_diagnostics(
    track_stats,
    out_folder,
    *,
    fits=("lin", "exp", "quad"),
    # histogram settings
    bins=80,
    r2_min=None,                 # applies to each fit's r2 key if provided
    # overlay settings
    n_examples=300,
    save_per_id_overlays=False,
    save_combined_overlay=True,
    overlay_include_linear=False,  # when plotting exp/quad overlays, also show linear if available
):
    """
    One function for:
      (A) parameter histograms for all enabled fits
      (B) trajectory overlays for all enabled fits

    Assumes these keys exist (as applicable):
      Linear:
        lin_ok, lin_slope, lin_intercept, lin_r2
        (or fallback: slope_raw/intercept/r2_lin already mapped in your code)
      Exponential:
        exp_ok, y0_exp, A_exp, x0_exp, a, y_inf, tau_px, r2_exp
      Quadratic:
        quad_ok, quad_a2, quad_b1, quad_c0, quad_r2
    """

    os.makedirs(out_folder, exist_ok=True)

    # ---- define fit specs (what params to histogram + how to predict) ----
    FIT_SPECS = {
        "lin": {
            "ok_key": "lin_ok",
            "r2_key": "lin_r2",
            "params": [("lin_slope", "slope"), ("lin_intercept", "intercept"), ("lin_r2", "R²")],
            "predict": lambda x, t: _pred_lin(x, float(t["lin_slope"]), float(t["lin_intercept"])),
        },
        "exp": {
            "ok_key": "exp_ok",
            "r2_key": "r2_exp",
            "params": [("y_inf", "y_inf"), ("a", "a"), ("tau_px", "tau"), ("r2_exp", "R²"),
                       ("A_exp", "A"), ("y0_exp", "y0"), ("x0_exp", "x0")],
            "predict": lambda x, t: _pred_exp(x, float(t["y0_exp"]), float(t["A_exp"]), float(t["a"]), float(t["x0_exp"])),
        },
        "quad": {
            "ok_key": "quad_ok",
            "r2_key": "quad_r2",
            "params": [("quad_a2", "a2"), ("quad_b1", "b1"), ("quad_c0", "c0"), ("quad_r2", "R²")],
            "predict": lambda x, t: _pred_quad(x, float(t["quad_a2"]), float(t["quad_b1"]), float(t["quad_c0"])),
        }
    }

    # Optional: allow linear fallback from your mapped keys if you aren't storing lin_slope/lin_intercept
    def _ensure_lin_fields(t):
        if "lin_slope" not in t and "slope_raw" in t:
            t["lin_slope"] = t["slope_raw"]
        if "lin_intercept" not in t and "intercept" in t:
            t["lin_intercept"] = t["intercept"]
        if "lin_r2" not in t and "r2_lin" in t:
            t["lin_r2"] = t["r2_lin"]
        if "lin_ok" not in t:
            t["lin_ok"] = np.isfinite(t.get("lin_slope", np.nan))

    # ---- apply fallback mapping for all tracks ----
    for t in track_stats:
        _ensure_lin_fields(t)
    overlay_dir = os.path.join(out_folder, "fit_overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    # ==========================================
    # (A) PARAMETER HISTOGRAMS
    # ==========================================
    for fit in fits:
        if fit not in FIT_SPECS:
            continue
        spec = FIT_SPECS[fit]
        ok_key = spec["ok_key"]
        r2_key = spec.get("r2_key")
        fit_dir = os.path.join(overlay_dir, fit)
        os.makedirs(fit_dir, exist_ok=True)
        valid = _get_valid_tracks(track_stats, ok_key, r2_key=r2_key, r2_min=r2_min)
        print(f"[{fit.upper()}] total={len(track_stats)} ok={len(valid)} fail={len(track_stats)-len(valid)}")

        if len(valid) == 0:
            continue

        for key, label in spec["params"]:
            vals = [t.get(key) for t in valid if t.get(key) is not None]
            arr = _finite(vals)
            if arr.size == 0:
                continue

            plt.figure(figsize=(10, 6))
            plt.hist(arr, bins=bins, density=True)
            plt.title(f"{fit.upper()} parameter histogram: {label}  (n={arr.size})")
            plt.xlabel(label)
            plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig(os.path.join(fit_dir, f"{fit}_{key}_hist.png"), dpi=300)
            plt.close()

    # ==========================================
    # (B) OVERLAYS
    # ==========================================
    

    # Sort best-first for nicer combined overlay
    def _sort_key(t, fit):
        spec = FIT_SPECS[fit]
        ok = 1 if t.get(spec["ok_key"], False) else 0
        r2 = t.get(spec.get("r2_key", ""), -1)
        r2 = r2 if (r2 is not None and np.isfinite(r2)) else -1
        return (ok, r2)

    for fit in fits:
        if fit not in FIT_SPECS:
            continue
        spec = FIT_SPECS[fit]
        ok_key = spec["ok_key"]

        # sort by fit quality for that fit
        sorted_tracks = sorted(track_stats, key=lambda t: _sort_key(t, fit), reverse=True)
        shown = sorted_tracks[:min(n_examples, len(sorted_tracks))]

        
        fit_dir = os.path.join(overlay_dir, fit)
        # ---- per-ID overlays ----
        if save_per_id_overlays:
            for t in shown:
                x = np.asarray(t["x_vals"], dtype=float)
                y = np.asarray(t["y_vals"], dtype=float)
                order = np.argsort(x)
                x = x[order]; y = y[order]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x, y, linewidth=2, label="Actual")

                if t.get(ok_key, False):
                    y_fit = spec["predict"](x, t)
                    ax.plot(x, y_fit, linestyle="--", linewidth=2, label=f"{fit} fit")

                    if overlay_include_linear and fit != "lin" and t.get("lin_ok", False):
                        y_lin = _pred_lin(x, float(t["lin_slope"]), float(t["lin_intercept"]))
                        ax.plot(x, y_lin, linestyle=":", linewidth=1.5, label="Linear fit")

                    r2_key = spec.get("r2_key", None)
                    r2 = t.get(r2_key, np.nan) if r2_key else np.nan
                    title = f"ID={t.get('id')} | {fit} ok | R²={float(r2):.3f}" if np.isfinite(r2) else f"ID={t.get('id')} | {fit} ok"
                else:
                    reason = t.get(f"{fit}_reason", t.get("exp_reason", "fit_failed"))
                    ax.text(0.02, 0.98, f"{fit.upper()} FIT FAILED\n{reason}",
                            transform=ax.transAxes, va="top", ha="left", fontsize=10)
                    title = f"ID={t.get('id')} | {fit.upper()} FIT FAILED"

                ax.set_title(title)
                ax.set_xlabel("x (pixels)")
                ax.set_ylabel("y (pixels)")
                ax.invert_yaxis()
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(fit_dir, f"id_{t.get('id')}_overlay.png"), dpi=300)
                plt.close(fig)

        # ---- combined overlay ----
        if save_combined_overlay:
            plt.figure(figsize=(10, 6))
            label_actual = True
            label_fit = True
            label_lin = True

            for t in shown:
                x = np.asarray(t["x_vals"], dtype=float)
                y = np.asarray(t["y_vals"], dtype=float)
                order = np.argsort(x)
                x = x[order]; y = y[order]

                plt.plot(x, y, linewidth=1.2, label="Actual" if label_actual else None)
                label_actual = False

                if t.get(ok_key, False):
                    y_fit = spec["predict"](x, t)
                    plt.plot(x, y_fit, linestyle="--", linewidth=1.2, label=f"{fit} fit" if label_fit else None)
                    label_fit = False

                    if overlay_include_linear and fit != "lin" and t.get("lin_ok", False):
                        y_lin = _pred_lin(x, float(t["lin_slope"]), float(t["lin_intercept"]))
                        plt.plot(x, y_lin, linestyle=":", linewidth=1.0, label="Linear fit" if label_lin else None)
                        label_lin = False

            plt.xlabel("x (pixels)")
            plt.ylabel("y (pixels)")
            plt.gca().invert_yaxis()
            plt.title(f"{fit.upper()} overlays (top {min(n_examples, len(sorted_tracks))} tracks)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fit_dir, f"overlay_{fit}.png"), dpi=300)
            plt.close()

    print(f"Saved fit parameter histograms + overlays to: {out_folder}")
    print(f"Saved overlays to: {overlay_dir}")