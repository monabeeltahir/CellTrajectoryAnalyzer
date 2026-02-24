import os
import numpy as np
import matplotlib.pyplot as plt

def _exp_saturating_predict(x, y0, A, a, x0):
    return y0 + A * (1.0 - np.exp(-a * (x - x0)))

def plot_exponential_parameter_histograms(
    exp_track_stats,
    out_folder,
    bins=80,
    r2_min=None,
    n_examples=8000,                  # for per-ID overlays
    overlay_include_linear=False,
    save_per_id_overlays=False,      # NEW
    save_combined_overlay=True      # NEW
):
    """
    Experiment-only:
      - histograms of exp-fit parameters (y_inf, a, tau, r2) [only exp_ok tracks]
      - overlays: actual trajectory always; exp fit only if exp_ok; if failed, it is shown as actual only.

    Requires each track dict to include:
      exp_ok, exp_reason, y_inf, a, tau_px, r2_exp,
      y0_exp, A_exp, x0_exp,
      slope_raw, intercept,
      x_vals, y_vals
    """

    os.makedirs(out_folder, exist_ok=True)

    # --- Histograms should use ONLY successful exponential fits ---
    valid = [t for t in exp_track_stats if t.get("exp_ok", False)]
    if r2_min is not None:
        valid = [t for t in valid if (t.get("r2_exp") is not None and t["r2_exp"] >= r2_min)]

    n_total = len(exp_track_stats)
    n_ok = len(valid)
    n_fail = n_total - n_ok
    print(f"[ExpFit] total={n_total} ok={n_ok} failed={n_fail}")

    if n_ok == 0:
        print("No valid exponential fits for histograms. (Overlays will still be produced.)")

    # -----------------------
    # Histograms (only if we have valid fits)
    # -----------------------
    if n_ok > 0:
        y_inf = np.array([t["y_inf"] for t in valid if t.get("y_inf") is not None], dtype=float)
        a_vals = np.array([t["a"] for t in valid if t.get("a") is not None], dtype=float)
        tau_vals = np.array([t["tau_px"] for t in valid if t.get("tau_px") is not None], dtype=float)
        r2_vals = np.array([t["r2_exp"] for t in valid if t.get("r2_exp") is not None], dtype=float)

        # y_inf
        plt.figure(figsize=(10, 6))
        plt.hist(y_inf, bins=bins, density=True)
        plt.title("Experiment: Asymptotic Deflection (y_inf)")
        plt.xlabel("y_inf (pixels)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "Exp_yinf_hist.png"), dpi=300)
        plt.close()

        # a
        plt.figure(figsize=(10, 6))
        plt.hist(a_vals, bins=bins, density=True)
        plt.title("Experiment: Rate Constant (a)")
        plt.xlabel("a (1/pixel)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "Exp_rate_hist.png"), dpi=300)
        plt.close()

        # tau
        plt.figure(figsize=(10, 6))
        finite_tau = tau_vals[np.isfinite(tau_vals)]
        plt.hist(finite_tau, bins=bins, density=True)
        plt.title("Experiment: Length Constant (tau = 1/a)")
        plt.xlabel("tau (pixels)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "Exp_tau_hist.png"), dpi=300)
        plt.close()

        # R^2
        plt.figure(figsize=(10, 6))
        plt.hist(r2_vals, bins=bins, density=True)
        plt.title("Experiment: Exponential Fit R²")
        plt.xlabel("R²")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "Exp_r2_hist.png"), dpi=300)
        plt.close()

    # -----------------------
    # Overlay plots
    # -----------------------
    overlay_dir = os.path.join(out_folder, "exp_fit_overlays_Expo")
    os.makedirs(overlay_dir, exist_ok=True)

    # For overlays: include ALL tracks (not only exp_ok)
    all_tracks = list(exp_track_stats)

    # Sort so best fits appear first (optional); keep failures too
    def sort_key(t):
        ok = 1 if t.get("exp_ok", False) else 0
        r2 = t.get("r2_exp", -1)
        return (ok, r2)

    all_sorted = sorted(all_tracks, key=sort_key, reverse=True)
    print("Number of successful exponential fits:",
      sum(t.get("exp_ok", False) for t in exp_track_stats))
    # ----- (1) Per-ID overlays (recommended) -----
    if save_per_id_overlays:
        shown = all_sorted[:min(n_examples, len(all_sorted))]
        for t in shown:
            x = np.asarray(t["x_vals"], dtype=float)
            y = np.asarray(t["y_vals"], dtype=float)

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y, linewidth=2, label="Actual")

            if t.get("exp_ok", False):
                y0 = float(t["y0_exp"])
                A  = float(t["A_exp"])
                a  = float(t["a"])
                x0 = float(t["x0_exp"])
                y_exp = _exp_saturating_predict(x, y0, A, a, x0)
                ax.plot(x, y_exp, linestyle="--", linewidth=2, label="Exp fit")

                if overlay_include_linear:
                    m = float(t["slope_raw"]); c = float(t["intercept"])
                    y_lin = m * x + c
                    ax.plot(x, y_lin, linestyle=":", linewidth=2, label="Linear fit")

                title = f"ID={t['id']} | R²exp={t.get('r2_exp', float('nan')):.3f} | a={t.get('a', float('nan')):.4g}"
            else:
                # show failure reason on the plot
                reason = t.get("exp_reason", "fit_failed")
                ax.text(0.02, 0.98, f"EXP FIT FAILED\n{reason}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=10)
                title = f"ID={t['id']} | EXP FIT FAILED"

            ax.set_title(title)
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
            ax.invert_yaxis()
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(overlay_dir, f"id_{t['id']}_overlay.png"), dpi=300)
            plt.close(fig)

    # ----- (2) One combined overlay figure (optional) -----
    if save_combined_overlay:
        # Use up to n_examples for readability
        shown = all_sorted[:min(n_examples, len(all_sorted))]

        plt.figure(figsize=(10, 6))
        label_actual = True
        label_exp = True
        label_lin = True

        for t in shown:
            x = np.asarray(t["x_vals"], dtype=float)
            y = np.asarray(t["y_vals"], dtype=float)
            order = np.argsort(x)
            x = x[order]; y = y[order]

            plt.plot(x, y, linewidth=1.5, label="Actual" if label_actual else None)
            label_actual = False

            if t.get("exp_ok", False):
                y0 = float(t["y0_exp"]); A = float(t["A_exp"]); a = float(t["a"]); x0 = float(t["x0_exp"])
                y_exp = _exp_saturating_predict(x, y0, A, a, x0)
                plt.plot(x, y_exp, linestyle="--", linewidth=1.5, label="Exp fit" if label_exp else None)
                label_exp = False

                if overlay_include_linear:
                    m = float(t["slope_raw"]); c = float(t["intercept"])
                    y_lin = m * x + c
                    plt.plot(x, y_lin, linestyle=":", linewidth=1.2, label="Linear fit" if label_lin else None)
                    label_lin = False

        plt.xlabel("x (pixels)")
        plt.ylabel("y (pixels)")
        plt.gca().invert_yaxis()
        plt.title(f"Overlays (top {min(n_examples, len(all_sorted))} tracks; failures shown as Actual-only)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, "overlay_expfit.png"), dpi=300)
        plt.close()

    print(f"Saved exponential parameter histograms to: {out_folder}")
    print(f"Saved trajectory overlays to: {overlay_dir}")
