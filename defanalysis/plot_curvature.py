import os
import numpy as np
import matplotlib.pyplot as plt
from .curaturesmothed import _curvature_smoothed_derivative_raw

def plot_curvature_overlays(
    track_stats,
    out_folder,
    n_examples=200,
    window_length=9,
    polyorder=3,
    save_per_id=True,
    save_combined=True,
):
    os.makedirs(out_folder, exist_ok=True)

    overlay_dir = os.path.join(out_folder, "curvature_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    valid_tracks = [t for t in track_stats if t.get("curvature_ok", False)]

    if len(valid_tracks) == 0:
        print("[Curvature] No valid curvature tracks found.")
        return

    shown = valid_tracks[:min(n_examples, len(valid_tracks))]

    # -------------------------
    # Per-track plots
    # -------------------------
    if save_per_id:
        for t in shown:
            x = np.asarray(t["x_vals"], dtype=float)
            y = np.asarray(t["y_vals"], dtype=float)

            x_s, y_raw, y_smooth, kappa = _curvature_smoothed_derivative_raw(
                x, y,
                window_length=window_length,
                polyorder=polyorder
            )

            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            # Top panel: trajectory overlay
            axes[0].plot(x_s, y_raw, linewidth=2, label="Original")
            axes[0].plot(x_s, y_smooth, linestyle="--", linewidth=2, label="Smoothed")
            axes[0].invert_yaxis()
            axes[0].set_ylabel("y (pixels)")
            axes[0].set_title(f"ID={t['id']} | Smoothed trajectory")
            axes[0].legend()

            # Bottom panel: curvature
            if len(kappa) > 0:
                axes[1].plot(x_s, kappa, linewidth=2, label="Curvature")
            axes[1].set_xlabel("x (pixels)")
            axes[1].set_ylabel("Curvature")
            axes[1].set_title(
                f"mean={t.get('curvature_mean', np.nan):.4g}, "
                f"median={t.get('curvature_median', np.nan):.4g}, "
                f"max={t.get('curvature_max', np.nan):.4g}"
            )
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(os.path.join(overlay_dir, f"id_{t['id']}_curvature_overlay.png"), dpi=300)
            plt.close(fig)

    # -------------------------
    # Combined overlay plot
    # -------------------------
    if save_combined:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        label_raw = True
        label_smooth = True
        label_curv = True

        for t in shown:
            x = np.asarray(t["x_vals"], dtype=float)
            y = np.asarray(t["y_vals"], dtype=float)

            x_s, y_raw, y_smooth, kappa = _curvature_smoothed_derivative_raw(
                x, y,
                window_length=window_length,
                polyorder=polyorder
            )

            axes[0].plot(x_s, y_raw, linewidth=1.2, alpha=0.8, label="Original" if label_raw else None)
            axes[0].plot(x_s, y_smooth, linestyle="--", linewidth=1.2, alpha=0.8, label="Smoothed" if label_smooth else None)

            if len(kappa) > 0:
                axes[1].plot(x_s, kappa, linewidth=1.0, alpha=0.7, label="Curvature" if label_curv else None)

            label_raw = False
            label_smooth = False
            label_curv = False

        axes[0].invert_yaxis()
        axes[0].set_ylabel("y (pixels)")
        axes[0].set_title(f"Curvature trajectory overlays (top {len(shown)} tracks)")
        axes[0].legend()

        axes[1].set_xlabel("x (pixels)")
        axes[1].set_ylabel("Curvature")
        axes[1].set_title("Curvature vs x")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(overlay_dir, "curvature_overlay_combined.png"), dpi=300)
        plt.close(fig)

    print(f"[Curvature] Saved curvature overlays to: {overlay_dir}")