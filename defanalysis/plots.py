import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def add_visual_rulers(ax, metrics, median_ctrl, correct_drift=True):
    mu_ctrl = metrics["ctrl_median"]
    mu_exp = metrics["exp_median"]
    gate = metrics["gate_threshold"]
    p_ctrl = metrics["ctrl_p"]

    y_min, y_max = ax.get_ylim()
    lane_1 = y_max * 0.5
    lane_2 = y_max * 0.2

    ax.axvline(mu_ctrl, color="m", alpha=0.6, linestyle="--", linewidth=2,
               label=f"Median Control ({median_ctrl:.3f})")
    ax.text(mu_ctrl, y_min * 1.1, "Ref", color="black", ha="left", fontsize=8, rotation=90)

    # noise width (control only)
    ax.annotate("", xy=(p_ctrl, lane_2), xytext=(mu_ctrl, lane_2),
                arrowprops=dict(arrowstyle="<->", color="yellow", lw=1.5))
    ax.text((mu_ctrl + p_ctrl) / 2, lane_2 * 1.2, "Noise", color="black", ha="center", fontsize=9)

    if correct_drift:
        ax.axvline(mu_exp, color="navy", linestyle=":", alpha=0.6)
        ax.annotate("", xy=(mu_exp, lane_1), xytext=(mu_ctrl, lane_1),
                    arrowprops=dict(arrowstyle="->", color="purple", lw=2))
        ax.text((mu_ctrl + mu_exp) / 2, lane_1 * 1.2, "Drift Correction",
                color="purple", ha="center", fontsize=9)

        ax.annotate("", xy=(gate, lane_2), xytext=(mu_exp, lane_2),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    else:
        ax.annotate("", xy=(gate, lane_2), xytext=(mu_ctrl, lane_2),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))

def plot_histogram(ctrl_slopes, exp_slopes, gate_threshold, metrics,
                   out_path, bin_count=500, correct_baseline_drift=False, xlabel="Slope (Deflection)", title=None, use_logx=False, 
                   eps=1e-12, robust_range=True, prc=(0.5, 99.5)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(14, 8))

    ctrl = np.asarray(ctrl_slopes, dtype=float)
    exp  = np.asarray(exp_slopes, dtype=float)
    ctrl = ctrl[np.isfinite(ctrl)]
    exp  = exp[np.isfinite(exp)]

    if use_logx:
        ctrl_plot = np.log10(np.clip(ctrl, eps, None))
        exp_plot  = np.log10(np.clip(exp,  eps, None))
    else:
        ctrl_plot = ctrl
        exp_plot  = exp

    if robust_range:
        lo = min(np.percentile(ctrl_plot, prc[0]), np.percentile(exp_plot, prc[0]))
        hi = max(np.percentile(ctrl_plot, prc[1]), np.percentile(exp_plot, prc[1]))
    else:
        lo = min(ctrl_plot.min(), exp_plot.min())
        hi = max(ctrl_plot.max(), exp_plot.max())

    shared_bins = np.linspace(lo, hi, bin_count)


    plt.hist(ctrl_plot, bins=shared_bins, alpha=0.7, color="green",
             label="Control", density=True)

    n, bins, patches = plt.hist(exp_plot, bins=shared_bins, alpha=0.8,
                                density=True, label="Experiment")

    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i + 1]) / 2
        patch.set_facecolor("red" if bin_center > gate_threshold else "blue")

    plt.axvline(gate_threshold, color="black", linestyle="--", linewidth=2,
                label=f"Gate ({gate_threshold:.3f})")

    add_visual_rulers(plt.gca(), metrics, median_ctrl=metrics["ctrl_median"],
                      correct_drift=correct_baseline_drift)

    plt.yscale("log")
    plt.legend(fontsize=14)
    plt.tick_params(axis="both", labelsize=16)
    plt.xlabel(xlabel, fontsize=18)
    if title is not None:
        plt.title(title)

    plt.ylabel("Frequency (Log Scale)", fontsize=18)

    #plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_classified_trajectories(exp_track_stats, gate_threshold, out_path, max_lines=1000):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(12, 8))
    sample = exp_track_stats[:max_lines] if len(exp_track_stats) > max_lines else exp_track_stats

    for t in sample:
        is_deflected = (t["slope_inverted"] > gate_threshold)
        color = "red" if is_deflected else "blue"
        alpha = 0.8 if is_deflected else 0.7

        x = t["x_vals"]
        y = t["y_vals"]

        plt.plot(x, y, color=color, alpha=alpha, linewidth=1)

        # fitted line overlay (dashed)
        y_fit = t["slope_raw"] * x + t["intercept"]
        plt.plot(x, y_fit, color=color, linestyle="--", alpha=alpha, linewidth=0.5)

    plt.gca().invert_yaxis()
    plt.title(f"Classified Trajectories (Gate: {gate_threshold:.3f})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    legend_elements = [
        Line2D([0], [0], color="blue", label="Straight"),
        Line2D([0], [0], color="red", label="Deflected"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    #plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
