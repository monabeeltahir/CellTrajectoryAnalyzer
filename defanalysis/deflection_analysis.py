import os
import numpy as np
import pandas as pd

from .io import ensure_dir, load_tracks_csv, save_summary_csv
from .track_stats import compute_track_stats, extract_slopes, save_track_parameters
from .gating import compute_gate, classify_tracks, percentile_clip
from .plots import plot_histogram, plot_classified_trajectories
from .report import summarize_results, print_report
from .plot_expopara import plot_exponential_parameter_histograms


def _save_params(stats, out_folder, name):
    save_track_parameters(stats, os.path.join(out_folder, name))


def _gate_plot_and_classify(
    ctrl_feat,
    exp_feat,
    exp_stats,
    out_folder,
    *,
    sensitivity,
    correct_baseline_drift,
    bin_count,
    png_name,
    csv_name,
    feature_key=None,
    xlabel=None,
    title=None,
    use_logx=False,
):
    """
    Generic helper: compute gate -> plot histogram -> classify exp tracks -> save CSV.
    If feature_key is None, classify_tracks defaults to slope_inverted (existing behavior).
    """
    gate, metrics = compute_gate(
        ctrl_feat,
        exp_feat,
        sensitivity=sensitivity,
        correct_baseline_drift=correct_baseline_drift
    )

    plot_histogram(
        ctrl_feat, exp_feat, gate, metrics,
        out_path=os.path.join(out_folder, png_name),
        bin_count=bin_count,
        correct_baseline_drift=correct_baseline_drift,
        xlabel=xlabel,
        title=title,
        use_logx=use_logx,
        eps=1e-12,
        robust_range=True,
        prc=(0.5, 99.5),
    )

    rows = classify_tracks(exp_stats, gate, feature_key=feature_key) if feature_key else classify_tracks(exp_stats, gate)
    pd.DataFrame(rows).to_csv(os.path.join(out_folder, csv_name), index=False)

    return gate, metrics, rows


def _extract_valid_feature(stats, key):
    arr = np.array(
        [t.get(key) for t in stats if t.get("exp_ok", False) and t.get(key) is not None],
        dtype=float
    )
    return arr[np.isfinite(arr)]


def DefAnalysis_Dynamic(
    ExperimentFile,
    ControlFile,
    OutputFolder,
    CntrlOutputFolder,
    min_track_length=10,
    sensitivity=99.9,
    bin_count=500,
    correct_baseline_drift=False,      # True means baseline is CONTROL? (keeping your comment/behavior)
    max_traj_lines=1000,
    fit_exponential_exp=True,
    exp_min_x_span_px=30,
    exp_r2_min_for_hists=None,
    exp_hist_bins=500
):
    ensure_dir(OutputFolder)
    ensure_dir(CntrlOutputFolder)
    print(f"Processing... (Drift Correction: {correct_baseline_drift})")

    # -------------------------
    # Load tracks
    # -------------------------
    df_ctrl = load_tracks_csv(ControlFile)
    df_exp  = load_tracks_csv(ExperimentFile)

    # -------------------------
    # Compute track stats
    # -------------------------
    ctrl_stats = compute_track_stats(
        df_ctrl,
        min_track_length=min_track_length,
        fit_exponential=fit_exponential_exp,
        min_x_span_px=exp_min_x_span_px,   # keep consistent gate rule
    )
    exp_stats = compute_track_stats(
        df_exp,
        min_track_length=min_track_length,
        fit_exponential=fit_exponential_exp,
        min_x_span_px=exp_min_x_span_px,
    )

    if not ctrl_stats or not exp_stats:
        print("Error: insufficient data.")
        return None

    # -------------------------
    # Save parameter CSVs
    # -------------------------
    _save_params(ctrl_stats, CntrlOutputFolder, "track_parameters_control.csv")
    _save_params(exp_stats,  OutputFolder,      "track_parameters_experiment.csv")

    # -------------------------
    # Primary gating on slope
    # -------------------------
    ctrl_slopes = extract_slopes(ctrl_stats)
    exp_slopes  = extract_slopes(exp_stats)

    gate_threshold, metrics = compute_gate(
        ctrl_slopes, exp_slopes,
        sensitivity=sensitivity,
        correct_baseline_drift=correct_baseline_drift
    )

    final_rows = classify_tracks(exp_stats, gate_threshold)
    summary_csv = os.path.join(OutputFolder, "deflection_summary_dynamic.csv")
    save_summary_csv(final_rows, summary_csv)

    plot_histogram(
        ctrl_slopes, exp_slopes, gate_threshold, metrics,
        out_path=os.path.join(OutputFolder, "Histogram_Analysis_StaticGate.png"),
        bin_count=bin_count,
        correct_baseline_drift=correct_baseline_drift
    )

    plot_classified_trajectories(
        exp_stats, gate_threshold,
        out_path=os.path.join(OutputFolder, "Trajectories_Classified.png"),
        max_lines=max_traj_lines
    )

    # -------------------------
    # Optional: exponential "a" gating + parameter histograms
    # -------------------------
    ctrl_a_plot = np.array([], dtype=float)
    exp_a_plot  = np.array([], dtype=float)
    rows_a = []
    metrics_a = {"mode_name": "n/a"}
    gate_a = None

    if fit_exponential_exp:
        ctrl_a = _extract_valid_feature(ctrl_stats, "a")
        exp_a  = _extract_valid_feature(exp_stats,  "a")

        if len(ctrl_a) > 0 and len(exp_a) > 0:
            # percentile clipping for plotting/gating robustness
            ctrl_a_plot = percentile_clip(ctrl_a, low=0.1, high=99.9)
            exp_a_plot  = percentile_clip(exp_a,  low=0.1, high=99.9)

            gate_a, metrics_a, rows_a = _gate_plot_and_classify(
                ctrl_a_plot,
                exp_a_plot,
                exp_stats,
                OutputFolder,
                sensitivity=sensitivity,
                correct_baseline_drift=correct_baseline_drift,
                bin_count=150,
                png_name="Histogram_a_Gate.png",
                csv_name="deflection_summary_a_gate.csv",
                feature_key="a",
                xlabel="Decay rate a (1/pixel)",
                title=f"Decay-rate gating: {metrics_a.get('mode_name','')}",
                use_logx=False,
            )
        else:
            print("[WARN] Not enough valid exp fits to compute a-gate histogram.")

        # Parameter histograms (exp + control)
        plot_exponential_parameter_histograms(
            exp_stats,
            out_folder=OutputFolder,
            bins=exp_hist_bins,
            r2_min=exp_r2_min_for_hists
        )
        plot_exponential_parameter_histograms(
            ctrl_stats,
            out_folder=CntrlOutputFolder,
            bins=exp_hist_bins,
            r2_min=exp_r2_min_for_hists
        )

    # -------------------------
    # Report
    # -------------------------
    summary = summarize_results(
        ctrl_slopes, exp_slopes, final_rows, metrics,
        ctrl_a=ctrl_a_plot, exp_a=exp_a_plot,
        final_rows_a=rows_a, metrics_a=metrics_a
    )
    print_report(summary, OutputFolder)

    return {
        "summary_csv": summary_csv,
        "gate_threshold": gate_threshold,
        "metrics": metrics,
        "summary": summary,
        "plots": {
            "histogram": os.path.join(OutputFolder, "Histogram_Analysis_StaticGate.png"),
            "trajectories": os.path.join(OutputFolder, "Trajectories_Classified.png"),
            "histogram_a": os.path.join(OutputFolder, "Histogram_a_Gate.png") if (fit_exponential_exp and gate_a is not None) else "",
        }
    }