import os
import numpy as np
import pandas as pd

from .plottrajectoryoverlays import plot_fit_diagnostics
from .io import ensure_dir, load_tracks_csv, save_summary_csv
from .track_stats import compute_track_stats, extract_slopes, save_track_parameters
from .gating import compute_gate, classify_tracks, percentile_clip
from .plots import plot_histogram, plot_classified_trajectories
from .report import summarize_results, print_report
from .plot_expopara import plot_exponential_parameter_histograms
from .plot_curvature import plot_curvature_overlays

def _save_params(stats, out_folder, name):
    save_track_parameters(stats, os.path.join(out_folder, name))


def _extract_feature(stats, key, ok_key=None):
    """
    Extract a scalar numeric feature from track_stats.
    If ok_key is provided, keep only tracks where t[ok_key] is True.
    """
    vals = []
    for t in stats:
        if ok_key is not None and not t.get(ok_key, False):
            continue
        v = t.get(key, None)
        if v is None:
            continue
        vals.append(v)

    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _gate_plot_and_classify_feature(
    ctrl_stats,
    exp_stats,
    exp_out_folder,
    ctrl_out_folder,
    *,
    feature_key,
    ok_key=None,
    sensitivity,
    correct_baseline_drift,
    bin_count,
    png_name,
    csv_name,
    ctrl_csv_name=None,
    xlabel=None,
    title=None,
    clip_low=0.1,
    clip_high=99.9,
    use_logx=False,
):
    """
    For a given feature:
      - compute gate (CTRL vs EXP)
      - save histogram (EXP folder)
      - classify BOTH exp + ctrl
      - save exp CSV → exp_out_folder
      - save ctrl CSV → ctrl_out_folder
    """

    ctrl = _extract_feature(ctrl_stats, feature_key, ok_key=ok_key)
    exp  = _extract_feature(exp_stats,  feature_key, ok_key=ok_key)

    if ctrl.size == 0 or exp.size == 0:
        print(f"[WARN] Not enough data for feature '{feature_key}' (ok_key={ok_key}). Skipping.")
        return None, None, [], [], None, None

    ctrl_plot = percentile_clip(ctrl, low=clip_low, high=clip_high)
    exp_plot  = percentile_clip(exp,  low=clip_low, high=clip_high)

    gate, metrics = compute_gate(
        ctrl_plot, exp_plot,
        sensitivity=sensitivity,
        correct_baseline_drift=correct_baseline_drift
    )

    # Ensure directories exist
    ensure_dir(exp_out_folder)
    ensure_dir(ctrl_out_folder)

    # -------------------------
    # Save histogram (keep in experiment folder)
    # -------------------------
    plot_histogram(
        ctrl_plot, exp_plot, gate, metrics,
        out_path=os.path.join(exp_out_folder, png_name),
        bin_count=bin_count,
        correct_baseline_drift=correct_baseline_drift,
        xlabel=xlabel or feature_key,
        title=title,
        use_logx=use_logx,
        eps=1e-12,
        robust_range=True,
        prc=(0.5, 99.5),
    )

    # -------------------------
    # Experiment classification
    # -------------------------
    exp_rows = classify_tracks(exp_stats, gate, feature_key=feature_key)
    pd.DataFrame(exp_rows).to_csv(
        os.path.join(exp_out_folder, csv_name), index=False
    )

    # -------------------------
    # Control classification
    # -------------------------
    if ctrl_csv_name is None:
        base, ext = os.path.splitext(csv_name)
        ctrl_csv_name = f"{base}_control{ext}"

    ctrl_rows = classify_tracks(ctrl_stats, gate, feature_key=feature_key)
    pd.DataFrame(ctrl_rows).to_csv(
        os.path.join(ctrl_out_folder, ctrl_csv_name), index=False
    )

    return gate, metrics, exp_rows, ctrl_rows, ctrl_plot, exp_plot


def DefAnalysis_Dynamic(
    ExperimentFile,
    ControlFile,
    OutputFolder,
    CntrlOutputFolder,
    min_track_length=10,
    sensitivity=99.9,
    bin_count=500,
    correct_baseline_drift=False,
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
    fits = ["lin", "quad", "curvature"]
    if fit_exponential_exp:
        fits.append("exp")
    

    ctrl_stats = compute_track_stats(
        df_ctrl,
        min_track_length=min_track_length,
        fits=fits,
        min_x_span_px=exp_min_x_span_px
    )

    exp_stats = compute_track_stats(
        df_exp,
        min_track_length=min_track_length,
        fits=fits,
        min_x_span_px=exp_min_x_span_px
    )

    if not ctrl_stats or not exp_stats:
        print("Error: insufficient data.")
        return None
    

    ## Saving the curvature data to respective folders for the trajectory smoothing and curvature calculations
    plot_curvature_overlays(
    exp_stats,
    out_folder=os.path.join(OutputFolder, "Curvature"),
    n_examples=300,
    window_length=9,
    polyorder=3,
    save_per_id=False,
    save_combined=True,
)

    plot_curvature_overlays(
    ctrl_stats,
    out_folder=os.path.join(CntrlOutputFolder, "Curvature"),
    n_examples=300,
    window_length=9,
    polyorder=3,
    save_per_id=False,
    save_combined=True,
)
    # -------------------------
    # Save parameter CSVs
    # -------------------------
    _save_params(ctrl_stats, CntrlOutputFolder, "track_parameters_control.csv")
    _save_params(exp_stats,  OutputFolder,      "track_parameters_experiment.csv")

    # -------------------------
    # Primary gating on slope (existing)
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

    extra_plots = {}
    feature_gates = {}   # ✅ collect everything here
    ## Linear Other Features
    lin_features = [
    ("rmse_lin", "Linear fit RMSE", bin_count),
    ("r2_lin", "Linear fit R²", bin_count),
    ("slope_inverted", "Linear Slope (Deflection)", bin_count),
    ("intercept", "Linear Intercept", bin_count),
]
    for key, xlabel, bins_ in lin_features:
        gate_l, metrics_l, rows_l, ctrl_rows_l, ctrl_l_plot, exp_l_plot = _gate_plot_and_classify_feature(
    ctrl_stats, exp_stats,
    OutputFolder + "/Linear/",
    CntrlOutputFolder + "/Linear/",
    feature_key=key,
    ok_key="lin_ok",
    sensitivity=sensitivity,
    correct_baseline_drift=correct_baseline_drift,
    bin_count=bins_,
    png_name=f"Histogram_{key}_Gate.png",
    csv_name=f"deflection_summary_{key}_gate.csv",
    xlabel=xlabel,
)

        if gate_l is not None:
            extra_plots[key] = os.path.join(OutputFolder, f"Histogram_{key}_Gate.png")

        # store for report
            feature_gates[key] = {
            "ctrl": ctrl_l_plot,
            "exp": exp_l_plot,
            "rows": rows_l,
            "metrics": metrics_l
        }
    # -------------------------
    # Extra histograms + gates for ALL fits
    # -------------------------
    curvature_features = [
    ("curvature_mean", "Curvature mean", bin_count),
    ("curvature_median", "Curvature median", bin_count),
    ("curvature_max", "Curvature max", bin_count),
    ("curvature_std", "Curvature std", bin_count),
     ("curvature_p90", "Curvature Percentile", bin_count),
    ]

    for key, xlabel, bins_ in curvature_features:
        gate_c, metrics_c, rows_c, ctrl_rows_c, ctrl_c_plot, exp_c_plot = _gate_plot_and_classify_feature(
    ctrl_stats, exp_stats,
    OutputFolder + "/Curvature/",
    CntrlOutputFolder + "/Curvature/",
    feature_key=key,
    ok_key="curvature_ok",
    sensitivity=sensitivity,
    correct_baseline_drift=correct_baseline_drift,
    bin_count=bins_,
    png_name=f"Histogram_{key}_Gate.png",
    csv_name=f"deflection_summary_{key}_gate.csv",
    xlabel=xlabel,
)

        if gate_c is not None:
            extra_plots[key] = os.path.join(OutputFolder, f"Histogram_{key}_Gate.png")

            feature_gates[key] = {
            "ctrl": ctrl_c_plot,
            "exp": exp_c_plot,
            "rows": rows_c,
            "metrics": metrics_c
                }
    # ---- Quadratic parameter gates/histograms (CTRL vs EXP) ----
    # Requires: quad_ok + quad_a2, quad_b1, quad_c0, quad_r2
    quad_features = [
        ("a2_quad", "Quadratic coefficient a2", bin_count),
        ("b1_quad", "Quadratic coefficient b1", bin_count),
        ("c0_quad", "Quadratic coefficient c0", bin_count),
        ("r2_quad", "Quadratic fit R²",         bin_count),
        ("rmse_quad", "Quadratic fit RMSE", bin_count),
    ]
    for key, xlabel, bins_ in quad_features:
        gate_q, metrics_q, rows_q, ctrl_rows_q, ctrl_q_plot, exp_q_plot = _gate_plot_and_classify_feature(
    ctrl_stats, exp_stats,
    OutputFolder + "/Quad/",
    CntrlOutputFolder + "/Quad/",
    feature_key=key,
    ok_key="quad_ok",
    sensitivity=sensitivity,
    correct_baseline_drift=correct_baseline_drift,
    bin_count=bins_,
    png_name=f"Histogram_{key}_Gate.png",
    csv_name=f"deflection_summary_{key}_gate.csv",
    xlabel=xlabel,
)
        if gate_q is not None:
            extra_plots[key] = os.path.join(OutputFolder, f"Histogram_{key}_Gate.png")
                # ✅ NEW: store for report
            feature_gates[key] = {
            "ctrl": ctrl_q_plot,
            "exp": exp_q_plot,
            "rows": rows_q,
            "metrics": metrics_q
        }


    # ---- Exponential parameter gates/histograms (CTRL vs EXP) ----
    # Requires: exp_ok + a, tau_px, y_inf, r2_exp, y0_exp, A_exp, x0_exp
    ctrl_a_plot = np.array([], dtype=float)
    exp_a_plot  = np.array([], dtype=float)
    rows_a = []
    metrics_a = {"mode_name": "n/a"}
    gate_a = None

    if fit_exponential_exp:
        exp_features = [
            ("a",      "Decay rate a (1/pixel)", 150),
            ("tau_px", "Tau (pixels)",           150),
            ("y_inf",  "Asymptotic y_inf (px)",  150),
            ("r2_exp", "Exponential fit R²",     150),
            ("A_exp",  "Exponential A (px)",     150),
            ("y0_exp", "Exponential y0 (px)",    150),
            ("x0_exp", "Exponential x0 (px)",    150),
        ]

        for key, xlabel, bins_ in exp_features:
            gate_e, metrics_e, rows_e,ctrl_e_plot, exp_e_plot = _gate_plot_and_classify_feature(
                ctrl_stats, exp_stats, OutputFolder,
                feature_key=key,
                ok_key="exp_ok",
                sensitivity=sensitivity,
                correct_baseline_drift=correct_baseline_drift,
                bin_count=bins_,
                png_name=f"Histogram_{key}_Gate.png",
                csv_name=f"deflection_summary_{key}_gate.csv",
                xlabel=xlabel,
                title=None
            )
            if gate_e is not None:
                extra_plots[key] = os.path.join(OutputFolder, f"Histogram_{key}_Gate.png")
                # ✅ NEW: store for report
                feature_gates[key] = {
                "ctrl": ctrl_e_plot,
                "exp": exp_e_plot,
                "rows": rows_e,
                "metrics": metrics_e
            }
            

            # keep backward-compatible variables for report’s a-gate block
            # keep backward-compatible variables for report’s a-gate block
            if key == "a" and gate_e is not None:
                gate_a = gate_e
                metrics_a = metrics_e
                rows_a = rows_e
                ctrl_a_plot = ctrl_e_plot   # ✅ exact same arrays used for gate/hist
                exp_a_plot  = exp_e_plot
        # Existing exp-only parameter histograms + overlays (your old function)
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
    # Experiment diagnostics
    plot_fit_diagnostics(
    exp_stats,
    out_folder=OutputFolder,
    fits=("lin", "quad", "exp") if fit_exponential_exp else ("lin", "quad"),
    bins=exp_hist_bins,
    r2_min=exp_r2_min_for_hists,
    n_examples=30,
    save_per_id_overlays=False,
    save_combined_overlay=True,
    overlay_include_linear=False,
)

# Control diagnostics
    plot_fit_diagnostics(
    ctrl_stats,
    out_folder=CntrlOutputFolder,
    fits=("lin", "quad", "exp") if fit_exponential_exp else ("lin", "quad"),
    bins=exp_hist_bins,
    r2_min=exp_r2_min_for_hists,
    n_examples=30,
    save_per_id_overlays=False,
    save_combined_overlay=True,
    overlay_include_linear=False,
)
    # -------------------------
    # Report
    # -------------------------
    summary = summarize_results(
        ctrl_slopes, exp_slopes, final_rows, metrics,
        ctrl_a=ctrl_a_plot, exp_a=exp_a_plot,
        final_rows_a=rows_a, metrics_a=metrics_a,
        feature_gates=feature_gates     # ✅ NEW
    )
    print_report(summary, OutputFolder)

    plots_out = {
        "histogram": os.path.join(OutputFolder, "Histogram_Analysis_StaticGate.png"),
        "trajectories": os.path.join(OutputFolder, "Trajectories_Classified.png"),
        "histogram_a": os.path.join(OutputFolder, "Histogram_a_Gate.png") if (fit_exponential_exp and gate_a is not None) else "",
    }
    # add all extra fit histograms
    plots_out.update(extra_plots)

    return {
        "summary_csv": summary_csv,
        "gate_threshold": gate_threshold,
        "metrics": metrics,
        "summary": summary,
        "plots": plots_out
    }