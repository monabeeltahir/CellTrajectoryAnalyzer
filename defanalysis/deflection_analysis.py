import os
from .io import ensure_dir, load_tracks_csv, save_summary_csv
from .track_stats import compute_track_stats, extract_slopes, save_track_parameters
from .gating import compute_gate, classify_tracks, percentile_clip
from .plots import plot_histogram, plot_classified_trajectories
from .report import summarize_results, print_report
from .plot_expopara import plot_exponential_parameter_histograms
import numpy as np
import pandas as pd



def DefAnalysis_Dynamic(
    ExperimentFile,
    ControlFile,
    OutputFolder,
    CntrlOutputFolder,
    min_track_length=10,
    sensitivity=99.9,
    bin_count=500,
    correct_baseline_drift=False, # This parameter is to select baseline experiment or control. Default is experiment so for control it false
    max_traj_lines=1000,
    fit_exponential_exp=True,
    exp_min_x_span_px=30,
    exp_r2_min_for_hists=None,
    exp_hist_bins=500
):
    ensure_dir(OutputFolder)
    ensure_dir(CntrlOutputFolder)
    print(f"Processing... (Drift Correction: {correct_baseline_drift})")

    # Load
    df_ctrl = load_tracks_csv(ControlFile)
    df_exp = load_tracks_csv(ExperimentFile)

    # Track stats
    ctrl_stats = compute_track_stats(df_ctrl, min_track_length=min_track_length, fit_exponential=fit_exponential_exp)
    exp_stats  = compute_track_stats(
    df_exp,
    min_track_length=min_track_length,
    fit_exponential=fit_exponential_exp,
    min_x_span_px=exp_min_x_span_px
        )

    if not ctrl_stats or not exp_stats:
        print("Error: insufficient data.")
        return None

    # Save parameter tables
    save_track_parameters(
        ctrl_stats,
        os.path.join(CntrlOutputFolder, "track_parameters_control.csv")
        )

    save_track_parameters(
        exp_stats,
        os.path.join(OutputFolder, "track_parameters_experiment.csv")
        )


    ctrl_slopes = extract_slopes(ctrl_stats)
    exp_slopes  = extract_slopes(exp_stats)

    # Gate
    gate_threshold, metrics = compute_gate(
        ctrl_slopes, exp_slopes,
        sensitivity=sensitivity,
        correct_baseline_drift=correct_baseline_drift
    )

    # Classify + save summary CSV
    final_rows = classify_tracks(exp_stats, gate_threshold)
    summary_csv = os.path.join(OutputFolder, "deflection_summary_dynamic.csv")
    save_summary_csv(final_rows, summary_csv)

    # Plots
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
    if fit_exponential_exp:
        # ---- NEW: decay-rate gating on 'a' (exponential rate constant) ----
        ctrl_a = np.array([t["a"] for t in ctrl_stats if t.get("exp_ok", False) and t.get("a") is not None], dtype=float)
        exp_a  = np.array([t["a"] for t in exp_stats  if t.get("exp_ok", False) and t.get("a") is not None], dtype=float)

        # remove non-finite
        ctrl_a = ctrl_a[np.isfinite(ctrl_a)]
        exp_a  = exp_a[np.isfinite(exp_a)]
        # Percentile clip for outlier removal, we can remove this and go back to ctrl_a
        ctrl_a_plot = percentile_clip(ctrl_a, low=0.1, high=99.9)
        exp_a_plot  = percentile_clip(exp_a,  low=0.1, high=99.9)
        if len(ctrl_a) > 0 and len(exp_a) > 0:
            gate_a, metrics_a = compute_gate(
        ctrl_a_plot, exp_a_plot,
        sensitivity=sensitivity,
        correct_baseline_drift=correct_baseline_drift
            )

            plot_histogram(
        ctrl_a_plot, exp_a_plot,
        gate_a, metrics_a,
        out_path=os.path.join(OutputFolder, "Histogram_a_Gate.png"),
        bin_count=150,
        correct_baseline_drift=correct_baseline_drift,
        xlabel="Decay rate a (1/pixel)",
        title=f"Decay-rate gating: {metrics_a['mode_name']}", use_logx=False, 
                   eps=1e-12, robust_range=True, prc=(0.5, 99.5)
        )

    # optional classification table for decay-rate gate:
            rows_a = classify_tracks(exp_stats, gate_a, feature_key="a")
            pd.DataFrame(rows_a).to_csv(os.path.join(OutputFolder, "deflection_summary_a_gate.csv"), index=False)

        else:
            print("[WARN] Not enough valid exp fits to compute a-gate histogram.")

        # Plotting the Experiment exponential fits
        plot_exponential_parameter_histograms(
        exp_stats,
        out_folder=OutputFolder,
        bins=exp_hist_bins,
        r2_min=exp_r2_min_for_hists
    )
        #Plotting the control fits
        plot_exponential_parameter_histograms(
        ctrl_stats,
        out_folder=CntrlOutputFolder,
        bins=exp_hist_bins,
        r2_min=exp_r2_min_for_hists
    )

    # Report
    summary = summarize_results(ctrl_slopes, exp_slopes, final_rows, metrics,
                           ctrl_a=ctrl_a_plot, exp_a=exp_a_plot,
                           final_rows_a=rows_a, metrics_a=metrics_a)
    print_report(summary, OutputFolder)

    return {
        "summary_csv": summary_csv,
        "gate_threshold": gate_threshold,
        "metrics": metrics,
        "summary": summary,
        "plots": {
            "histogram": os.path.join(OutputFolder, "Histogram_Analysis_StaticGate.png"),
            "trajectories": os.path.join(OutputFolder, "Trajectories_Classified.png"),
            "histogram_a": os.path.join(OutputFolder, "Histogram_a_Gate.png") if fit_exponential_exp else "",
        }
    }
