import numpy as np

def _safe_stats(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "p99": np.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99.0)),
    }

def summarize_results(
    ctrl_slopes, exp_slopes, final_rows, metrics,
    ctrl_a=None, exp_a=None, final_rows_a=None, metrics_a=None
):
    """
    Summary for slope-gating always.
    Optional: add a-gating summary if ctrl_a/exp_a and metrics_a are provided.
    """

    # --- Slope gating summary (existing) ---
    num_total = len(final_rows)
    num_deflected = int(np.sum([r.get("is_deflected", False) for r in final_rows]))
    percent_deflected = (num_deflected / num_total) * 100 if num_total else 0.0

    exp_s = _safe_stats(exp_slopes)
    ctrl_s = _safe_stats(ctrl_slopes)

    out = {
        "mode_name": metrics.get("mode_name", ""),
        "gate_threshold": float(metrics.get("gate_threshold", np.nan)),
        "noise_width": float(metrics.get("noise_width", np.nan)),

        "num_total": int(num_total),
        "num_deflected": int(num_deflected),
        "percent_deflected": float(percent_deflected),

        "exp_mean": exp_s["mean"],
        "exp_std": exp_s["std"],
        "exp_median": exp_s["median"],

        "ctrl_mean": ctrl_s["mean"],
        "ctrl_std": ctrl_s["std"],
        "ctrl_median": ctrl_s["median"],

        "ctrl_n": ctrl_s["n"],
        "exp_n": exp_s["n"],
    }

    # --- Optional: a-gating summary ---
    if (ctrl_a is not None) and (exp_a is not None) and (metrics_a is not None):
        exp_a_stats = _safe_stats(exp_a)
        ctrl_a_stats = _safe_stats(ctrl_a)

        if final_rows_a is not None:
            n_total_a = len(final_rows_a)
            n_defl_a = int(np.sum([r.get("is_deflected", False) for r in final_rows_a]))
            pct_defl_a = (n_defl_a / n_total_a) * 100 if n_total_a else 0.0
        else:
            n_total_a = 0
            n_defl_a = 0
            pct_defl_a = 0.0

        out["a_gate"] = {
            "mode_name": metrics_a.get("mode_name", ""),
            "gate_threshold": float(metrics_a.get("gate_threshold", np.nan)),
            "noise_width": float(metrics_a.get("noise_width", np.nan)),
            "ctrl_n": ctrl_a_stats["n"],
            "exp_n": exp_a_stats["n"],

            "ctrl_mean": ctrl_a_stats["mean"],
            "ctrl_std": ctrl_a_stats["std"],
            "ctrl_median": ctrl_a_stats["median"],

            "exp_mean": exp_a_stats["mean"],
            "exp_std": exp_a_stats["std"],
            "exp_median": exp_a_stats["median"],

            "num_total": int(n_total_a),
            "num_deflected": int(n_defl_a),
            "percent_deflected": float(pct_defl_a),
        }

    return out


def print_report(summary, output_folder):
    print("\n" + "=" * 40)
    print("📊 FINAL STATISTICS REPORT")
    print("=" * 40)

    # --- Slope report (existing) ---
    print(f"⚙️  Mode: {summary.get('mode_name','')}")
    print("-" * 40)
    print(f"✅ Total particles: {summary['num_total']}")
    print(f"❌ Deflected particles (> {summary['gate_threshold']:.4f}): {summary['num_deflected']}")
    print(f"📈 Percent deflected: {summary['percent_deflected']:.3f}%")
    print("-" * 40)
    print(f"Experiment Mean Slope: {summary['exp_mean']:.5f}")
    print(f"Experiment STD Slope:  {summary['exp_std']:.5f}")
    print(f"Experiment Median:     {summary['exp_median']:.5f}")
    print("-" * 40)
    print(f"Control Mean Slope:    {summary['ctrl_mean']:.5f}")
    print(f"Control STD Slope:     {summary['ctrl_std']:.5f}")
    print(f"Control Median:        {summary['ctrl_median']:.5f}")
    print(f"Control Noise Width:   {summary['noise_width']:.5f}")

    # --- Optional a-gate report ---
    if "a_gate" in summary:
        a = summary["a_gate"]
        print("\n" + "-" * 40)
        print("📉 EXPONENTIAL DECAY-RATE (a) GATING")
        print("-" * 40)
        print(f"⚙️  Mode: {a.get('mode_name','')}")
        print(f"✅ Tracks with valid exp fit (CTRL/EXP): {a['ctrl_n']} / {a['exp_n']}")
        print(f"❌ Deflected by a-gate (> {a['gate_threshold']:.6g}): {a['num_deflected']}/{a['num_total']} "
              f"({a['percent_deflected']:.3f}%)")
        print("-" * 40)
        print(f"Experiment a Mean:   {a['exp_mean']:.6g}")
        print(f"Experiment a STD:    {a['exp_std']:.6g}")
        print(f"Experiment a Median: {a['exp_median']:.6g}")
        print("-" * 40)
        print(f"Control a Mean:      {a['ctrl_mean']:.6g}")
        print(f"Control a STD:       {a['ctrl_std']:.6g}")
        print(f"Control a Median:    {a['ctrl_median']:.6g}")
        print(f"Control Noise Width: {a['noise_width']:.6g}")

    print("-" * 40)
    print(f"📂 Results saved to: {output_folder}")
    print("=" * 40 + "\n")
