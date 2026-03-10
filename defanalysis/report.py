import numpy as np
import os
import csv
from datetime import datetime


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

def _gate_summary(ctrl_vals, exp_vals, rows, metrics):
    """
    ctrl_vals/exp_vals are 1D arrays (already clipped/filtered is ok).
    rows is classify_tracks output for that feature.
    metrics is compute_gate output.
    """
    ctrl_stats = _safe_stats(ctrl_vals)
    exp_stats = _safe_stats(exp_vals)

    n_total = len(rows) if rows is not None else 0
    n_defl = int(np.sum([r.get("is_deflected", False) for r in (rows or [])]))
    pct = (n_defl / n_total) * 100 if n_total else 0.0

    return {
        "mode_name": metrics.get("mode_name", "") if metrics else "",
        "gate_threshold": float(metrics.get("gate_threshold", np.nan)) if metrics else np.nan,
        "noise_width": float(metrics.get("noise_width", np.nan)) if metrics else np.nan,

        "ctrl_n": ctrl_stats["n"],
        "exp_n": exp_stats["n"],

        "ctrl_mean": ctrl_stats["mean"],
        "ctrl_std": ctrl_stats["std"],
        "ctrl_median": ctrl_stats["median"],

        "exp_mean": exp_stats["mean"],
        "exp_std": exp_stats["std"],
        "exp_median": exp_stats["median"],

        "num_total": int(n_total),
        "num_deflected": int(n_defl),
        "percent_deflected": float(pct),
    }

def summarize_results(
    ctrl_slopes, exp_slopes, final_rows, metrics,
    ctrl_a=None, exp_a=None, final_rows_a=None, metrics_a=None,
    feature_gates=None,   # NEW: dict of named gates
):
    """
    Always returns slope-gating summary.
    Backward-compatible: still supports a_gate via ctrl_a/exp_a/metrics_a.
    NEW: feature_gates lets you add any number of other gates (quad, tau, y_inf, etc.)

    feature_gates format:
      {
        "quad_a2": {"ctrl": ctrl_arr, "exp": exp_arr, "rows": rows, "metrics": metrics},
        "tau_px":  {"ctrl": ...,     "exp": ...,     "rows": ...,  "metrics": ...},
      }
    """

    # --- Slope gating summary ---
    num_total = len(final_rows)
    num_deflected = int(np.sum([r.get("is_deflected", False) for r in final_rows]))
    percent_deflected = (num_deflected / num_total) * 100 if num_total else 0.0

    exp_s = _safe_stats(exp_slopes)
    ctrl_s = _safe_stats(ctrl_slopes)

    out = {
        "mode_name": metrics.get("mode_name", "") if metrics else "",
        "gate_threshold": float(metrics.get("gate_threshold", np.nan)) if metrics else np.nan,
        "noise_width": float(metrics.get("noise_width", np.nan)) if metrics else np.nan,

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

    # --- Backward-compatible: a-gating summary ---
    if (ctrl_a is not None) and (exp_a is not None) and (metrics_a is not None):
        out["a_gate"] = _gate_summary(ctrl_a, exp_a, final_rows_a, metrics_a)

    # --- NEW: any number of additional feature gates ---
    if feature_gates:
        out["feature_gates"] = {}
        for name, payload in feature_gates.items():
            ctrl_vals = payload.get("ctrl", None)
            exp_vals  = payload.get("exp", None)
            rows      = payload.get("rows", None)
            met       = payload.get("metrics", None)

            if ctrl_vals is None or exp_vals is None or met is None:
                continue

            out["feature_gates"][name] = _gate_summary(ctrl_vals, exp_vals, rows, met)

    return out



def _fmt(x, nd=6):
    try:
        if x is None:
            return "nan"
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "nan"
        return f"{float(x):.{nd}g}"
    except Exception:
        return str(x)


def _build_report_text(summary, output_folder):
    lines = []

    lines.append("")
    lines.append("=" * 40)
    lines.append("📊 FINAL STATISTICS REPORT")
    lines.append("=" * 40)

    lines.append(f"⚙️  Mode: {summary.get('mode_name','')}")
    lines.append("-" * 40)
    lines.append(f"✅ Total particles: {summary['num_total']}")
    lines.append(f"❌ Deflected particles (> {summary['gate_threshold']:.4f}): {summary['num_deflected']}")
    lines.append(f"📈 Percent deflected: {summary['percent_deflected']:.3f}%")
    lines.append("-" * 40)
    lines.append(f"Experiment Mean Slope: {summary['exp_mean']:.5f}")
    lines.append(f"Experiment STD Slope:  {summary['exp_std']:.5f}")
    lines.append(f"Experiment Median:     {summary['exp_median']:.5f}")
    lines.append("-" * 40)
    lines.append(f"Control Mean Slope:    {summary['ctrl_mean']:.5f}")
    lines.append(f"Control STD Slope:     {summary['ctrl_std']:.5f}")
    lines.append(f"Control Median:        {summary['ctrl_median']:.5f}")
    lines.append(f"Control Noise Width:   {summary['noise_width']:.5f}")

    # # Backward-compatible exponential a-gate block
    # if "a_gate" in summary:
    #     a = summary["a_gate"]
    #     lines.append("")
    #     lines.append("-" * 40)
    #     lines.append("📉 EXPONENTIAL DECAY-RATE (a) GATING")
    #     lines.append("-" * 40)
    #     lines.append(f"⚙️  Mode: {a.get('mode_name','')}")
    #     lines.append(f"✅ Tracks with valid exp fit (CTRL/EXP): {a['ctrl_n']} / {a['exp_n']}")
    #     lines.append(
    #         f"❌ Deflected by a-gate (> {_fmt(a['gate_threshold'])}): "
    #         f"{a['num_deflected']}/{a['num_total']} ({a['percent_deflected']:.3f}%)"
    #     )
    #     lines.append("-" * 40)
    #     lines.append(f"Experiment a Mean:   {_fmt(a['exp_mean'])}")
    #     lines.append(f"Experiment a STD:    {_fmt(a['exp_std'])}")
    #     lines.append(f"Experiment a Median: {_fmt(a['exp_median'])}")
    #     lines.append("-" * 40)
    #     lines.append(f"Control a Mean:      {_fmt(a['ctrl_mean'])}")
    #     lines.append(f"Control a STD:       {_fmt(a['ctrl_std'])}")
    #     lines.append(f"Control a Median:    {_fmt(a['ctrl_median'])}")
    #     lines.append(f"Control Noise Width: {_fmt(a['noise_width'])}")

    # Additional feature gates
    fg = summary.get("feature_gates", {})
    if isinstance(fg, dict) and len(fg) > 0:
        lines.append("")
        lines.append("-" * 40)
        lines.append("🧩 ADDITIONAL FEATURE GATES")
        lines.append("-" * 40)

        for name, g in fg.items():
            lines.append("")
            lines.append(f"🔸 {name}")
            lines.append(f"  Mode: {g.get('mode_name','')}")
            lines.append(f"  Valid tracks (CTRL/EXP): {g['ctrl_n']} / {g['exp_n']}")
            lines.append(f"  Gate: {_fmt(g['gate_threshold'])} | Noise width: {_fmt(g['noise_width'])}")
            lines.append(f"  Deflected: {g['num_deflected']}/{g['num_total']} ({g['percent_deflected']:.3f}%)")
            lines.append(
                f"  EXP mean/median/std: "
                f"{_fmt(g['exp_mean'])} / {_fmt(g['exp_median'])} / {_fmt(g['exp_std'])}"
            )
            lines.append(
                f"  CTRL mean/median/std: "
                f"{_fmt(g['ctrl_mean'])} / {_fmt(g['ctrl_median'])} / {_fmt(g['ctrl_std'])}"
            )

    lines.append("-" * 40)
    lines.append(f"📂 Results saved to: {output_folder}")
    lines.append("=" * 40)
    lines.append("")

    return "\n".join(lines)


def print_report(summary, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # ========================
    # BUILD REPORT ONCE
    # ========================
    report_text = _build_report_text(summary, output_folder)

    # ========================
    # PRINT TO CONSOLE
    # ========================
    print(report_text)

    # ========================
    # SAVE TEXT VERSION
    # ========================
    text_path = os.path.join(output_folder, "final_statistics_report.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ========================
    # SAVE CSV VERSION
    # ========================
    csv_path = os.path.join(output_folder, "final_statistics_report.csv")

    flat_summary = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, dict):
                    for sub_sub_key, sub_sub_val in sub_val.items():
                        flat_summary[f"{key}_{sub_key}_{sub_sub_key}"] = sub_sub_val
                else:
                    flat_summary[f"{key}_{sub_key}"] = sub_val
        else:
            flat_summary[key] = value

    flat_summary["timestamp"] = datetime.now().isoformat()

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=flat_summary.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_summary)