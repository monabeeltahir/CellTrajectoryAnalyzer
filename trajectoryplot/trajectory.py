import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from .tilt_ridge import estimate_tilt_mode
from .tilt_sobel import estimate_tilt_from_sobel_edges, estimate_tilt_from_sobel_auto
from .geometry import rotate_points
from .tilt_sobel_simple import estimate_tilt_sobel_zonal
from .tilt_scharr_ransac import estimate_tilt_scharr_ransac


def TrajctoryPlot(
    min_frames=100,
    FileName="id_tracking_trajectories.csv",
    FolderName="",
    gray_image_path="",
    do_tilt_correction=True,

    # User-facing knobs (ALL dictated from here)
    tilt_method="sobel_auto",        # "sobel_auto", "sobel", "ridge_mode"
    tilt_roi="bottom",              # for "sobel": "top"/"bottom"; for ridge_mode: "top"/"bottom"/"both"
    tilt_mode="both_midline",        # used only when tilt_method="ridge_mode"

    # Sobel knobs (used by sobel/sobel_auto)
    top_roi_frac=0.25,
    bottom_roi_frac=0.25,
    top_search_band=(0.10, 0.90),
    bottom_search_band=(0.10, 0.90),
    polarity_top="pos",
    polarity_bottom="neg",
    thr_percentile=92.0,
    blur_ksize=5,
    open_ksize=3,
    close_ksize=7,
    min_component_area=200,
    min_pts=200,

    save_corrected_plot=True
):
    """
    Reads trajectories, filters by min_frames, saves filtered CSV,
    optionally estimates channel tilt from gray image and saves corrected CSV.

    tilt_method:
      - "sobel_auto": tries both top+bottom, falls back if one missing
      - "sobel": uses single ROI specified by tilt_roi ("top" or "bottom")
      - "ridge_mode": uses estimate_tilt_mode(...) with tilt_roi + tilt_mode
    """

    if FileName.strip() == "":
        print("Not a valid File Name or Empty")
        return

    out_dir = FolderName if FolderName else "."
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(FileName)
    if "id" not in df.columns or "center_x" not in df.columns or "center_y" not in df.columns:
        raise ValueError("CSV must contain columns: 'id', 'center_x', 'center_y'")

    # === Filter short-lived tracks ===
    id_counts = df["id"].value_counts()
    valid_ids = id_counts[id_counts >= min_frames].index
    df_filtered = df[df["id"].isin(valid_ids)].copy()
    
    filtered_csv_path = os.path.join(out_dir, "filtered_trajectories.csv")
    df_filtered.to_csv(filtered_csv_path, index=False)

    # === Tilt correction ===
    angle_deg = 0.0
    dbg = {}
    df_corrected = None

    if do_tilt_correction and gray_image_path:
        method = (tilt_method or "").strip().lower()
        roi = (tilt_roi or "bottom").strip().lower()

        if method == "sobel_auto":
            angle_deg, dbg = estimate_tilt_from_sobel_auto(
                gray_image_path=gray_image_path,
                out_dir=out_dir,
                top_roi_frac=top_roi_frac,
                bottom_roi_frac=bottom_roi_frac,
                top_search_band=top_search_band,
                bottom_search_band=bottom_search_band,
                polarity_top=polarity_top,
                polarity_bottom=polarity_bottom,
                thr_percentile=thr_percentile,
                blur_ksize=blur_ksize,
                open_ksize=open_ksize,
                close_ksize=close_ksize,
                min_component_area=min_component_area,
                min_pts=min_pts
            )

        elif method == "sobel":
            # single ROI only
            if roi not in ["top", "bottom"]:
                roi = "bottom"

            roi_frac = top_roi_frac if roi == "top" else bottom_roi_frac
            band = top_search_band if roi == "top" else bottom_search_band
            polarity = polarity_top if roi == "top" else polarity_bottom

            angle_deg, dbg = estimate_tilt_from_sobel_edges(
                gray_image_path=gray_image_path,
                out_dir=out_dir,
                roi=roi,
                roi_frac=roi_frac,
                search_band=band,
                polarity=polarity,
                thr_percentile=thr_percentile,
                blur_ksize=blur_ksize,
                open_ksize=open_ksize,
                close_ksize=close_ksize,
                min_component_area=min_component_area,
                min_pts=min_pts
            )

        elif method == "ridge_mode":
            # your ridge dispatcher (uses its own internal defaults unless you extended it)
            angle_deg, dbg = estimate_tilt_mode(
                gray_image_path=gray_image_path,
                out_dir=out_dir,
                roi=roi,          # "top"/"bottom"/"both"
                mode=tilt_mode
            )
        elif method == "scharr_ransac":
           
            angle_deg, dbg = estimate_tilt_scharr_ransac(
                gray_image_path=gray_image_path,
                out_dir=out_dir,
                mode=tilt_roi,           # "top", "bottom", "both"
                band_frac=0.5,
                blur_ksize=5,
                thr_frac=0.35,
                close_kernel=(41, 3),
                ransac_inlier_px=2.5,
                save_debug=True
                )

        elif method =="sobel_zonal":
            angle_deg, dbg = estimate_tilt_sobel_zonal(
                gray_image_path=gray_image_path,
                out_dir=out_dir,
                save_debug=True,        # set False later if you don’t want debug images

                mode=tilt_roi,          # "top", "bottom", or "both"

                top_frac=0.30,
                bottom_frac=0.30,

                blur_ksize=9,
                sobel_ksize=5,
                close_len=30,
                min_pts=20
                )
        

        else:
            angle_deg, dbg = 0.0, {"reason": f"unknown tilt_method '{tilt_method}'"}

        print(f"[Tilt] method={tilt_method} roi={tilt_roi} mode={tilt_mode} angle={angle_deg:.3f} deg | dbg={dbg}")

        # Pivot: image center (preferred)
        img = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
            cx, cy = w / 2.0, h / 2.0
        else:
            cx = float(df_filtered["center_x"].mean())
            cy = float(df_filtered["center_y"].mean())

        df_corrected = df_filtered.copy()
        x_corr, y_corr = rotate_points(
            df_corrected["center_x"].values,
            df_corrected["center_y"].values,
            angle_deg=angle_deg,
            cx=cx,
            cy=cy
        )
        df_corrected["center_x"] = x_corr
        df_corrected["center_y"] = y_corr
        df_corrected["tilt_angle_deg"] = angle_deg

        corrected_csv_path = os.path.join(out_dir, "filtered_trajectories_corrected.csv")
        df_corrected.to_csv(corrected_csv_path, index=False)

        print(f"Saved: {filtered_csv_path}")
        print(f"Saved: {corrected_csv_path}")
    else:
        print(f"Saved: {filtered_csv_path}")

    # === Plot filtered trajectories ===
    plt.figure(figsize=(12, 8))
    for obj_id in sorted(df_filtered["id"].unique()):
        track = df_filtered[df_filtered["id"] == obj_id]
        plt.plot(track["center_x"], track["center_y"],
                 label=f"ID {obj_id}", marker="o", markersize=2, linewidth=1)
    plt.gca().invert_yaxis()
    plt.title(f"Filtered Cell Trajectories by ID (min {min_frames} frames)", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.xlabel("X Position (pixels)", fontsize=18)
    plt.ylabel("Y Position (pixels)", fontsize=18)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "filtered_trajectories_plot.png"), dpi=300)
    plt.close()

    # === Plot corrected trajectories ===
    if df_corrected is not None and save_corrected_plot:
        plt.figure(figsize=(12, 8))
        for obj_id in sorted(df_corrected["id"].unique()):
            track = df_corrected[df_corrected["id"] == obj_id]
            plt.plot(track["center_x"], track["center_y"],
                     label=f"ID {obj_id}", marker="o", markersize=2, linewidth=1)
        plt.gca().invert_yaxis()
        plt.title(f"Corrected Trajectories (tilt removed: {angle_deg:.3f}°)", fontsize=16)
        plt.tick_params(axis="both", labelsize=16)
        plt.xlabel("X Position (pixels) [corrected]", fontsize=18)
        plt.ylabel("Y Position (pixels) [corrected]", fontsize=18)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "filtered_trajectories_corrected_plot.png"), dpi=300)
        plt.close()

    # === Plot original trajectories ===
    plt.figure(figsize=(12, 8))
    for obj_id in sorted(df["id"].unique()):
        track = df[df["id"] == obj_id]
        plt.plot(track["center_x"], track["center_y"],
                 label=f"ID {obj_id}", marker="o", markersize=2, linewidth=1)
    plt.gca().invert_yaxis()
    plt.title("Original Trajectories", fontsize=16)
    plt.tick_params(axis="both", labelsize=16)
    plt.xlabel("X Position (pixels)", fontsize=18)
    plt.ylabel("Y Position (pixels)", fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Original_trajectories_plot.png"), dpi=300)
    plt.close()

    return {
        "filtered_csv": filtered_csv_path,
        "tilt_angle_deg": angle_deg,
        "tilt_debug": dbg,
        "corrected_csv": os.path.join(out_dir, "filtered_trajectories_corrected.csv") if df_corrected is not None else ""
    }
