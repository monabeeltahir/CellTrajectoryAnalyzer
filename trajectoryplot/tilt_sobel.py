import os
import numpy as np
import cv2


def estimate_tilt_from_sobel_edges(
    gray_image_path: str,
    out_dir: str = None,
    roi: str = "bottom",            # "top" or "bottom"
    roi_frac: float = 0.25,
    search_band: tuple = (0.10, 0.90),  # inside ROI
    polarity: str = "neg",          # "pos", "neg", or "abs"
    blur_ksize: int = 5,
    thr_percentile: float = 92.0,
    close_ksize: int = 7,
    open_ksize: int = 3,
    min_component_area: int = 200,
    min_pts: int = 200,
    fit_method: int = None,
    return_points: bool = False,    # NEW
    debug_prefix: str = ""          # optional: to avoid overwriting if you ever want
):
    """
    Sobel vertical-gradient -> threshold -> morph clean/connect -> robust line fit.

    Saves debug (same filenames, unless debug_prefix provided):
      tilt_roi_raw.png, tilt_roi_eq.png, tilt_gradmag.png, tilt_edges.png, tilt_overlay.png

    Returns:
      - angle_deg, dbg
      - optionally: (angle_deg, dbg, pts_full_image) if return_points=True
    """
    if fit_method is None:
        fit_method = cv2.DIST_HUBER

    if not gray_image_path or not os.path.exists(gray_image_path):
        out = (0.0, {"reason": "gray image not found", "n_pts": 0, "roi": roi})
        return out if not return_points else (out[0], out[1], None)

    img = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        out = (0.0, {"reason": "failed to read gray image", "n_pts": 0, "roi": roi})
        return out if not return_points else (out[0], out[1], None)

    H, W = img.shape[:2]

    # --- ROI crop (strip) ---
    band_h = max(int(H * roi_frac), 30)
    if roi.lower() == "top":
        y0, y1 = 0, min(band_h, H)
    else:
        y0, y1 = max(0, H - band_h), H

    roi_img = img[y0:y1, :]

    # --- Contrast normalize ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_img)

    # --- Blur ---
    if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
        roi_blur = cv2.GaussianBlur(roi_eq, (blur_ksize, blur_ksize), 0)
    else:
        roi_blur = roi_eq

    # --- Sobel vertical gradient ---
    gy = cv2.Sobel(roi_blur, cv2.CV_32F, 0, 1, ksize=3)

    # --- Restrict y band inside ROI ---
    rH, rW = gy.shape[:2]
    y_min = int(np.clip(int(rH * search_band[0]), 0, rH - 2))
    y_max = int(np.clip(int(rH * search_band[1]), y_min + 1, rH - 1))
    gy_band = gy[y_min:y_max, :]

    # --- Polarity selection ---
    if polarity == "pos":
        resp = gy_band
    elif polarity == "neg":
        resp = -gy_band
    else:
        resp = np.abs(gy_band)

    # --- Threshold by percentile ---
    thr = float(np.percentile(resp, thr_percentile))
    edge_mask_band = (resp >= thr).astype(np.uint8) * 255

    # --- Morph: open then close ---
    if open_ksize and open_ksize >= 3:
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        edge_mask_band = cv2.morphologyEx(edge_mask_band, cv2.MORPH_OPEN, k_open)

    if close_ksize and close_ksize >= 3:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        edge_mask_band = cv2.morphologyEx(edge_mask_band, cv2.MORPH_CLOSE, k_close)

    # --- Remove tiny components ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_mask_band, connectivity=8)
    cleaned = np.zeros_like(edge_mask_band)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_component_area):
            cleaned[labels == i] = 255
    edge_mask_band = cleaned

    # --- Edge pixels in ROI coords ---
    ys_local, xs = np.where(edge_mask_band > 0)
    n_pts = int(len(xs))

    dbg = {
        "roi": roi,
        "roi_frac": roi_frac,
        "search_band": search_band,
        "polarity": polarity,
        "thr_percentile": thr_percentile,
        "n_pts": n_pts,
        "min_pts": int(min_pts),
        "open_ksize": open_ksize,
        "close_ksize": close_ksize,
        "min_component_area": int(min_component_area),
    }

    # Convert to full-image coordinates for returning/combining
    pts_full = None
    if n_pts > 0:
        ys_roi = ys_local + y_min
        ys_full = ys_roi + y0
        xs_full = xs
        pts_full = np.column_stack([xs_full.astype(np.float32), ys_full.astype(np.float32)])

    # Not enough points -> optionally still save debug, then fail
    if n_pts < min_pts:
        dbg["reason"] = "too few edge pixels"

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_roi_raw.png"), roi_img)
            cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_roi_eq.png"), roi_eq)

            grad_vis = cv2.normalize(np.abs(gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_gradmag.png"), grad_vis)

            edges_full_roi = np.zeros((rH, rW), dtype=np.uint8)
            edges_full_roi[y_min:y_max, :] = edge_mask_band
            cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_edges.png"), edges_full_roi)

            overlay = cv2.cvtColor(roi_eq, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_overlay.png"), overlay)

        out = (0.0, dbg)
        return out if not return_points else (out[0], out[1], pts_full)

    # --- Fit line using ROI coordinates (for debug drawing), but compute angle from that fit
    ys_roi = (ys_local + y_min).astype(np.float32)
    xs_roi = xs.astype(np.float32)
    pts_roi = np.column_stack([xs_roi, ys_roi]).reshape(-1, 1, 2)

    vx, vy, x0_line, y0_line = cv2.fitLine(pts_roi, fit_method, 0, 0.01, 0.01).flatten()
    m = (vy / vx) if abs(vx) > 1e-8 else 0.0
    angle_deg = float(np.degrees(np.arctan(m)))
    dbg["angle_deg"] = angle_deg
    dbg["reason"] = "ok"

    # --- Debug saves (single ROI view) ---
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_roi_raw.png"), roi_img)
        cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_roi_eq.png"), roi_eq)

        grad_vis = cv2.normalize(np.abs(gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_gradmag.png"), grad_vis)

        edges_full_roi = np.zeros((rH, rW), dtype=np.uint8)
        edges_full_roi[y_min:y_max, :] = edge_mask_band
        cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_edges.png"), edges_full_roi)

        overlay = cv2.cvtColor(roi_eq, cv2.COLOR_GRAY2BGR)

        # draw a subsample of edge pixels
        step = max(1, n_pts // 1500)
        for i in range(0, n_pts, step):
            cv2.circle(overlay, (int(xs_roi[i]), int(ys_roi[i])), 1, (0, 255, 255), -1)

        # fitted line across ROI width
        xL, xR = 0, rW - 1
        yL = int(y0_line + (xL - x0_line) * m)
        yR = int(y0_line + (xR - x0_line) * m)
        cv2.line(overlay, (xL, yL), (xR, yR), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(out_dir, f"{debug_prefix}tilt_overlay.png"), overlay)

    return (angle_deg, dbg) if not return_points else (angle_deg, dbg, pts_full)


def estimate_tilt_from_sobel_auto(
    gray_image_path: str,
    out_dir: str = None,
    top_roi_frac: float = 0.25,
    bottom_roi_frac: float = 0.25,
    top_search_band: tuple = (0.10, 0.90),
    bottom_search_band: tuple = (0.10, 0.90),
    polarity_top: str = "pos",
    polarity_bottom: str = "neg",
    thr_percentile: float = 92.0,
    blur_ksize: int = 5,
    open_ksize: int = 3,
    close_ksize: int = 7,
    min_component_area: int = 200,
    min_pts: int = 200,
    fit_method: int = None
):
    """
    Tries BOTH top and bottom Sobel edges and handles missing cases:

    - If both valid: fit ONE robust line to combined edge points (FULL IMAGE coords)
    - If only one valid: use that one
    - If none valid: return 0.0

    Debug output:
      - If both valid: saves FULL-image overlay in tilt_overlay.png (and a combined edges mask in tilt_edges.png)
      - If only one valid: uses that ROI’s debug saves (same filenames)
    """
    if fit_method is None:
        fit_method = cv2.DIST_HUBER

    if not gray_image_path or not os.path.exists(gray_image_path):
        return 0.0, {"reason": "gray image not found", "mode": "sobel_auto"}

    img = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, {"reason": "failed to read gray image", "mode": "sobel_auto"}

    H, W = img.shape[:2]

    # Run TOP without writing debug (avoid overwriting)
    ang_t, dbg_t, pts_t = estimate_tilt_from_sobel_edges(
        gray_image_path=gray_image_path,
        out_dir=None,
        roi="top",
        roi_frac=top_roi_frac,
        search_band=top_search_band,
        polarity=polarity_top,
        blur_ksize=blur_ksize,
        thr_percentile=thr_percentile,
        open_ksize=open_ksize,
        close_ksize=close_ksize,
        min_component_area=min_component_area,
        min_pts=min_pts,
        fit_method=fit_method,
        return_points=True
    )

    # Run BOTTOM without writing debug
    ang_b, dbg_b, pts_b = estimate_tilt_from_sobel_edges(
        gray_image_path=gray_image_path,
        out_dir=None,
        roi="bottom",
        roi_frac=bottom_roi_frac,
        search_band=bottom_search_band,
        polarity=polarity_bottom,
        blur_ksize=blur_ksize,
        thr_percentile=thr_percentile,
        open_ksize=open_ksize,
        close_ksize=close_ksize,
        min_component_area=min_component_area,
        min_pts=min_pts,
        fit_method=fit_method,
        return_points=True
    )

    ok_t = (dbg_t.get("reason") == "ok")
    ok_b = (dbg_b.get("reason") == "ok")

    dbg = {
        "mode": "sobel_auto",
        "top_ok": bool(ok_t),
        "bottom_ok": bool(ok_b),
        "top": dbg_t,
        "bottom": dbg_b,
    }

    # --- Case 1: none valid ---
    if (not ok_t) and (not ok_b):
        dbg["reason"] = "no valid top/bottom edge"
        # Save something useful for debugging: full image overlay with nothing
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(out_dir, "tilt_overlay.png"), overlay)
            cv2.imwrite(os.path.join(out_dir, "tilt_edges.png"), np.zeros((H, W), dtype=np.uint8))
            # also save the ROIs for quick inspection (keeping your filenames)
            # choose bottom ROI raw/eq for consistency
            band_h = max(int(H * bottom_roi_frac), 30)
            y0, y1 = max(0, H - band_h), H
            roi_img = img[y0:y1, :]
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            roi_eq = clahe.apply(roi_img)
            cv2.imwrite(os.path.join(out_dir, "tilt_roi_raw.png"), roi_img)
            cv2.imwrite(os.path.join(out_dir, "tilt_roi_eq.png"), roi_eq)
        return 0.0, dbg

    # --- Case 2: only one valid -> write that ROI debug and return its angle ---
    if ok_t and (not ok_b):
        dbg["reason"] = "used_top_only"
        if out_dir:
            _ = estimate_tilt_from_sobel_edges(
                gray_image_path=gray_image_path,
                out_dir=out_dir,
                roi="top",
                roi_frac=top_roi_frac,
                search_band=top_search_band,
                polarity=polarity_top,
                blur_ksize=blur_ksize,
                thr_percentile=thr_percentile,
                open_ksize=open_ksize,
                close_ksize=close_ksize,
                min_component_area=min_component_area,
                min_pts=min_pts,
                fit_method=fit_method,
                return_points=False
            )
        dbg["angle_deg"] = float(ang_t)
        return float(ang_t), dbg

    if ok_b and (not ok_t):
        dbg["reason"] = "used_bottom_only"
        if out_dir:
            _ = estimate_tilt_from_sobel_edges(
                gray_image_path=gray_image_path,
                out_dir=out_dir,
                roi="bottom",
                roi_frac=bottom_roi_frac,
                search_band=bottom_search_band,
                polarity=polarity_bottom,
                blur_ksize=blur_ksize,
                thr_percentile=thr_percentile,
                open_ksize=open_ksize,
                close_ksize=close_ksize,
                min_component_area=min_component_area,
                min_pts=min_pts,
                fit_method=fit_method,
                return_points=False
            )
        dbg["angle_deg"] = float(ang_b)
        return float(ang_b), dbg

    # --- Case 3: both valid -> combine FULL-image points and fit ONE line ---
    pts_all = np.vstack([pts_t, pts_b]).astype(np.float32)  # (N,2) full image coords
    pts_fit = pts_all.reshape(-1, 1, 2)

    vx, vy, x0, y0 = cv2.fitLine(pts_fit, fit_method, 0, 0.01, 0.01).flatten()
    m = (vy / vx) if abs(vx) > 1e-8 else 0.0
    angle_deg = float(np.degrees(np.arctan(m)))

    dbg["reason"] = "used_top_and_bottom_combined"
    dbg["angle_deg"] = angle_deg
    dbg["n_pts_combined"] = int(pts_all.shape[0])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        # Save ROI debug images using bottom ROI (keeps your existing convention)
        band_h = max(int(H * bottom_roi_frac), 30)
        y0b, y1b = max(0, H - band_h), H
        roi_bot = img[y0b:y1b, :]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        roi_eq = clahe.apply(roi_bot)
        cv2.imwrite(os.path.join(out_dir, "tilt_roi_raw.png"), roi_bot)
        cv2.imwrite(os.path.join(out_dir, "tilt_roi_eq.png"), roi_eq)

        # Save a combined edge mask for visualization (not perfect “edges”, but where points are)
        edges = np.zeros((H, W), dtype=np.uint8)
        xs = np.clip(pts_all[:, 0].astype(np.int32), 0, W - 1)
        ys = np.clip(pts_all[:, 1].astype(np.int32), 0, H - 1)
        edges[ys, xs] = 255
        # thicken for visibility
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cv2.imwrite(os.path.join(out_dir, "tilt_edges.png"), edges)

        # Overlay on FULL image: points + fitted line
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        step = max(1, int(len(xs) // 2000))
        for i in range(0, len(xs), step):
            cv2.circle(overlay, (int(xs[i]), int(ys[i])), 1, (0, 255, 255), -1)

        xL, xR = 0, W - 1
        yL = int(y0 + (xL - x0) * m)
        yR = int(y0 + (xR - x0) * m)
        cv2.line(overlay, (xL, yL), (xR, yR), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(out_dir, "tilt_overlay.png"), overlay)

        # (optional) gradmag: save a generic one from full image for consistency
        gy_full = cv2.Sobel(cv2.GaussianBlur(img, (5, 5), 0), cv2.CV_32F, 0, 1, ksize=3)
        grad_vis = cv2.normalize(np.abs(gy_full), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "tilt_gradmag.png"), grad_vis)

    return angle_deg, dbg
