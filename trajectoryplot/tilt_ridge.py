import os
import numpy as np
import cv2


def estimate_tilt_from_gradient_ridge(
    gray_image_path: str,
    out_dir: str = None,
    roi: str = "bottom",        # "top" or "bottom"
    roi_frac: float = 0.45,
    search_band: tuple = (0.15, 0.70),
    smooth_ksize: int = 21,
    sample_step: int = 2,
    fit_method: int = None,
    return_ridge_points: bool = False
):
    if fit_method is None:
        fit_method = cv2.DIST_HUBER

    if not gray_image_path or not os.path.exists(gray_image_path):
        return (0.0, {"reason": "gray image not found"}) if not return_ridge_points else (0.0, {"reason": "gray image not found"}, None)

    img = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (0.0, {"reason": "failed to read gray image"}) if not return_ridge_points else (0.0, {"reason": "failed to read gray image"}, None)

    H, W = img.shape[:2]

    band_h = int(H * roi_frac)
    if band_h < 10:
        band_h = min(50, H)

    if roi.lower() == "top":
        y0, y1 = 0, min(band_h, H)
    else:
        y0, y1 = max(0, H - band_h), H

    roi_img = img[y0:y1, :]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_img)
    roi_blur = cv2.GaussianBlur(roi_eq, (5, 5), 0)

    gy = cv2.Sobel(roi_blur, cv2.CV_32F, 0, 1, ksize=3)
    abs_gy = np.abs(gy)

    rH, rW = abs_gy.shape[:2]
    y_min = int(rH * search_band[0])
    y_max = int(rH * search_band[1])
    y_min = int(np.clip(y_min, 0, rH - 2))
    y_max = int(np.clip(y_max, y_min + 1, rH - 1))

    band = abs_gy[y_min:y_max, :]

    cols = np.arange(0, rW, sample_step)
    ridge_y_raw = (np.argmax(band[:, cols], axis=0) + y_min).astype(np.int32)
    ridge_x = cols.astype(np.int32)

    if smooth_ksize and smooth_ksize >= 3 and smooth_ksize % 2 == 1:
        ry_u8 = np.clip(ridge_y_raw, 0, 255).astype(np.uint8)
        ridge_y = cv2.medianBlur(ry_u8, smooth_ksize).astype(np.float32)
    else:
        ridge_y = ridge_y_raw.astype(np.float32)

    ridge_pts = np.column_stack([ridge_x.astype(np.float32), ridge_y.astype(np.float32)])
    pts = ridge_pts.reshape(-1, 1, 2).astype(np.float32)

    try:
        vx, vy, x0, y0_line = cv2.fitLine(pts, fit_method, 0, 0.01, 0.01).flatten()
    except Exception as e:
        out = (0.0, {"reason": f"fitLine failed: {e}"})
        return out if not return_ridge_points else (out[0], out[1], None)

    m = (vy / vx) if abs(vx) > 1e-8 else 0.0
    angle_deg = float(np.degrees(np.arctan(m)))

    dbg = {
        "angle_deg": angle_deg,
        "roi": roi,
        "roi_frac": roi_frac,
        "search_band": search_band,
        "sample_step": sample_step,
        "smooth_ksize": smooth_ksize,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_dir, "tilt_roi_raw.png"), roi_img)
        cv2.imwrite(os.path.join(out_dir, "tilt_roi_eq.png"), roi_eq)

        grad_vis = cv2.normalize(abs_gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "tilt_gradmag.png"), grad_vis)

        overlay = cv2.cvtColor(roi_eq, cv2.COLOR_GRAY2BGR)

        step = max(1, int(10 / max(sample_step, 1)))
        for i in range(0, len(ridge_x), step):
            cv2.circle(overlay, (int(ridge_x[i]), int(ridge_y[i])), 1, (0, 255, 255), -1)

        x_left = 0
        y_left = int(y0_line + (x_left - x0) * m)
        x_right = rW - 1
        y_right = int(y0_line + (x_right - x0) * m)
        cv2.line(overlay, (x_left, y_left), (x_right, y_right), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(out_dir, "tilt_overlay.png"), overlay)

    if return_ridge_points:
        return angle_deg, dbg, ridge_pts
    return angle_deg, dbg


def estimate_tilt_midline_from_two_full_rois(
    gray_image_path: str,
    out_dir: str = None,
    top_roi_frac: float = 0.25,
    bottom_roi_frac: float = 0.25,
    top_search_band: tuple = (0.10, 0.90),
    bottom_search_band: tuple = (0.10, 0.90),
    smooth_ksize: int = 21,
    sample_step: int = 2,
):
    if not gray_image_path or not os.path.exists(gray_image_path):
        return 0.0, {"reason": "gray image not found"}

    img = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, {"reason": "failed to read gray image"}

    H, W = img.shape[:2]
    top_h = max(int(H * top_roi_frac), 30)
    bot_h = max(int(H * bottom_roi_frac), 30)

    y0_top, y1_top = 0, min(top_h, H)
    y0_bot, y1_bot = max(0, H - bot_h), H

    ang_t, dbg_t, pts_top_roi = estimate_tilt_from_gradient_ridge(
        gray_image_path=gray_image_path,
        out_dir=None,
        roi="top",
        roi_frac=top_roi_frac,
        search_band=top_search_band,
        smooth_ksize=smooth_ksize,
        sample_step=sample_step,
        return_ridge_points=True
    )
    ang_b, dbg_b, pts_bot_roi = estimate_tilt_from_gradient_ridge(
        gray_image_path=gray_image_path,
        out_dir=None,
        roi="bottom",
        roi_frac=bottom_roi_frac,
        search_band=bottom_search_band,
        smooth_ksize=smooth_ksize,
        sample_step=sample_step,
        return_ridge_points=True
    )

    if pts_top_roi is None or pts_bot_roi is None:
        return 0.0, {"reason": "ridge points missing for top/bottom"}

    pts_top_full = pts_top_roi.copy()
    pts_top_full[:, 1] += float(y0_top)

    pts_bot_full = pts_bot_roi.copy()
    pts_bot_full[:, 1] += float(y0_bot)

    x = pts_top_full[:, 0]
    if pts_top_full.shape[0] == pts_bot_full.shape[0]:
        y_mid = 0.5 * (pts_top_full[:, 1] + pts_bot_full[:, 1])
    else:
        y_bot_i = np.interp(x, pts_bot_full[:, 0], pts_bot_full[:, 1])
        y_mid = 0.5 * (pts_top_full[:, 1] + y_bot_i)

    pts_mid = np.column_stack([x, y_mid]).astype(np.float32).reshape(-1, 1, 2)

    vx, vy, x0, y0_line = cv2.fitLine(pts_mid, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    m = (vy / vx) if abs(vx) > 1e-8 else 0.0
    angle_deg = float(np.degrees(np.arctan(m)))

    dbg = {
        "angle_deg": angle_deg,
        "theta_top": float(ang_t),
        "theta_bottom": float(ang_b),
        "mode": "both_midline_full_rois",
        "top_roi_frac": top_roi_frac,
        "bottom_roi_frac": bottom_roi_frac,
        "top_search_band": top_search_band,
        "bottom_search_band": bottom_search_band,
        "smooth_ksize": smooth_ksize,
        "sample_step": sample_step,
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        roi_bot_raw = img[y0_bot:y1_bot, :]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        roi_bot_eq = clahe.apply(roi_bot_raw)

        cv2.imwrite(os.path.join(out_dir, "tilt_roi_raw.png"), roi_bot_raw)
        cv2.imwrite(os.path.join(out_dir, "tilt_roi_eq.png"), roi_bot_eq)

        roi_blur = cv2.GaussianBlur(roi_bot_eq, (5, 5), 0)
        gy = cv2.Sobel(roi_blur, cv2.CV_32F, 0, 1, ksize=3)
        abs_gy = np.abs(gy)
        grad_vis = cv2.normalize(abs_gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "tilt_gradmag.png"), grad_vis)

        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        step = max(1, int(10 / max(sample_step, 1)))
        for pts, color in [(pts_top_full, (0, 255, 255)), (pts_bot_full, (255, 255, 0))]:
            for i in range(0, pts.shape[0], step):
                cv2.circle(overlay, (int(pts[i, 0]), int(pts[i, 1])), 1, color, -1)

        pts_mid_xy = np.column_stack([x, y_mid])
        for i in range(0, pts_mid_xy.shape[0], step):
            cv2.circle(overlay, (int(pts_mid_xy[i, 0]), int(pts_mid_xy[i, 1])), 1, (0, 0, 255), -1)

        xL, xR = 0, W - 1
        yL = int(y0_line + (xL - x0) * m)
        yR = int(y0_line + (xR - x0) * m)
        cv2.line(overlay, (xL, yL), (xR, yR), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(out_dir, "tilt_overlay.png"), overlay)

    return angle_deg, dbg


def estimate_tilt_mode(
    gray_image_path: str,
    out_dir: str = None,
    roi: str = "bottom",         # "top", "bottom", or "both"
    roi_frac: float = 0.5,
    top_search_band: tuple = (0, 0.5),
    bottom_search_band: tuple = (0.5, 1.0),
    mode: str = "bottom",        # "top", "bottom", "both_avg", "both_midline"
    smooth_ksize: int = 21,
    sample_step: int = 2,
    max_angle_diff_deg: float = 2.0
):
    if roi.lower() == "both" and mode == "both_midline":
        return estimate_tilt_midline_from_two_full_rois(
            gray_image_path=gray_image_path,
            out_dir=out_dir,
            top_roi_frac=0.25,
            bottom_roi_frac=0.25,
            top_search_band=(0.10, 0.90),
            bottom_search_band=(0.10, 0.90),
            smooth_ksize=smooth_ksize,
            sample_step=sample_step
        )

    if mode == "top":
        ang, dbg = estimate_tilt_from_gradient_ridge(
            gray_image_path=gray_image_path,
            out_dir=out_dir,
            roi=roi,
            roi_frac=roi_frac,
            search_band=top_search_band,
            smooth_ksize=smooth_ksize,
            sample_step=sample_step
        )
        dbg["mode"] = "top"
        return float(ang), dbg

    if mode == "bottom":
        ang, dbg = estimate_tilt_from_gradient_ridge(
            gray_image_path=gray_image_path,
            out_dir=out_dir,
            roi=roi,
            roi_frac=roi_frac,
            search_band=bottom_search_band,
            smooth_ksize=smooth_ksize,
            sample_step=sample_step
        )
        dbg["mode"] = "bottom"
        return float(ang), dbg

    ang_top, dbg_top, pts_top = estimate_tilt_from_gradient_ridge(
        gray_image_path=gray_image_path,
        out_dir=None,
        roi=roi,
        roi_frac=roi_frac,
        search_band=top_search_band,
        smooth_ksize=smooth_ksize,
        sample_step=sample_step,
        return_ridge_points=True
    )
    ang_bot, dbg_bot, pts_bot = estimate_tilt_from_gradient_ridge(
        gray_image_path=gray_image_path,
        out_dir=None,
        roi=roi,
        roi_frac=roi_frac,
        search_band=bottom_search_band,
        smooth_ksize=smooth_ksize,
        sample_step=sample_step,
        return_ridge_points=True
    )

    ang_top = float(ang_top)
    ang_bot = float(ang_bot)
    diff = abs(ang_top - ang_bot)

    dbg = {
        "mode": mode,
        "theta_top": ang_top,
        "theta_bottom": ang_bot,
        "angle_diff_deg": diff,
        "roi": roi,
        "roi_frac": roi_frac,
        "top_search_band": top_search_band,
        "bottom_search_band": bottom_search_band,
        "smooth_ksize": smooth_ksize,
        "sample_step": sample_step
    }

    if mode == "both_avg":
        if diff > max_angle_diff_deg:
            chosen = ang_top if abs(ang_top) < abs(ang_bot) else ang_bot
            dbg["fallback_used"] = True
            dbg["angle_deg"] = chosen
            return float(chosen), dbg

        avg = 0.5 * (ang_top + ang_bot)
        dbg["fallback_used"] = False
        dbg["angle_deg"] = avg
        return float(avg), dbg

    if mode == "both_midline":
        if pts_top is None or pts_bot is None:
            avg = 0.5 * (ang_top + ang_bot)
            dbg["note"] = "missing ridge points; fell back to angle average"
            dbg["angle_deg"] = avg
            return float(avg), dbg

        x = pts_top[:, 0]
        if pts_top.shape[0] == pts_bot.shape[0]:
            y_mid = 0.5 * (pts_top[:, 1] + pts_bot[:, 1])
        else:
            y_bot_i = np.interp(x, pts_bot[:, 0], pts_bot[:, 1])
            y_mid = 0.5 * (pts_top[:, 1] + y_bot_i)

        pts_mid = np.column_stack([x, y_mid]).astype(np.float32).reshape(-1, 1, 2)
        vx, vy, x0, y0 = cv2.fitLine(pts_mid, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        m = (vy / vx) if abs(vx) > 1e-8 else 0.0
        theta_mid = float(np.degrees(np.arctan(m)))

        dbg["angle_deg"] = theta_mid
        dbg["theta_midline"] = theta_mid
        dbg["fallback_used"] = False
        return float(theta_mid), dbg

    # default
    ang, dbg2 = estimate_tilt_from_gradient_ridge(
        gray_image_path=gray_image_path,
        out_dir=out_dir,
        roi=roi if roi.lower() != "both" else "bottom",
        roi_frac=roi_frac,
        search_band=bottom_search_band,
        smooth_ksize=smooth_ksize,
        sample_step=sample_step
    )
    dbg2["mode"] = "bottom (default)"
    return float(ang), dbg2
