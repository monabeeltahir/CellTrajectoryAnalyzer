# tilt_scharr_ransac.py
import os
import cv2
import numpy as np


def estimate_tilt_scharr_ransac(
    gray_image_path,
    out_dir="",
    mode="both",              # "top", "bottom", "both"
    band_frac=0.40,
    blur_ksize=5,
    thr_frac=0.35,
    close_kernel=(41, 3),
    close_iters=1,
    ransac_iters=900,
    ransac_inlier_px=2.5,
    ransac_min_inliers=600,
    save_debug=True
):
    """
    Returns:
        tilt_deg, dbg_dict

    Debug images (if save_debug=True) saved in out_dir:
      - scharr_input_gray.png
      - scharr_input_clahe.png
      - top_mag.png, top_bw.png, top_inliers_overlay.png (if top used)
      - bottom_mag.png, bottom_bw.png, bottom_inliers_overlay.png (if bottom used)
      - scharr_ransac_final_overlay.png
    """

    img = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, {"reason": "image_not_found"}

    # Ensure out_dir exists if we want debug outputs
    if save_debug:
        out_dir = out_dir if out_dir else "."
        os.makedirs(out_dir, exist_ok=True)

    # --- your pre-processing (keep as you wrote) ---
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(blurred)

    h, w = img.shape
    bh = int(h * band_frac)

    if save_debug:
        cv2.imwrite(os.path.join(out_dir, "scharr_input_gray.png"), img)
        cv2.imwrite(os.path.join(out_dir, "scharr_input_clahe.png"), gray_clahe)

    def _save_roi_debug(tag, mag_u8, bw, roi_gray, pts, inliers, m_refit, c_refit):
        """
        tag: 'top' or 'bottom'
        roi_gray: ROI grayscale image (uint8) used for Scharr
        pts: Nx2 float32 in ROI coords
        inliers: boolean mask over pts
        m_refit, c_refit: line y = m*x + c in ROI coords
        """
        if not save_debug:
            return

        cv2.imwrite(os.path.join(out_dir, f"{tag}_mag.png"), mag_u8)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_bw.png"), bw)

        overlay = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

        # draw inlier points (green)
        in_pts = pts[inliers].astype(int)
        for x, y in in_pts[::20]:  # decimate for speed
            if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)

        # draw fitted line in ROI (red)
        xA, xB = 0, overlay.shape[1] - 1
        yA = int(m_refit * xA + c_refit)
        yB = int(m_refit * xB + c_refit)
        cv2.line(overlay, (xA, yA), (xB, yB), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(out_dir, f"{tag}_inliers_overlay.png"), overlay)

    def fit_one(y0, y1, seed, tag):
        roi = gray_clahe[y0:y1, :].copy()

        if blur_ksize > 1:
            roi = cv2.GaussianBlur(roi, (blur_ksize, blur_ksize), 0)

        scharr_y = cv2.Scharr(roi, cv2.CV_32F, 0, 1)
        mag = np.abs(scharr_y)
        mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        thr = max(int(thr_frac * mag_u8.max()), 10)
        bw = (mag_u8 >= thr).astype(np.uint8) * 255

        k = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=close_iters)

        ys, xs = np.where(bw > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)

        if pts.shape[0] < ransac_min_inliers:
            if save_debug:
                # still save what we saw for troubleshooting
                cv2.imwrite(os.path.join(out_dir, f"{tag}_mag.png"), mag_u8)
                cv2.imwrite(os.path.join(out_dir, f"{tag}_bw.png"), bw)
            return None

        # --- RANSAC ---
        rng = np.random.default_rng(seed)
        best_inliers = None
        best_count = 0

        for _ in range(ransac_iters):
            i1, i2 = rng.integers(0, pts.shape[0], size=2)
            if i1 == i2:
                continue
            x1, y1p = pts[i1]
            x2, y2p = pts[i2]
            if abs(x2 - x1) < 1e-6:
                continue

            m = (y2p - y1p) / (x2 - x1)
            c = y1p - m * x1

            dist = np.abs(m * pts[:, 0] - pts[:, 1] + c) / np.sqrt(m * m + 1)
            inliers = dist < ransac_inlier_px
            count = int(np.sum(inliers))

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < ransac_min_inliers:
            if save_debug:
                cv2.imwrite(os.path.join(out_dir, f"{tag}_mag.png"), mag_u8)
                cv2.imwrite(os.path.join(out_dir, f"{tag}_bw.png"), bw)
            return None

        # Refit on inliers (least squares)
        xin = pts[best_inliers][:, 0]
        yin = pts[best_inliers][:, 1]
        A = np.vstack([xin, np.ones_like(xin)]).T
        m_refit, c_refit = np.linalg.lstsq(A, yin, rcond=None)[0]

        # save per-ROI debug images
        _save_roi_debug(tag, mag_u8, bw, roi, pts, best_inliers, float(m_refit), float(c_refit))

        # Return in full-image coords: y = m*x + (c_refit + y0)
        return float(m_refit), float(c_refit + y0), int(best_count)

    top_fit = None
    bot_fit = None

    if mode in ["top", "both"]:
        top_fit = fit_one(0, bh, seed=0, tag="top")

    if mode in ["bottom", "both"]:
        bot_fit = fit_one(h - bh, h, seed=1, tag="bottom")

    fits = []
    if top_fit is not None:
        fits.append(("top", top_fit))
    if bot_fit is not None:
        fits.append(("bottom", bot_fit))

    if len(fits) == 0:
        return 0.0, {"reason": "no_valid_fit"}

    # --- Average slope ---
    slopes = [f[1][0] for f in fits]
    avg_slope = float(np.mean(slopes))
    tilt_deg = float(np.degrees(np.arctan(avg_slope)))

    # Intercept for the "average line" (for overlay)
    avg_c = float(np.mean([f[1][1] for f in fits]))

    if save_debug:
        # full image overlay with per-wall and average line
        full_overlay = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)

        # Draw individual lines
        for name, (m, c, _) in fits:
            xA, xB = 0, w - 1
            yA = int(m * xA + c)
            yB = int(m * xB + c)
            color = (0, 255, 255) if name == "top" else (255, 255, 0)  # yellow / cyan
            cv2.line(full_overlay, (xA, yA), (xB, yB), color, 2)

        # Draw average line in red
        xA, xB = 0, w - 1
        yA = int(avg_slope * xA + avg_c)
        yB = int(avg_slope * xB + avg_c)
        cv2.line(full_overlay, (xA, yA), (xB, yB), (0, 0, 255), 2)

        # Put tilt text
        cv2.putText(
            full_overlay,
            f"tilt={tilt_deg:.3f} deg | mode={mode} | used={[n for n,_ in fits]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imwrite(os.path.join(out_dir, "scharr_ransac_final_overlay.png"), full_overlay)

    dbg = {
        "reason": "ok",
        "mode": mode,
        "used": [f[0] for f in fits],
        "avg_slope": avg_slope,
        "tilt_deg": tilt_deg,
        "top_fit": None if top_fit is None else {
            "m": top_fit[0], "c": top_fit[1], "n_pts": top_fit[2]
        },
        "bottom_fit": None if bot_fit is None else {
            "m": bot_fit[0], "c": bot_fit[1], "n_pts": bot_fit[2]
        },
        "params": {
            "band_frac": band_frac,
            "blur_ksize": blur_ksize,
            "thr_frac": thr_frac,
            "close_kernel": close_kernel,
            "close_iters": close_iters,
            "ransac_iters": ransac_iters,
            "ransac_inlier_px": ransac_inlier_px,
            "ransac_min_inliers": ransac_min_inliers,
        }
    }

    return tilt_deg, dbg
