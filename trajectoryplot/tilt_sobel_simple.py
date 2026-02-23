# tilt_sobel_simple.py
import os
import cv2
import numpy as np
import math


def estimate_tilt_sobel_zonal(
    gray_image_path: str,
    out_dir: str = None,
    save_debug: bool = True,     # NEW: set False to skip saving debug images

    mode: str = "both",          # "top", "bottom", "both"
    top_frac: float = 0.30,
    bottom_frac: float = 0.30,

    blur_ksize: int = 9,         # Gaussian blur kernel size (odd)
    clahe_clip: float = 3.0,
    clahe_grid=(8, 8),

    sobel_ksize: int = 5,        # Sobel kernel size (odd)
    open_ksize: int = 3,         # morphology open kernel (square)
    close_len: int = 30,         # morphology close kernel (1 x close_len)

    min_pts: int = 20            # minimum edge pixels to accept fit
):
    """
    Simple, robust tilt estimation using YOUR working pipeline:
      - blur
      - ignore middle region via top/bottom zones
      - CLAHE + Sobel-Y + Otsu threshold
      - open + horizontal close to connect the wall
      - fit line y = m x + c for top/bottom
      - angle_deg = atan(mean slope)

    Returns: angle_deg, dbg_dict

    Debug outputs (if save_debug=True and out_dir is provided), using your standard filenames:
      - tilt_roi_raw.png
      - tilt_roi_eq.png
      - tilt_gradmag.png
      - tilt_edges.png
      - tilt_overlay.png
    """
    if not gray_image_path or not os.path.exists(gray_image_path):
        return 0.0, {"reason": "gray image not found"}

    gray = cv2.imread(gray_image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0.0, {"reason": "failed to read gray image"}

    h, w = gray.shape[:2]

    # enforce odd sizes
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if sobel_ksize % 2 == 0:
        sobel_ksize += 1

    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    top_limit = int(h * top_frac)
    bottom_start = int(h * (1.0 - bottom_frac))

    top_limit = max(1, min(top_limit, h - 1))
    bottom_start = max(0, min(bottom_start, h - 1))

    def get_filtered_edges(zone_img):
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        enhanced = clahe.apply(zone_img)

        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        sobely_u8 = np.uint8(np.absolute(sobely))

        # Otsu decides threshold; value "50" is ignored by Otsu but kept for consistency
        _, thresh = cv2.threshold(sobely_u8, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        kernel_open = np.ones((open_ksize, open_ksize), np.uint8)
        kernel_close = np.ones((1, max(3, int(close_len))), np.uint8)

        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        return enhanced, sobely_u8, thresh

    # Zones
    top_zone = blurred[0:top_limit, :]
    bottom_zone = blurred[bottom_start:h, :]

    top_enh, top_sob, top_edges = get_filtered_edges(top_zone)
    bot_enh, bot_sob, bot_edges = get_filtered_edges(bottom_zone)

    # Full edge map (for debug)
    full_edges = np.zeros_like(gray)
    full_edges[0:top_limit, :] = top_edges
    full_edges[bottom_start:h, :] = bot_edges

    def get_pts(edge_zone, y_offset):
        coords = np.column_stack(np.where(edge_zone > 0))  # (y,x) in zone coords
        if coords.shape[0] > 0:
            coords[:, 0] += int(y_offset)                 # convert to full-image y
            return coords
        return None

    top_pts = get_pts(top_edges, 0)
    bot_pts = get_pts(bot_edges, bottom_start)

    def fit_line(pts):
        if pts is None or pts.shape[0] < int(min_pts):
            return None
        y = pts[:, 0].astype(np.float32)
        x = pts[:, 1].astype(np.float32)
        m, c = np.polyfit(x, y, 1)  # y = m x + c
        return float(m), float(c), int(pts.shape[0])

    top_fit = fit_line(top_pts)
    bot_fit = fit_line(bot_pts)

    use = (mode or "both").strip().lower()
    fits = []
    if use in ["top", "both"] and top_fit is not None:
        fits.append(("top", top_fit))
    if use in ["bottom", "both"] and bot_fit is not None:
        fits.append(("bottom", bot_fit))

    # fallback if requested side missing
    if len(fits) == 0:
        if use == "top" and bot_fit is not None:
            fits = [("bottom", bot_fit)]
            used_mode = "top_fallback_to_bottom"
        elif use == "bottom" and top_fit is not None:
            fits = [("top", top_fit)]
            used_mode = "bottom_fallback_to_top"
        else:
            dbg = {
                "reason": "no valid line fits",
                "mode": mode,
                "top_pts": 0 if top_pts is None else int(top_pts.shape[0]),
                "bottom_pts": 0 if bot_pts is None else int(bot_pts.shape[0]),
                "min_pts": int(min_pts),
            }
            if save_debug and out_dir:
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, "tilt_edges.png"), full_edges)
                cv2.imwrite(os.path.join(out_dir, "tilt_roi_raw.png"), gray)
                cv2.imwrite(os.path.join(out_dir, "tilt_roi_eq.png"), gray)
                cv2.imwrite(os.path.join(out_dir, "tilt_gradmag.png"), np.zeros_like(gray))
                overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(os.path.join(out_dir, "tilt_overlay.png"), overlay)
            return 0.0, dbg
    else:
        used_mode = use

    slopes = [fit[0] for _, fit in fits]  # fit=(m,c,n)
    avg_slope = float(np.mean(slopes)) if slopes else 0.0
    tilt_rad = math.atan(avg_slope)
    tilt_deg = float(math.degrees(tilt_rad))

    dbg = {
        "reason": "ok",
        "mode": mode,
        "used_mode": used_mode,
        "used": [name for name, _ in fits],
        "avg_slope": avg_slope,
        "tilt_deg": tilt_deg,
        "top_fit": None if top_fit is None else {"m": top_fit[0], "c": top_fit[1], "n_pts": top_fit[2]},
        "bottom_fit": None if bot_fit is None else {"m": bot_fit[0], "c": bot_fit[1], "n_pts": bot_fit[2]},
        "top_limit": int(top_limit),
        "bottom_start": int(bottom_start),
        "params": {
            "top_frac": top_frac,
            "bottom_frac": bottom_frac,
            "blur_ksize": blur_ksize,
            "clahe_clip": clahe_clip,
            "sobel_ksize": sobel_ksize,
            "open_ksize": open_ksize,
            "close_len": close_len,
            "min_pts": int(min_pts),
        }
    }

    # ---- Debug images (optional) ----
    if save_debug and out_dir:
        os.makedirs(out_dir, exist_ok=True)

        # keep your convention: save bottom ROI raw/eq
        roi_raw = gray[bottom_start:h, :]
        cv2.imwrite(os.path.join(out_dir, "tilt_roi_raw.png"), roi_raw)
        cv2.imwrite(os.path.join(out_dir, "tilt_roi_eq.png"), bot_enh)

        # grad magnitude visualization (full frame sobel-y abs)
        sob_full = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        grad_vis = cv2.normalize(np.uint8(np.absolute(sob_full)), None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(out_dir, "tilt_gradmag.png"), grad_vis)

        # edges
        cv2.imwrite(os.path.join(out_dir, "tilt_edges.png"), full_edges)

        # overlay lines
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # draw top/bottom fits in cyan
        for fit in [top_fit, bot_fit]:
            if fit is None:
                continue
            m, c, _n = fit
            yL = int(c)
            yR = int(m * (w - 1) + c)
            cv2.line(overlay, (0, yL), (w - 1, yR), (0, 255, 255), 2)

        # draw avg line in red through image center: y = m(x-cx)+cy
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        yL = int(cy + avg_slope * (0 - cx))
        yR = int(cy + avg_slope * ((w - 1) - cx))
        cv2.line(overlay, (0, yL), (w - 1, yR), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(out_dir, "tilt_overlay.png"), overlay)

    return tilt_deg, dbg
