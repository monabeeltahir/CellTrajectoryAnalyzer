import cv2

def label_and_resize(image, label, scale=0.8):
    img = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    cv2.putText(
        img, label, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return img


def draw_tracked_objects(frame, tracked, radii_info, use_sort):
    contour_frame = frame.copy()

    if use_sort:
        for track in tracked:
            x1, y1, x2, y2, track_id = track.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max((x2 - x1) // 2, 1)
            cv2.circle(contour_frame, (cx, cy), radius, (0, 255, 0), 2)
            cv2.putText(
                contour_frame, f"ID {track_id}", (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2
            )
    else:
        for obj_id, (x, y) in tracked.items():
            radius = radii_info.get((x, y), 0)
            cv2.circle(contour_frame, (x, y), radius, (0, 255, 0), 1)
            cv2.putText(
                contour_frame, f"ID {obj_id}", (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )

    return contour_frame


def make_combined_view(gray, blurred_image_diff, thresh, contour_frame, scale):
    gray_bgr = label_and_resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Gray Image", scale)
    diff_bgr = label_and_resize(
        cv2.cvtColor(blurred_image_diff, cv2.COLOR_GRAY2BGR),
        "Difference from Background", scale
    )
    thresh_bgr = label_and_resize(
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
        "Cleaned Binary Motion", scale
    )
    contour_resized = label_and_resize(contour_frame, "Motion with ID Tracking", scale)

    top_row = cv2.hconcat([gray_bgr, diff_bgr])
    bottom_row = cv2.hconcat([thresh_bgr, contour_resized])
    return cv2.vconcat([top_row, bottom_row])