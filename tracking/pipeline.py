import os
import cv2
import numpy as np

from .config import TrackingConfig
from .tracker_factory import create_tracker
from .preprocess import setup_gpu, create_preprocessors, preprocess_frame
from .detector import detect_objects
from .visualize import draw_tracked_objects, make_combined_view
from .io_utils import (
    make_output_dir,
    create_video_writer,
    save_sample_frames,
    save_tracking_csvs,
)

def track_objects_and_display(
    video_path,
    output_csv="id_tracking_trajectories.csv",
    output_count_csv="id_tracking_frame_counts.csv",
    config: TrackingConfig = TrackingConfig(),
):
    directory_path = make_output_dir(video_path)
    updated_output_csv = os.path.join(directory_path, output_csv)
    updated_output_count_csv = os.path.join(directory_path, output_count_csv)

    use_gpu = setup_gpu(config.use_gpu)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video = None
    video_output_path = os.path.join(directory_path, "Recorded_Annotated.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    fgbg, clahe = create_preprocessors(config)
    tracker = create_tracker(config.use_sort, max_distance=config.max_distance)

    particle_trajectories = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray, enhanced, fgmask, blurred, thresh = preprocess_frame(frame, fgbg, clahe, config)
        detections, radii_info = detect_objects(
            thresh,
            min_radius=config.min_radius,
            max_radius=config.max_radius
        )

        if config.use_sort:
            sort_dets = []
            for (x, y), r in radii_info.items():
                sort_dets.append([x - r, y - r, x + r, y + r])
            sort_dets = np.array(sort_dets)

            tracked = np.empty((0, 5)) if sort_dets.shape[0] == 0 else tracker.update(sort_dets)

            for track in tracked:
                x1, y1, x2, y2, track_id = track.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max((x2 - x1) // 2, 1)
                particle_trajectories.append({
                    "frame": frame_index,
                    "id": track_id,
                    "center_x": cx,
                    "center_y": cy,
                    "radius": radius
                })

            tracked_for_draw = tracked

        else:
            tracked_objects = tracker.update(detections)
            for obj_id, (x, y) in tracked_objects.items():
                radius = radii_info.get((x, y), 0)
                particle_trajectories.append({
                    "frame": frame_index,
                    "id": obj_id,
                    "center_x": x,
                    "center_y": y,
                    "radius": radius
                })

            tracked_for_draw = tracker.get_objects()

        if config.display_video or config.save_video:
            contour_frame = draw_tracked_objects(
                frame, tracked_for_draw, radii_info, config.use_sort
            )

            combined = make_combined_view(
                gray, blurred, thresh, contour_frame, config.display_scale
            )

            if config.save_video:
                if output_video is None:
                    output_video = create_video_writer(
                        video_output_path, fourcc, fps, combined.shape
                    )
                output_video.write(combined)

            if config.display_video:
                cv2.imshow("Tracking Pipeline with Labeled Views", combined)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break

        if config.save_sample_frames and frame_index == 20:
            save_sample_frames(directory_path, frame, gray, thresh)

        frame_index += 1

    cap.release()
    if config.display_video:
        cv2.destroyAllWindows()
    if output_video is not None:
        output_video.release()

    save_tracking_csvs(
        particle_trajectories,
        updated_output_csv,
        updated_output_count_csv
    )

    print(f"✅ Tracking complete using {'SORT' if config.use_sort else 'CentroidTracker'}.")
    if config.save_video:
        print(f"   Video saved: {video_output_path}")
    print(f"   CSVs saved:\n   - {updated_output_csv}\n   - {updated_output_count_csv}")

    return {
        "output_dir": directory_path,
        "trajectory_csv": updated_output_csv,
        "count_csv": updated_output_count_csv,
        "video_path": video_output_path if config.save_video else None,
    }