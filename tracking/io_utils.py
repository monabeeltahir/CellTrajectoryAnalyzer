import os
import cv2
import pandas as pd

def make_output_dir(video_path):
    directory_path = os.path.splitext(video_path)[0]
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def create_video_writer(output_path, fourcc, fps, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


def save_sample_frames(directory_path, frame, gray, thresh):
    cv2.imwrite(os.path.join(directory_path, "OriginalFrame.png"), frame)
    cv2.imwrite(os.path.join(directory_path, "GrayImage.png"), gray)
    cv2.imwrite(os.path.join(directory_path, "Threshold.png"), thresh)


def save_tracking_csvs(particle_trajectories, output_csv, output_count_csv):
    expected_cols = ["frame", "id", "center_x", "center_y", "radius"]

    if len(particle_trajectories) == 0:
        trajectory_df = pd.DataFrame(columns=expected_cols)
        frame_counts_df = pd.DataFrame(columns=["frame", "count"])
    else:
        trajectory_df = pd.DataFrame(particle_trajectories)

        # ensure expected columns exist even if malformed rows appear
        for col in expected_cols:
            if col not in trajectory_df.columns:
                trajectory_df[col] = pd.NA

        frame_counts_df = (
            trajectory_df.groupby("frame", dropna=False)
            .size()
            .reset_index(name="count")
        )

    trajectory_df.to_csv(output_csv, index=False)
    frame_counts_df.to_csv(output_count_csv, index=False)

    return trajectory_df, frame_counts_df