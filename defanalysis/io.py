import os
import pandas as pd

REQUIRED_COLS = {"id", "frame", "center_x", "center_y"}

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_tracks_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found: {file_path}")
    df = pd.read_csv(file_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)} in {file_path}")
    return df

def save_summary_csv(rows, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    pd.DataFrame(rows).to_csv(out_path, index=False)
