from dataclasses import dataclass

@dataclass
class TrackingConfig:
    use_sort: bool = True
    use_gpu: bool = False
    max_distance: int = 30
    history: int = 100
    dist2_threshold: float = 400.0
    detect_shadows: bool = False
    clahe_clip_limit: float = 10.0
    clahe_tile_grid_size: tuple = (8, 8)
    blur_kernel: tuple = (11, 11)
    threshold_value: int = 30
    min_radius: int = 3
    max_radius: int = 50
    display_scale: float = 0.7
    save_video: bool = True
    display_video: bool = True
    save_sample_frames: bool = True