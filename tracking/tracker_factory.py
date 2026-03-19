from .centroid_tracker import CentroidTracker

def create_tracker(use_sort: bool, max_distance: int = 30):
    if use_sort:
        from .sort import Sort
        return Sort()
    return CentroidTracker(max_distance=max_distance)