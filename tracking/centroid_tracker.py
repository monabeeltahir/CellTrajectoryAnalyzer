class CentroidTracker:
    def __init__(self, max_distance=25):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def update(self, detections):
        updated_ids = {}

        if not self.objects:
            for det in detections:
                self.objects[self.next_id] = det
                updated_ids[self.next_id] = det
                self.next_id += 1
            return updated_ids

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))
        unmatched_detections = detections.copy()

        for det in detections:
            dist = np.linalg.norm(obj_centroids - det, axis=1)
            min_idx = np.argmin(dist)
            if dist[min_idx] < self.max_distance:
                matched_id = obj_ids[min_idx]
                updated_ids[matched_id] = det
                self.objects[matched_id] = det
                if det in unmatched_detections:
                    unmatched_detections.remove(det)

        for det in unmatched_detections:
            self.objects[self.next_id] = det
            updated_ids[self.next_id] = det
            self.next_id += 1

        return updated_ids

    def get_objects(self):
        return self.objects