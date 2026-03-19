import cv2

def detect_objects(thresh, min_radius=3, max_radius=50):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    radii_info = {}

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        radius = int(radius)
        if min_radius < radius < max_radius:
            center = (int(x), int(y))
            detections.append(center)
            radii_info[center] = radius

    return detections, radii_info