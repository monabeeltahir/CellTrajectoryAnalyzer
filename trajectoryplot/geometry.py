import numpy as np

def rotate_points(x, y, angle_deg, cx, cy):
    """
    Rotate points (x,y) by -angle_deg about (cx,cy) to remove tilt.
    """
    theta = np.radians(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xr = cx + (x - cx) * c + (y - cy) * s
    yr = cy - (x - cx) * s + (y - cy) * c
    return xr, yr
