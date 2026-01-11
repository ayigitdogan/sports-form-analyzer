from typing import Dict, Tuple, List, Optional
import math

KeypointDict = Dict[str, Tuple[float, float, float]]  # name -> (x, y, conf)


def get_kp(kps: KeypointDict, name: str) -> Optional[Tuple[float, float, float]]:
    """Safe getter for a keypoint."""
    return kps.get(name, None)


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle_3pts(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float]
) -> float:
    """
    Angle ABC in degrees, where B is the vertex.
    """
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0.0

    cos_angle = max(min(dot / (mag_ab * mag_cb), 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def angle_from_vertical(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """
    Angle of vector p1->p2 relative to vertical axis (negative y is up in pixels).
    Returns degrees. 0° means perfectly vertical; larger means more horizontal.
    """
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]
    # vertical vector is (0, -1)
    # angle = arccos( (v • v_ref) / (|v||v_ref|) )
    mag = math.sqrt(vx * vx + vy * vy)
    if mag == 0:
        return 0.0
    # dot with (0, -1) = -vy
    cos_angle = (-vy) / mag
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def find_peak_idx(signal: List[float]) -> int:
    """
    Returns index of max value in a list; 0 if empty.
    """
    if not signal:
        return 0
    return max(range(len(signal)), key=lambda i: signal[i])
