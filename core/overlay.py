import cv2
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

# we'll hardcode a simple skeleton for MediaPipe Pose
# (this is a subset, enough for upper-body stuff)
MEDIAPIPE_EDGES = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]


def draw_keypoints(
    frame: np.ndarray,
    keypoints: Dict[str, Tuple[float, float, float]],
    radius: int = 4,
    color: Tuple[int, int, int] = (0, 255, 0),
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """
    Draws circles for keypoints on the frame.
    """
    out = frame.copy()
    for name, (x, y, c) in keypoints.items():
        if c < conf_thresh:
            continue
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def draw_skeleton(
    frame: np.ndarray,
    keypoints: Dict[str, Tuple[float, float, float]],
    edges: List[Tuple[str, str]] = MEDIAPIPE_EDGES,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """
    Draws lines between keypoints according to edges.
    """
    out = frame.copy()
    for a, b in edges:
        if a not in keypoints or b not in keypoints:
            continue
        x1, y1, c1 = keypoints[a]
        x2, y2, c2 = keypoints[b]
        if c1 < conf_thresh or c2 < conf_thresh:
            continue
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return out


def draw_label(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int] = (10, 30),
    bgcolor: Tuple[int, int, int] = (0, 0, 0),
    fgcolor: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Draws a text label with background.
    """
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(out, (x, y - h - 4), (x + w + 4, y + 4), bgcolor, -1)
    cv2.putText(out, text, (x + 2, y), font, scale, fgcolor, thickness, cv2.LINE_AA)
    return out


def render_pose_frame(
    frame: np.ndarray,
    frame_pose: Dict[str, Any],
    extra_labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Convenience: takes raw frame + one element from pose_seq, returns annotated frame.
    """
    out = draw_skeleton(frame, frame_pose["keypoints"])
    out = draw_keypoints(out, frame_pose["keypoints"])

    if extra_labels:
        y = 30
        for text in extra_labels:
            out = draw_label(out, text, org=(10, y))
            y += 25

    return out


def draw_bar_line(
    frame: np.ndarray,
    y: int,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a horizontal line at bar height (for muscle-up reference).
    """
    h, w, _ = frame.shape
    out = frame.copy()
    cv2.line(out, (0, y), (w, y), color, thickness)
    return out
