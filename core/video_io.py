import cv2
import numpy as np
from typing import List, Tuple, Optional, Union

def load_video(
    source: Union[str, bytes],
    max_frames: Optional[int] = None,
    target_fps: Optional[float] = None
) -> Tuple[List[np.ndarray], float]:
    """
    Load a video from disk (path) and return a list of BGR frames and the original fps.

    Args:
        source: path to video file.
        max_frames: if given, stop after this many frames.
        target_fps: if given, downsample frames to approximately this fps.

    Returns:
        frames: list of np.ndarray (H, W, 3) in BGR
        fps: original fps of the video
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0

    # compute sampling stride if target_fps is set
    stride = 1
    if target_fps is not None and orig_fps > 0:
        stride = max(int(round(orig_fps / target_fps)), 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # apply stride
        if target_fps is not None:
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

        frames.append(frame)
        frame_idx += 1

        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    return frames, orig_fps


def save_frame(frame: np.ndarray, out_path: str) -> None:
    """
    Save a single frame to disk (for debugging).
    """
    cv2.imwrite(out_path, frame)


def get_video_size(path: str) -> Tuple[int, int]:
    """
    Returns (width, height) of the video.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h
