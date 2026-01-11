from typing import List, Dict, Any
import numpy as np

# MediaPipe imports
import mediapipe as mp

mp_pose = mp.solutions.pose

# we will name keypoints for easier use later
POSE_LANDMARKS = {lm.name: lm.value for lm in mp_pose.PoseLandmark}


def run_pose_on_frames(
    frames: List[np.ndarray],
    static_image_mode: bool = False,
    model_complexity: int = 1,
    smooth_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Run MediaPipe Pose on a list of BGR frames.
    Returns a list of dicts, one per frame:
    {
        "frame_idx": i,
        "keypoints": {
            "LEFT_SHOULDER": (x, y, conf),
            ...
        },
        "image_shape": (h, w)
    }
    """
    results_seq: List[Dict[str, Any]] = []

    with mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:

        for i, frame in enumerate(frames):
            # MediaPipe expects RGB
            image_rgb = frame[:, :, ::-1]
            res = pose.process(image_rgb)

            h, w, _ = frame.shape
            kps: Dict[str, Any] = {}

            if res.pose_landmarks:
                for name, idx in POSE_LANDMARKS.items():
                    lm = res.pose_landmarks.landmark[idx]
                    x_px = lm.x * w
                    y_px = lm.y * h
                    conf = lm.visibility
                    kps[name] = (float(x_px), float(y_px), float(conf))
            else:
                # no detection
                kps = {}

            results_seq.append({
                "frame_idx": i,
                "keypoints": kps,
                "image_shape": (h, w)
            })

    return results_seq


def exp_smooth_pose(
    pose_seq: List[Dict[str, Any]],
    alpha: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Exponential smoothing over keypoints to reduce jitter.
    Assumes all frames have the same key set (ok if some are missing; we'll skip).
    """
    if not pose_seq:
        return pose_seq

    smoothed = []
    prev_points = {}

    for frame_data in pose_seq:
        kps = frame_data["keypoints"]
        sm_kps = {}
        for name, kp in kps.items():
            x, y, c = kp
            if name not in prev_points:
                sm_kps[name] = (x, y, c)
            else:
                px, py, pc = prev_points[name]
                sm_x = alpha * x + (1 - alpha) * px
                sm_y = alpha * y + (1 - alpha) * py
                sm_c = alpha * c + (1 - alpha) * pc
                sm_kps[name] = (sm_x, sm_y, sm_c)
        smoothed.append({
            "frame_idx": frame_data["frame_idx"],
            "keypoints": sm_kps,
            "image_shape": frame_data["image_shape"]
        })
        prev_points = sm_kps

    return smoothed
