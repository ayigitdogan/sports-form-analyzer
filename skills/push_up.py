from typing import List, Dict, Any, Optional
from core.features import angle_3pts, angle_from_vertical, euclidean


MIN_CONF = 0.35


def _avg_conf(pose_seq: List[Dict[str, Any]], side: str, names: List[str]) -> float:
    total = 0.0
    count = 0
    for f in pose_seq:
        kps = f["keypoints"]
        for name in names:
            kp = kps.get(f"{side}_{name}")
            if kp is not None:
                total += kp[2]
                count += 1
    return total / count if count else 0.0


def _choose_side(pose_seq: List[Dict[str, Any]]) -> str:
    names = ["SHOULDER", "ELBOW", "WRIST", "HIP", "KNEE", "ANKLE"]
    left_conf = _avg_conf(pose_seq, "LEFT", names)
    right_conf = _avg_conf(pose_seq, "RIGHT", names)
    return "LEFT" if left_conf >= right_conf else "RIGHT"


def _line_y_at_x(p1: tuple, p2: tuple, x: float) -> float:
    if abs(p2[0] - p1[0]) < 1e-6:
        return p1[1]
    t = (x - p1[0]) / (p2[0] - p1[0])
    return p1[1] + t * (p2[1] - p1[1])


def extract_features(
    pose_seq: List[Dict[str, Any]],
    aux: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if not pose_seq:
        return {}

    side = _choose_side(pose_seq)
    def series(name: str):
        return [f["keypoints"].get(f"{side}_{name}") for f in pose_seq]

    sh = series("SHOULDER")
    el = series("ELBOW")
    wr = series("WRIST")
    hip = series("HIP")
    knee = series("KNEE")
    ankle = series("ANKLE")

    elbow_angles: List[tuple] = []
    for i, (s, e, w) in enumerate(zip(sh, el, wr)):
        if not (s and e and w):
            continue
        if min(s[2], e[2], w[2]) < MIN_CONF:
            continue
        ang = angle_3pts((s[0], s[1]), (e[0], e[1]), (w[0], w[1]))
        elbow_angles.append((i, ang))

    if not elbow_angles:
        return {}

    min_idx, min_ang = min(elbow_angles, key=lambda x: x[1])
    max_idx, max_ang = max(elbow_angles, key=lambda x: x[1])

    s = sh[min_idx]
    e = el[min_idx]
    h = hip[min_idx]
    a = ankle[min_idx]

    depth_px = 0.0
    if s and e:
        depth_px = float(s[1] - e[1])

    body_line_deg = 0.0
    hip_offset_px = 0.0
    hip_offset_norm = 0.0
    if s and a and h:
        body_line_deg = abs(angle_from_vertical((s[0], s[1]), (a[0], a[1])))
        line_y = _line_y_at_x(s, a, h[0])
        hip_offset_px = float(h[1] - line_y)
        body_len = euclidean((s[0], s[1]), (a[0], a[1]))
        if body_len > 1.0:
            hip_offset_norm = hip_offset_px / body_len

    return {
        "elbow_min_deg": float(min_ang),
        "elbow_max_deg": float(max_ang),
        "depth_px": float(depth_px),
        "body_line_deg": float(body_line_deg),
        "hip_offset_px": float(hip_offset_px),
        "hip_offset_norm": float(hip_offset_norm),
        "bottom_frame_idx": int(min_idx),
        "top_frame_idx": int(max_idx),
    }


def evaluate(features: Dict[str, float]) -> Dict[str, Any]:
    issues: List[str] = []
    elbow_min = features.get("elbow_min_deg", 0.0)
    elbow_max = features.get("elbow_max_deg", 0.0)
    hip_offset = features.get("hip_offset_norm", 0.0)

    if elbow_min > 95.0:
        issues.append("shallow_depth")
    if elbow_max < 160.0:
        issues.append("no_lockout")
    if hip_offset > 0.08:
        issues.append("hip_sag")
    if hip_offset < -0.08:
        issues.append("hips_piking")

    score = 1.0 - 0.2 * len(issues)
    score = max(score, 0.0)
    return {"score": score, "issues": issues}


def recommend_drills(issues: List[str], kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = []
    for iss in issues:
        if iss in kb:
            item = kb[iss]
            recs.append({
                "issue": iss,
                "explain": item.get("explain", ""),
                "drills": item.get("drills", []),
            })
    return recs


def get_upload_spec() -> List[Dict[str, Any]]:
    return [
        {
            "key": "side",
            "label": "Side clip",
            "instruction": "True side view to judge elbow depth and body line.",
            "required": True,
        }
    ]


def get_aux_sidebar(st) -> Optional[Dict[str, float]]:
    return None


def select_key_frame(features: Dict[str, Any]) -> int:
    return int(features.get("bottom_frame_idx", 0))


def select_key_frames(features: Dict[str, Any]) -> List[int]:
    return [
        int(features.get("bottom_frame_idx", 0)),
        int(features.get("top_frame_idx", 0)),
    ]


def get_overlay_labels(features: Dict[str, Any]) -> List[str]:
    return [
        f"elbow_min={features.get('elbow_min_deg', 0):.1f} deg",
        f"hip_offset={features.get('hip_offset_norm', 0):.2f}",
    ]


def summarize_for_prompt(features: Dict[str, Any]) -> str:
    elbow_min = features.get("elbow_min_deg", 0.0)
    elbow_max = features.get("elbow_max_deg", 0.0)
    body_line = features.get("body_line_deg", 0.0)
    hip_offset = features.get("hip_offset_norm", 0.0)
    hip_line = "level"
    if hip_offset > 0.08:
        hip_line = "sagging"
    elif hip_offset < -0.08:
        hip_line = "piked"
    return (
        f"elbow range about {elbow_min:.0f}-{elbow_max:.0f} deg, "
        f"body line angle {body_line:.0f} deg, hips {hip_line}"
    )
