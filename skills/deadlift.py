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
    names = ["SHOULDER", "HIP", "KNEE", "ANKLE"]
    left_conf = _avg_conf(pose_seq, "LEFT", names)
    right_conf = _avg_conf(pose_seq, "RIGHT", names)
    return "LEFT" if left_conf >= right_conf else "RIGHT"


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
    hip = series("HIP")
    knee = series("KNEE")
    ankle = series("ANKLE")

    hip_y = []
    for h in hip:
        hip_y.append(h[1] if h else None)
    hip_vals = [(y if y is not None else -1e9) for y in hip_y]
    bottom_idx = max(range(len(hip_vals)), key=lambda i: hip_vals[i]) if hip_vals else 0
    top_idx = min(range(len(hip_vals)), key=lambda i: hip_vals[i]) if hip_vals else 0

    s = sh[bottom_idx]
    h = hip[bottom_idx]
    k = knee[bottom_idx]
    a = ankle[bottom_idx]

    torso_forward_deg = 0.0
    if s and h:
        if min(s[2], h[2]) >= MIN_CONF:
            torso_forward_deg = abs(angle_from_vertical((s[0], s[1]), (h[0], h[1])))

    shin_angle_deg = 0.0
    if k and a:
        if min(k[2], a[2]) >= MIN_CONF:
            shin_angle_deg = abs(angle_from_vertical((k[0], k[1]), (a[0], a[1])))

    hip_hinge_deg = 0.0
    if s and h and k and min(s[2], h[2], k[2]) >= MIN_CONF:
        hip_hinge_deg = angle_3pts((s[0], s[1]), (h[0], h[1]), (k[0], k[1]))

    hip_vs_knee_px = 0.0
    hip_vs_knee_norm = 0.0
    shoulder_over_ankle_x = 0.0
    shoulder_over_ankle_norm = 0.0
    if h and k and a:
        hip_vs_knee_px = float(h[1] - k[1])
        leg_len = euclidean((k[0], k[1]), (a[0], a[1]))
        if leg_len > 1.0:
            hip_vs_knee_norm = hip_vs_knee_px / leg_len
            if s:
                shoulder_over_ankle_x = float(s[0] - a[0])
                shoulder_over_ankle_norm = shoulder_over_ankle_x / leg_len

    return {
        "torso_forward_deg": float(torso_forward_deg),
        "shin_angle_deg": float(shin_angle_deg),
        "hip_hinge_deg": float(hip_hinge_deg),
        "hip_vs_knee_px": float(hip_vs_knee_px),
        "hip_vs_knee_norm": float(hip_vs_knee_norm),
        "shoulder_over_ankle_x": float(shoulder_over_ankle_x),
        "shoulder_over_ankle_norm": float(shoulder_over_ankle_norm),
        "bottom_frame_idx": int(bottom_idx),
        "top_frame_idx": int(top_idx),
    }


def evaluate(features: Dict[str, float]) -> Dict[str, Any]:
    issues: List[str] = []
    torso = features.get("torso_forward_deg", 0.0)
    shin = features.get("shin_angle_deg", 0.0)
    hip_norm = features.get("hip_vs_knee_norm", 0.0)
    shoulder_norm = features.get("shoulder_over_ankle_norm", 0.0)
    hip_hinge = features.get("hip_hinge_deg", 0.0)

    if torso > 70.0:
        issues.append("excessive_forward_lean")
    if torso < 25.0:
        issues.append("too_upright")
    if shin > 30.0:
        issues.append("knees_forward")
    if hip_norm > 0.35:
        issues.append("hips_too_low")
    if hip_norm < -0.35:
        issues.append("hips_too_high")
    if shoulder_norm > 0.35:
        issues.append("shoulders_forward")
    if hip_hinge and hip_hinge < 140.0:
        issues.append("rounded_back_risk")

    score = 1.0 - 0.15 * len(issues)
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
            "instruction": "True side view to assess hip/shoulder positions and back angle.",
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
        f"torso={features.get('torso_forward_deg', 0):.1f} deg",
        f"shin={features.get('shin_angle_deg', 0):.1f} deg",
    ]


def summarize_for_prompt(features: Dict[str, Any]) -> str:
    torso = features.get("torso_forward_deg", 0.0)
    shin = features.get("shin_angle_deg", 0.0)
    hip_norm = features.get("hip_vs_knee_norm", 0.0)
    shoulder_norm = features.get("shoulder_over_ankle_norm", 0.0)
    hip_hinge = features.get("hip_hinge_deg", 0.0)

    hip_line = "hips around knee height"
    if hip_norm > 0.2:
        hip_line = "hips below knees"
    elif hip_norm < -0.2:
        hip_line = "hips above knees"

    shoulder_line = "shoulders near midfoot"
    if shoulder_norm > 0.2:
        shoulder_line = "shoulders in front of midfoot"
    elif shoulder_norm < -0.2:
        shoulder_line = "shoulders behind midfoot"

    return (
        f"torso angle {torso:.0f} deg, shin angle {shin:.0f} deg, "
        f"{hip_line}, {shoulder_line}, hip hinge {hip_hinge:.0f} deg"
    )
