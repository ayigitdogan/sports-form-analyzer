from typing import List, Dict, Any, Optional
from core.features import angle_from_vertical, find_peak_idx


def extract_features(
    pose_seq: List[Dict[str, Any]],
    aux: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if not pose_seq:
        return {}
    view = "unknown"
    if isinstance(aux, dict):
        view = aux.get("view", "unknown")

    # helper to extract series
    def kp_series(name: str):
        return [f["keypoints"].get(name) for f in pose_seq]

    l_hip = kp_series("LEFT_HIP")
    r_hip = kp_series("RIGHT_HIP")
    l_knee = kp_series("LEFT_KNEE")
    r_knee = kp_series("RIGHT_KNEE")
    l_ankle = kp_series("LEFT_ANKLE")
    r_ankle = kp_series("RIGHT_ANKLE")
    l_sh = kp_series("LEFT_SHOULDER")
    r_sh = kp_series("RIGHT_SHOULDER")

    # mid-hip y for bottom detection (max y since pixels increase downward)
    hip_mid_y = []
    for lh, rh in zip(l_hip, r_hip):
        cand = None
        if lh and rh:
            cand = (lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0, (lh[2] + rh[2]) / 2.0
        else:
            cand = lh or rh
        hip_mid_y.append(cand[1] if cand else None)

    # choose bottom frame as max hip_mid_y
    hip_vals = [(y if y is not None else -1e9) for y in hip_mid_y]
    bottom_idx = max(range(len(hip_vals)), key=lambda i: hip_vals[i]) if hip_vals else 0
    hip_vals_top = [(y if y is not None else 1e9) for y in hip_mid_y]
    top_idx = min(range(len(hip_vals_top)), key=lambda i: hip_vals_top[i]) if hip_vals_top else 0

    frame_kps = pose_seq[bottom_idx]["keypoints"]
    # mid hip and mid shoulder at bottom
    lh = frame_kps.get("LEFT_HIP")
    rh = frame_kps.get("RIGHT_HIP")
    ls = frame_kps.get("LEFT_SHOULDER")
    rs = frame_kps.get("RIGHT_SHOULDER")
    lkn = frame_kps.get("LEFT_KNEE")
    rkn = frame_kps.get("RIGHT_KNEE")
    lak = frame_kps.get("LEFT_ANKLE")
    rak = frame_kps.get("RIGHT_ANKLE")

    hip_mid = None
    sh_mid = None
    if lh and rh:
        hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, (lh[2] + rh[2]) / 2)
    else:
        hip_mid = lh or rh

    if ls and rs:
        sh_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, (ls[2] + rs[2]) / 2)
    else:
        sh_mid = ls or rs

    # depth: hip relative to knee (avg of both knees if present)
    depth_px = 0.0
    sagittal_reliable = 1.0 if view == "side" else 0.0
    if sagittal_reliable and hip_mid and (lkn or rkn):
        knees_y = []
        if lkn:
            knees_y.append(lkn[1])
        if rkn:
            knees_y.append(rkn[1])
        if knees_y:
            knee_avg_y = sum(knees_y) / len(knees_y)
            depth_px = float(hip_mid[1] - knee_avg_y)  # positive means hip below knee

    # torso angle from vertical
    torso_forward_deg = 0.0
    if sagittal_reliable and hip_mid and sh_mid:
        torso_forward_deg = abs(angle_from_vertical((sh_mid[0], sh_mid[1]), (hip_mid[0], hip_mid[1])))

    # valgus ratio: knee distance vs ankle distance (horizontal spacing proxy)
    valgus_ratio = 1.0
    valgus_reliable = 1.0 if view == "front" else 0.0
    if valgus_reliable and lkn and rkn and lak and rak:
        knee_dist = abs(lkn[0] - rkn[0])
        ankle_dist = abs(lak[0] - rak[0])
        if ankle_dist > 1e-6:
            valgus_ratio = float(knee_dist / ankle_dist)

    return {
        "depth_px": float(depth_px),
        "torso_forward_deg": float(torso_forward_deg),
        "valgus_ratio": float(valgus_ratio),
        "valgus_reliable": float(valgus_reliable),
        "sagittal_reliable": float(sagittal_reliable),
        "bottom_frame_idx": int(bottom_idx),
        "top_frame_idx": int(top_idx),
    }


def evaluate(features: Dict[str, float]) -> Dict[str, Any]:
    issues: List[str] = []
    depth = features.get("depth_px", 0.0)
    torso = features.get("torso_forward_deg", 0.0)
    valgus = features.get("valgus_ratio", 1.0)

    if features.get("sagittal_reliable", 0.0):
        if depth <= 0.0:
            issues.append("shallow_squat")
        if torso > 45.0:
            issues.append("excessive_forward_lean")
    if features.get("valgus_reliable", 0.0) and valgus < 0.9:  # knees closer than ankles
        issues.append("knee_valgus")

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


# ---- Optional hooks for UI/LLM generalization ----
def get_upload_spec() -> List[Dict[str, Any]]:
    """Two slots: front/oblique and side; either angle is accepted."""
    return [
        {
            "key": "front",
            "label": "Front/oblique clip (preferred)",
            "instruction": "Front-on or ~30-45 deg oblique to assess knee valgus and stance.",
            "required": False,
        },
        {
            "key": "side",
            "label": "Side clip",
            "instruction": "True side/sagittal view to assess torso angle and depth.",
            "required": False,
        },
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
        f"depth={features.get('depth_px', 0):.1f}px",
        f"torso={features.get('torso_forward_deg', 0):.1f} deg",
    ]


def summarize_for_prompt(features: Dict[str, Any]) -> str:
    depth = features.get("depth_px", 0.0)
    torso = features.get("torso_forward_deg", 0.0)
    valgus = "not assessed (side view)"
    if features.get("valgus_reliable", 0.0):
        valgus = "knees track over feet" if features.get("valgus_ratio", 1.0) >= 0.9 else "knees drift inward"
    if features.get("sagittal_reliable", 0.0):
        depth_line = "below knee" if depth > 0 else "above knee"
        depth_txt = f"depth {depth_line}, torso angle {torso:.0f} deg"
    else:
        depth_txt = "depth/torso not assessed (front view)"
    return f"{depth_txt}, {valgus}"
