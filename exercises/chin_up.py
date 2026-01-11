from typing import List, Dict, Any, Optional
from core.features import angle_3pts, angle_from_vertical, find_peak_idx


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
    names = ["SHOULDER", "ELBOW", "WRIST", "HIP"]
    left_conf = _avg_conf(pose_seq, "LEFT", names)
    right_conf = _avg_conf(pose_seq, "RIGHT", names)
    return "LEFT" if left_conf >= right_conf else "RIGHT"


def extract_features(
    pose_seq: List[Dict[str, Any]],
    aux: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if not pose_seq:
        return {}

    # peak frame: highest wrist (min y)
    wrist_y = []
    for f in pose_seq:
        kps = f["keypoints"]
        lw = kps.get("LEFT_WRIST")
        rw = kps.get("RIGHT_WRIST")
        cand = None
        if lw and rw:
            cand = lw if lw[1] < rw[1] else rw
        else:
            cand = lw or rw
        wrist_y.append(cand[1] if cand else None)

    inv_wrist = [(-y if y is not None else -1e9) for y in wrist_y]
    peak_idx = find_peak_idx(inv_wrist)

    view = "unknown"
    if isinstance(aux, dict):
        view = aux.get("view", "unknown")
    side = _choose_side(pose_seq)
    def series(name: str, side_name: str):
        return [f["keypoints"].get(f"{side_name}_{name}") for f in pose_seq]

    sh = series("SHOULDER", side)
    el = series("ELBOW", side)
    wr = series("WRIST", side)

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

    frame_kps = pose_seq[peak_idx]["keypoints"]
    l_sh = frame_kps.get("LEFT_SHOULDER")
    r_sh = frame_kps.get("RIGHT_SHOULDER")
    l_el = frame_kps.get("LEFT_ELBOW")
    r_el = frame_kps.get("RIGHT_ELBOW")
    l_wr = frame_kps.get("LEFT_WRIST")
    r_wr = frame_kps.get("RIGHT_WRIST")
    l_hip = frame_kps.get("LEFT_HIP")
    r_hip = frame_kps.get("RIGHT_HIP")

    shoulder_width_px = 0.0
    if l_sh and r_sh:
        shoulder_width_px = float(abs(l_sh[0] - r_sh[0]))

    elbow_sym_norm = 0.0
    wrist_sym_norm = 0.0
    symmetry_reliable = 1.0 if view == "front" else 0.0
    swing_reliable = 1.0 if view == "side" else 0.0
    if symmetry_reliable and l_el and r_el and shoulder_width_px > 1.0:
        elbow_sym_norm = abs(l_el[1] - r_el[1]) / shoulder_width_px
    if symmetry_reliable and l_wr and r_wr and shoulder_width_px > 1.0:
        wrist_sym_norm = abs(l_wr[1] - r_wr[1]) / shoulder_width_px

    torso_forward_deg = 0.0
    if l_sh and r_sh and l_hip and r_hip:
        sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        torso_forward_deg = abs(angle_from_vertical(sh_mid, hip_mid))
    else:
        s = sh[peak_idx]
        h = series("HIP", side)[peak_idx]
        if s and h:
            torso_forward_deg = abs(angle_from_vertical((s[0], s[1]), (h[0], h[1])))

    pulling_angle_deg = 0.0
    s = sh[peak_idx]
    w = wr[peak_idx]
    if s and w:
        pulling_angle_deg = abs(angle_from_vertical((s[0], s[1]), (w[0], w[1])))

    return {
        "elbow_min_deg": float(min_ang),
        "elbow_max_deg": float(max_ang),
        "pulling_angle_deg": float(pulling_angle_deg),
        "torso_forward_deg": float(torso_forward_deg),
        "elbow_sym_norm": float(elbow_sym_norm),
        "wrist_sym_norm": float(wrist_sym_norm),
        "symmetry_reliable": float(symmetry_reliable),
        "swing_reliable": float(swing_reliable),
        "shoulder_width_px": float(shoulder_width_px),
        "peak_wrist_frame": int(peak_idx),
        "top_frame_idx": int(min_idx),
        "bottom_frame_idx": int(max_idx),
    }


def evaluate(features: Dict[str, float]) -> Dict[str, Any]:
    issues: List[str] = []
    elbow_min = features.get("elbow_min_deg", 0.0)
    elbow_max = features.get("elbow_max_deg", 0.0)
    torso = features.get("torso_forward_deg", 0.0)
    pull_ang = features.get("pulling_angle_deg", 0.0)
    elbow_sym = features.get("elbow_sym_norm", 0.0)
    wrist_sym = features.get("wrist_sym_norm", 0.0)

    if elbow_min > 85.0:
        issues.append("limited_top_range")
    if elbow_max < 160.0:
        issues.append("no_dead_hang")
    if features.get("swing_reliable", 0.0) and (torso > 35.0 or pull_ang > 55.0):
        issues.append("swinging")
    if features.get("symmetry_reliable", 0.0) and (elbow_sym > 0.15 or wrist_sym > 0.15):
        issues.append("asymmetry")

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
            "instruction": "Side view to judge swing, torso angle, and pulling path.",
            "required": False,
        },
        {
            "key": "front",
            "label": "Front clip",
            "instruction": "Front view to judge symmetry; upload at least one angle.",
            "required": False,
        },
    ]


def get_aux_sidebar(st) -> Optional[Dict[str, float]]:
    return None


def select_key_frame(features: Dict[str, Any]) -> int:
    return int(features.get("peak_wrist_frame", 0))


def select_key_frames(features: Dict[str, Any]) -> List[int]:
    return [
        int(features.get("top_frame_idx", 0)),
        int(features.get("bottom_frame_idx", 0)),
    ]


def get_overlay_labels(features: Dict[str, Any]) -> List[str]:
    return [
        f"elbow_min={features.get('elbow_min_deg', 0):.1f} deg",
        f"torso={features.get('torso_forward_deg', 0):.1f} deg",
    ]


def summarize_for_prompt(features: Dict[str, Any]) -> str:
    elbow_min = features.get("elbow_min_deg", 0.0)
    elbow_max = features.get("elbow_max_deg", 0.0)
    torso = features.get("torso_forward_deg", 0.0)
    pull = features.get("pulling_angle_deg", 0.0)
    symmetry = "not assessed (side view)"
    if features.get("symmetry_reliable", 0.0):
        symmetry = "even left/right" if max(features.get("elbow_sym_norm", 0.0), features.get("wrist_sym_norm", 0.0)) <= 0.15 else "uneven left/right"
    return (
        f"elbow range about {elbow_min:.0f}-{elbow_max:.0f} deg, "
        f"torso angle {torso:.0f} deg, pull angle {pull:.0f} deg, "
        f"symmetry {symmetry}"
    )
