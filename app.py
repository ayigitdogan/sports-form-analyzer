import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import cv2
import numpy as np

from core.video_io import load_video
from core.pose import run_pose_on_frames, exp_smooth_pose
from core.overlay import render_pose_frame, draw_bar_line
from skills.registry import get_skill, SKILLS

# Run with:
# py -3.11 -m streamlit run app.py

st.set_page_config(page_title="Sports Form Analyzer", layout="wide")
st.title("Sports Form Analyzer")

# Hide Streamlit's default toolbar (Stop/Deploy/kebab) since we don't use it.
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Upload a short training clip and I'll extract pose, compute key metrics, "
    "flag issues based on the selected exercise, and recommend drills."
)

# Utility: rotate a frame by multiples of 90 degrees if user requests.
def rotate_frame(frame, degrees: int):
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# --- sidebar: params ---
with st.sidebar:
    st.header("Settings")
    exercise = st.selectbox("Exercise", sorted(list(SKILLS.keys())))
    target_fps = st.number_input("Target FPS (processing)", min_value=5, max_value=30, value=15, step=1)
    # Aux controls (if any) will be shown by the selected skill
    show_overlay = st.checkbox("Show annotated key frame", value=True)
    feedback_mode = st.selectbox(
        "Feedback mode",
        ["Use rule-based comments", "Angles only (AI decides)"],
        index=0
    )
    rotate_deg = st.selectbox("Rotate input clips", options=[0, 90, 180, 270], index=0)


skill = get_skill(exercise)

# Define upload slots per skill
upload_spec = []
if hasattr(skill, "get_upload_spec"):
    try:
        upload_spec = skill.get_upload_spec() or []
    except Exception:
        upload_spec = []
if not upload_spec:
    upload_spec = [
        {
            "key": "primary",
            "label": "Primary clip",
            "instruction": "Best angle for this exercise.",
            "required": True,
        }
    ]

# Render uploaders with instructions
uploaded_files = {}
for slot in upload_spec:
    col = st.container()
    with col:
        f = st.file_uploader(
            slot["label"], type=["mp4", "mov", "m4v", "avi"], key=f"uploader_{slot['key']}"
        )
        if slot.get("instruction"):
            st.caption(slot["instruction"])
        uploaded_files[slot["key"]] = f

# Validate: at least one required slot filled (or any slot if none are marked required)
required_keys = [s["key"] for s in upload_spec if s.get("required")]
if required_keys:
    has_required = any(uploaded_files.get(k) is not None for k in required_keys)
else:
    has_required = any(f is not None for f in uploaded_files.values())

if not has_required:
    st.info("Please upload at least the required clip to start.")
else:
    st.info("Processing video(s)... this may take a few seconds.")

    # Aux inputs via skill hook (optional)
    aux = None
    if hasattr(skill, "get_aux_sidebar"):
        try:
            aux = skill.get_aux_sidebar(st.sidebar)
        except Exception:
            aux = None

    # Load KB
    kb_path = Path("kb") / f"{exercise}.json"
    if kb_path.exists():
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    else:
        kb = {}

    # Process each uploaded clip
    attempts = []
    tmp_dir = tempfile.mkdtemp()
    for slot in upload_spec:
        file = uploaded_files.get(slot["key"])
        if not file:
            continue
        tmp_path = os.path.join(tmp_dir, file.name)
        with open(tmp_path, "wb") as f:
            f.write(file.read())

        frames, fps = load_video(tmp_path, target_fps=target_fps)
        if rotate_deg:
            frames = [rotate_frame(frm, rotate_deg) for frm in frames]
        if not frames:
            continue
        pose_seq = run_pose_on_frames(frames)
        pose_seq_smooth = exp_smooth_pose(pose_seq)
        if isinstance(aux, dict):
            aux_in = dict(aux)
        else:
            aux_in = {}
        aux_in["view"] = slot["key"]
        features = skill.extract_features(pose_seq_smooth, aux=aux_in)
        eval_res = skill.evaluate(features)
        recs = skill.recommend_drills(eval_res.get("issues", []), kb)

        attempts.append({
            "label": slot["label"],
            "frames": frames,
            "pose_seq": pose_seq_smooth,
            "features": features,
            "eval": eval_res,
            "recs": recs,
            "aux": aux_in,
        })

    if not attempts:
        st.error("No valid frames were read from the uploaded video(s).")
        st.stop()

    # LLM feedback over all attempts
    from core.llm_feedback import generate_feedback_multi
    mode = "kb" if feedback_mode == "Use KB drills" else "angles_only"
    coach_text = generate_feedback_multi(attempts, exercise, mode=mode)

    # --- layout ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("LLM Coaching Feedback")
        st.write(coach_text)

    with col2:
        if show_overlay:
            for att in attempts:
                frames = att["frames"]
                pose_seq_smooth = att["pose_seq"]
                features = att["features"]
                label = att["label"]

                if hasattr(skill, "select_key_frames"):
                    key_indices = skill.select_key_frames(features) or [0]
                elif hasattr(skill, "select_key_frame"):
                    key_indices = [int(skill.select_key_frame(features) or 0)]
                else:
                    key_indices = [int(features.get("key_frame_idx", features.get("peak_wrist_frame", 0)))]

                for key_idx in key_indices:
                    key_idx = int(max(0, min(len(frames) - 1, key_idx)))
                    frame = frames[key_idx]
                    frame_pose = pose_seq_smooth[key_idx]

                    if hasattr(skill, "get_overlay_labels"):
                        labels = skill.get_overlay_labels(features)
                    else:
                        labels = []
                        for k in [k for k in features.keys() if isinstance(features.get(k), (int, float))][:2]:
                            v = features.get(k)
                            labels.append(f"{k}={v:.2f}")

                    annot = render_pose_frame(frame, frame_pose, extra_labels=labels)

                    aux = att.get("aux")
                    if aux is not None and isinstance(aux, dict) and "bar_y" in aux:
                        annot = draw_bar_line(annot, int(aux["bar_y"]), color=(0, 0, 255), thickness=2)

                    annot_rgb = cv2.cvtColor(annot, cv2.COLOR_BGR2RGB)
                    st.image(
                    annot_rgb,
                    caption=f"{label} — key frame (#{key_idx})",
                    use_container_width=True,
                )
