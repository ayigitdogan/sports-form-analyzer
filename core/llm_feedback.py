# core/llm_feedback.py

import os
from typing import Dict, List, Any, Optional


def _summarize_metrics_for_prompt(features: Dict[str, Any], exercise: str) -> str:
    """Try skill-specific summary; else fall back to generic top features."""
    try:
        from exercises.registry import get_skill
        skill = get_skill(exercise)
        if hasattr(skill, "summarize_for_prompt"):
            txt = skill.summarize_for_prompt(features)
            if isinstance(txt, str) and txt.strip():
                return txt
    except Exception:
        pass

    pairs = []
    for k, v in features.items():
        if isinstance(v, (int, float)):
            pairs.append(f"{k}={v:.2f}")
        if len(pairs) >= 3:
            break
    return ", ".join(pairs) if pairs else "n/a"


def build_prompt(features: Dict[str, Any],
                 issues: List[str],
                 recs: List[Dict[str, Any]],
                 exercise: str) -> str:
    issues_txt = ", ".join(issues) if issues else "none"
    drills: List[str] = []
    for r in recs:
        drills.extend(r.get("drills", []))
    drills_txt = "; ".join(drills[:4]) if drills else "none"
    metrics_txt = _summarize_metrics_for_prompt(features, exercise)
    return (
        f"Act as an expert strength and conditioning coach. Analyze this {exercise} attempt using the details below.\n"
        f"Metrics: {metrics_txt}.\n"
        f"Detected issues: {issues_txt}.\n"
        f"Available drills: {drills_txt}.\n"
        "Do not label the form as simply good or bad. Describe what the athlete does well, "
        "what tends to break down, and how posture/form could improve. Use simple language.\n"
        "Do not mention angles, ratios, or any numbers.\n"
        "Write 3 concise sentences: 1) quick snapshot with at least one positive, "
        "2) main tendency to address and why, 3) one drill or cue to prioritize. Keep it specific and under 90 words."
    )


def build_prompt_angles(features: Dict[str, Any], exercise: str) -> str:
    metrics_txt = _summarize_metrics_for_prompt(features, exercise)
    return (
        f"Act as an expert strength and conditioning coach. Analyze this {exercise} attempt using only angle summaries "
        "from multiple keyframes.\n"
        f"Angles: {metrics_txt}.\n"
        "Do not label the form as simply good or bad. Describe what the athlete does well, "
        "what tends to break down, and how posture/form could improve. Use simple language.\n"
        "Do not mention angles, ratios, or any numbers.\n"
        "Write 3 concise sentences: 1) quick snapshot with at least one positive, "
        "2) main tendency to address and why, 3) one drill or cue to prioritize. Keep it specific and under 90 words."
    )


def generate_feedback(features: Dict[str, Any],
                      eval_res: Dict[str, Any],
                      recs: List[Dict[str, Any]],
                      exercise: str = "squat",
                      mode: str = "kb") -> str:
    issues = eval_res.get("issues", [])
    prompt = build_prompt_angles(features, exercise) if mode == "angles_only" else build_prompt(features, issues, recs, exercise)
    text = _call_gemini(prompt)
    if text:
        return text
    return _template_feedback(features, issues, recs, exercise, mode=mode)


def build_prompt_multi(attempts: List[Dict[str, Any]], exercise: str, mode: str = "kb") -> str:
    """Compose a multi-clip prompt. attempts: [{label, features, eval, recs}]"""
    if mode == "angles_only":
        lines = [
            f"Act as an expert strength and conditioning coach. Analyze this {exercise} session across multiple clips using only angle summaries from multiple keyframes.",
            "For each clip you get angle summaries only; do not use predefined issue labels or drills."
        ]
    else:
        lines = [
            f"Act as an expert strength and conditioning coach. Analyze this {exercise} session across multiple clips.",
            "For each clip you get metrics and detected issues."
        ]
    all_issues: List[str] = []
    all_drills: List[str] = []
    for i, att in enumerate(attempts, start=1):
        label = att.get("label") or f"clip_{i}"
        feats = att.get("features", {})
        ev = att.get("eval", {})
        recs = att.get("recs", [])
        metrics_txt = _summarize_metrics_for_prompt(feats, exercise)
        view = "unknown"
        aux = att.get("aux")
        if isinstance(aux, dict):
            view = aux.get("view", "unknown")
        issues = ev.get("issues", [])
        all_issues.extend(issues)
        for r in recs:
            all_drills.extend(r.get("drills", []))
        if mode == "angles_only":
            lines.append(f"- {label} (view={view}): angles={metrics_txt}.")
        else:
            lines.append(f"- {label} (view={view}): metrics={metrics_txt}; issues={', '.join(issues) if issues else 'none'}.")
    if all_drills:
        all_drills = list(dict.fromkeys(all_drills))  # de-duplicate, preserve order
    issues_txt = ", ".join(list(dict.fromkeys(all_issues))) if all_issues else "none"
    drills_txt = "; ".join(all_drills[:6]) if all_drills else "none"
    if mode != "angles_only":
        lines.append(f"Overall issues: {issues_txt}.")
        lines.append(f"Candidate drills: {drills_txt}.")
    lines.append(
        "Do not label the form as simply good or bad. Describe positives, shared tendencies, and how posture/form could improve."
        " Use simple language. Do not mention angles, ratios, or any numbers."
    )
    if mode == "angles_only":
        lines.append("Write 3 concise sentences total: 1) overall snapshot with at least one positive, 2) main shared tendency to improve, 3) 1-2 drills or cues to prioritize. Keep it under 90 words.")
    else:
        lines.append("Write 3 concise sentences total: 1) overall snapshot with at least one positive, 2) main shared tendency to improve, 3) 1-2 drills to prioritize. Keep it under 90 words.")
    return "\n".join(lines)


def generate_feedback_multi(attempts: List[Dict[str, Any]],
                            exercise: str,
                            mode: str = "kb") -> str:
    prompt = build_prompt_multi(attempts, exercise, mode=mode)
    text = _call_gemini(prompt)
    if text:
        return text
    return _template_feedback_multi(attempts, exercise, mode=mode)


def _template_feedback_multi(attempts: List[Dict[str, Any]], exercise: str, mode: str = "kb") -> str:
    blocks = []
    for i, att in enumerate(attempts, start=1):
        label = att.get("label") or f"clip_{i}"
        feats = att.get("features", {})
        pairs = []
        for k, v in feats.items():
            if isinstance(v, (int, float)):
                pairs.append(f"{k}={v:.1f}")
            if len(pairs) >= 2:
                break
        blocks.append(f"{label}: {', '.join(pairs) if pairs else 'n/a'}")
    all_issues: List[str] = []
    all_drills: List[str] = []
    for att in attempts:
        ev = att.get("eval", {})
        all_issues.extend(ev.get("issues", []))
        for r in att.get("recs", []):
            all_drills.extend(r.get("drills", [])[:1])
    issues_txt = ", ".join(list(dict.fromkeys(all_issues))) if all_issues else "none"
    drills_txt = "; ".join(list(dict.fromkeys(all_drills))[:3]) if all_drills else "practice fundamentals"
    if mode == "angles_only":
        return " ".join([
            f"{exercise} session across {len(attempts)} clip(s):",
            " | ".join(blocks) + ".",
            "Focus on consistent positions and smooth control.",
            "Prioritize one drill or cue that improves control and range.",
        ])
    return " ".join([
        f"{exercise} session across {len(attempts)} clip(s):",
        " | ".join(blocks) + ".",
        f"Key improvements: {issues_txt}.",
        f"Try: {drills_txt}.",
    ])


def _call_gemini(prompt: str) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return (getattr(resp, "text", "") or "").strip() or None
    except Exception as e:
        print("[llm_feedback] Gemini error:", e)
        return None


def _template_feedback(features, issues, recs, exercise, mode: str = "kb") -> str:
    # generic, 3 short sentences
    metrics_pairs = []
    for k, v in features.items():
        if isinstance(v, (int, float)):
            metrics_pairs.append(f"{k}={v:.1f}")
        if len(metrics_pairs) >= 2:
            break
    summary = f"{exercise} attempt. Snapshot: consistent effort and control."

    if mode == "angles_only":
        improve = "Keep the strong positions you already have, and smooth out any rushed or uneven parts."
    else:
        improve = "Keep the strong positions you already have, and clean up the main tendency that shows up most."

    drills = []
    for r in recs:
        drills.extend(r.get("drills", [])[:1])
    drill_txt = "Try: " + "; ".join(drills) + "." if drills else "Try one simple drill that reinforces control."

    return " ".join([summary, improve, drill_txt])
