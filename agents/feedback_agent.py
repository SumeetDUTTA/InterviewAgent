import sys
import os
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.llm import generate_with_mode
from core.domain_profiles import normalize_domain, get_domain_profile
from core.rubrics import RUBRIC_VERSION
from pydantic import BaseModel, Field


class InterviewFeedback(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    improvement_plan: list[str] = Field(default_factory=list)


def _validate_feedback_payload(payload: Any) -> dict:
    validated = InterviewFeedback.model_validate(payload)
    return validated.model_dump()


def _to_percent(score: Any) -> float:
    value = float(score or 0)
    if value <= 5:
        return value * 20.0
    return value


def _build_deterministic_findings(summary_data: dict):
    strengths = []
    weaknesses = []
    improvement_plan = []

    if summary_data["avg_depth_pct"] >= 75:
        strengths.append("Strong role-relevant depth across most answers.")
    elif summary_data["avg_depth_pct"] < 60:
        weaknesses.append("Depth is below expected level for this role.")
        improvement_plan.append(
            "Use concrete examples with decisions, trade-offs, and outcomes to strengthen depth."
        )

    if summary_data["avg_clarity_pct"] >= 75:
        strengths.append("Strong clarity and structure in responses.")
    elif summary_data["avg_clarity_pct"] < 60:
        weaknesses.append("Response clarity needs improvement.")
        improvement_plan.append(
            "Use a clear structure (context, action, result) in each answer."
        )

    if summary_data["avg_confidence_pct"] >= 75:
        strengths.append("Good confidence and ownership in explanations.")
    elif summary_data["avg_confidence_pct"] < 60:
        weaknesses.append("Confidence is inconsistent in key responses.")
        improvement_plan.append(
            "State decisions more directly and reduce hedging language."
        )

    if summary_data["avg_speech_clarity_pct"] >= 75:
        strengths.append("Speech delivery is generally clear and understandable.")
    elif summary_data["avg_speech_clarity_pct"] < 60:
        weaknesses.append("Speech clarity is below target.")
        improvement_plan.append(
            "Practice concise spoken delivery with fewer filler phrases."
        )

    if summary_data["followups_triggered"] > 3:
        weaknesses.append(
            "Frequent follow-ups indicate some responses lacked initial completeness."
        )
        improvement_plan.append(
            "Answer the main question first, then add one concrete example and impact."
        )

    if not strengths:
        strengths.append("Shows consistent engagement across interview rounds.")

    if not weaknesses:
        weaknesses.append(
            "No major weaknesses detected from current interview metrics."
        )

    if not improvement_plan:
        improvement_plan.append(
            "Maintain consistency by continuing structured, concise, evidence-backed answers."
        )

    return strengths, weaknesses, improvement_plan


def _merge_unique(primary, fallback, limit=5):
    merged = []
    for item in (primary or []) + (fallback or []):
        txt = str(item).strip()
        if txt and txt not in merged:
            merged.append(txt)
        if len(merged) >= limit:
            break
    return merged

def _get_speech_score(entry: dict) -> float:
    return float(entry.get("speech_clarity", entry.get("clarity", 0)) or 0)


def generate_feedback(all_scores, domain="industry"):

    resolved_domain = normalize_domain(domain)
    profile = get_domain_profile(resolved_domain)
    labels = profile.get("score_labels", {})

    if not all_scores:
        return {"error": "Interview incomplete. Not enough data."}

    total_questions = len(all_scores)
    avg_depth = sum(float(s.get("depth", 0)) for s in all_scores) / total_questions
    avg_clarity = sum(float(s.get("clarity", 0)) for s in all_scores) / total_questions
    avg_conf = sum(float(s.get("confidence", 0)) for s in all_scores) / total_questions
    avg_speech = sum(_get_speech_score(s) for s in all_scores) / total_questions

    weak_depth = sum(1 for s in all_scores if _to_percent(s.get("depth", 0)) < 50)
    weak_clarity = sum(1 for s in all_scores if _to_percent(s.get("clarity", 0)) < 50)
    weak_conf = sum(1 for s in all_scores if _to_percent(s.get("confidence", 0)) < 50)
    weak_speech = sum(1 for s in all_scores if _to_percent(_get_speech_score(s)) < 50)

    followups_triggered = sum(1 for s in all_scores if s.get("needs_followup"))
    short_answers = sum(1 for s in all_scores if int(s.get("answer_length", 0)) < 20)

    summary_data = {
        "domain": resolved_domain,
        "rubric_version": RUBRIC_VERSION,
        "total_questions": total_questions,
        "avg_depth": round(avg_depth, 2),
        "avg_clarity": round(avg_clarity, 2),
        "avg_confidence": round(avg_conf, 2),
        "avg_speech_clarity": round(avg_speech, 2),
        "avg_depth_pct": round(_to_percent(avg_depth), 2),
        "avg_clarity_pct": round(_to_percent(avg_clarity), 2),
        "avg_confidence_pct": round(_to_percent(avg_conf), 2),
        "avg_speech_clarity_pct": round(_to_percent(avg_speech), 2),
        "weak_depth_count": weak_depth,
        "weak_clarity_count": weak_clarity,
        "weak_confidence_count": weak_conf,
        "weak_speech_clarity_count": weak_speech,
        "followups_triggered": followups_triggered,
        "short_answers": short_answers,
        "score_labels": labels,
    }

    deterministic_strengths, deterministic_weaknesses, deterministic_plan = (
        _build_deterministic_findings(summary_data)
    )

    system_prompt = """
    You are an interview feedback assistant for universal role interviews.

    Use the provided metrics to produce concise structured feedback.

    Rules:
    - Only mention a weakness if the metric data supports it.
    - Do not claim weak speech clarity when avg_speech_clarity_pct >= 75 and weak_speech_clarity_count <= 1.
    - Do not claim weak confidence when avg_confidence_pct >= 75 and weak_confidence_count <= 1.
    - Keep advice specific, actionable, and conservative.

    Return JSON with exactly these keys:
    - strengths: list of strings
    - weaknesses: list of strings
    - improvement_plan: list of strings
    """

    user_prompt = f"""
    Performance Metrics:
    {summary_data}

    Generate structured feedback.
    """

    llm_feedback = generate_with_mode(
        "feedback",
        system_prompt,
        user_prompt,
        expect_json=True,
        retries=2,
        validator=_validate_feedback_payload,
    )

    if not isinstance(llm_feedback, dict) or llm_feedback.get("error"):
        llm_feedback = {"strengths": [], "weaknesses": [], "improvement_plan": []}

    merged_strengths = _merge_unique(
        llm_feedback.get("strengths", []), deterministic_strengths, limit=5
    )
    merged_weaknesses = _merge_unique(
        llm_feedback.get("weaknesses", []), deterministic_weaknesses, limit=5
    )
    merged_plan = _merge_unique(
        llm_feedback.get("improvement_plan", []), deterministic_plan, limit=5
    )

    if (
        summary_data["avg_speech_clarity_pct"] >= 75
        and summary_data["weak_speech_clarity_count"] <= 1
    ):
        merged_weaknesses = [
            w
            for w in merged_weaknesses
            if "speech" not in w.lower() and "spoken" not in w.lower()
        ]

    if (
        summary_data["avg_confidence_pct"] >= 75
        and summary_data["weak_confidence_count"] <= 1
    ):
        merged_weaknesses = [
            w for w in merged_weaknesses if "confidence" not in w.lower()
        ]

    if not merged_weaknesses:
        merged_weaknesses = [
            "No major weaknesses detected from current interview metrics."
        ]

    return {
        "metrics": summary_data,
        "structured_feedback": {
            "strengths": merged_strengths,
            "weaknesses": merged_weaknesses,
            "improvement_plan": merged_plan,
        },
    }
