import json
import re
from typing import List, Literal
from pydantic import BaseModel
from utils.llm import generate_with_mode
from core.domain_profiles import get_domain_profile, normalize_domain


def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return None


class InterviewRound(BaseModel):
    type: Literal["Technical", "Behavioral", "Case Study", "Situational", "Domain-Specific", "Warm-up"]
    focus_areas: List[str]
    stress_test_areas: List[str]
    competencies: List[str] = []


class InterviewPlan(BaseModel):
    domain: str
    difficulty: Literal["BEGINNER", "INTERMEDIATE", "ADVANCED"]
    rounds: List[InterviewRound]


def _safe_list(items, fallback):
    cleaned = [str(x).strip() for x in (items or []) if str(x).strip()]
    return cleaned if cleaned else fallback


def _difficulty_from_readiness(readiness_score: int) -> str:
    if readiness_score < 40:
        return "BEGINNER"
    if readiness_score <= 70:
        return "INTERMEDIATE"
    return "ADVANCED"


def _fallback_interview_plan(
    resolved_domain: str,
    profile: dict,
    strong_skills: list,
    partial_skills: list,
    skill_gaps: list,
    competencies: list,
    readiness_score: int,
):
    default_focus = profile.get("default_focus_areas", ["Core Role Fundamentals"])
    strong = _safe_list(strong_skills, default_focus)
    partial = _safe_list(partial_skills, ["Communication and Collaboration"])
    gaps = _safe_list(skill_gaps, ["Role-Specific Problem Solving"])

    rounds = [
        InterviewRound(
            type="Warm-up",
            focus_areas=["Candidate Introduction", "Experience Overview", "Career Aspirations"],
            stress_test_areas=[],
            competencies=[],
        ),
        InterviewRound(
            type="Domain-Specific",
            focus_areas=strong[:2],
            stress_test_areas=gaps[:2],
            competencies=(competencies or [])[:2],
        ),
        InterviewRound(
            type="Behavioral",
            focus_areas=partial[:2],
            stress_test_areas=["Communication", "Decision Making"],
            competencies=(competencies or [])[:2],
        ),
        InterviewRound(
            type="Situational",
            focus_areas=gaps[:2],
            stress_test_areas=["Trade-offs", "Risk Management"],
            competencies=(competencies or [])[:2],
        ),
    ]

    return InterviewPlan(
        domain=resolved_domain,
        difficulty=_difficulty_from_readiness(int(readiness_score)),
        rounds=rounds,
    ).model_dump()


def generate_interview_plan(
    strong_skills,
    skill_gaps,
    experience_years,
    domain: str = "industry",
    competencies=None,
    partial_skills=None,
    readiness_score: int | None = None,
):
    resolved_domain = normalize_domain(domain)
    profile = get_domain_profile(resolved_domain)
    round_templates = profile.get("round_templates", [])
    competencies = competencies or []
    partial_skills = partial_skills or []
    readiness_score = int(readiness_score if readiness_score is not None else 0)

    system_prompt = """
    You are a universal professional interviewer.

    You conduct structured, role-specific interviews across any domain.

    Create a detailed interview progression plan.

    Responsibilities:
    - Adapt interview type based on readiness score
    - Adjust difficulty based on experience
    - Include appropriate round types for the role context
    - Probe skill gaps intelligently

    Scoring Rules:
    - readiness < 40 -> BEGINNER
    - readiness 40-70 -> INTERMEDIATE
    - readiness > 70 -> ADVANCED

    Strict output rules:
    - Return valid JSON only.
    - difficulty must be one of: BEGINNER, INTERMEDIATE, ADVANCED
    - rounds must be a non-empty list.
    - each round type must be one of: Technical, Behavioral, Case Study, Situational, Domain-Specific
    - each round must include at least one focus_areas item.
    - stress_test_areas can be empty but must be a list.
    - Do not assume software/backend context unless inputs indicate it.
    """

    user_prompt = f"""
    Domain: {resolved_domain}
    Domain round templates: {round_templates}
    Strong skills: {strong_skills}
    Partial skills: {partial_skills}
    Skill gaps: {skill_gaps}
    Required competencies: {competencies}
    Readiness score: {readiness_score}
    Experience years: {experience_years}
    """

    print("Generating interview plan with:")
    print("Domain:", resolved_domain)
    print("Strong skills:", strong_skills)
    print("Partial skills:", partial_skills)
    print("Skill gaps:", skill_gaps)
    print("Competencies:", competencies)
    print("Readiness score:", readiness_score)
    print("Experience years:", experience_years)

    response = generate_with_mode(
        "planner", system_prompt, user_prompt, expect_json=True, retries=3
    )

    parsed = response if isinstance(response, dict) else extract_json(response)
    if not parsed:
        print("⚠ Raw Model Output:\n", response)
        return _fallback_interview_plan(
            resolved_domain,
            profile,
            strong_skills,
            partial_skills,
            skill_gaps,
            competencies,
            readiness_score,
        )

    valid_types = {"Technical", "Behavioral", "Case Study", "Situational", "Domain-Specific"}
    rounds_raw = parsed.get("rounds", []) if isinstance(parsed, dict) else []
    validated_rounds = []
    for rnd in rounds_raw:
        if not isinstance(rnd, dict):
            continue
        round_type = rnd.get("type")
        if round_type not in valid_types:
            continue
        validated_rounds.append(
            InterviewRound(
                type=round_type,
                focus_areas=_safe_list(rnd.get("focus_areas", []), profile.get("default_focus_areas", ["Core Role Fundamentals"])),
                stress_test_areas=_safe_list(rnd.get("stress_test_areas", []), []),
                competencies=_safe_list(rnd.get("competencies", []), []),
            )
        )

    if not validated_rounds:
        return _fallback_interview_plan(
            resolved_domain,
            profile,
            strong_skills,
            partial_skills,
            skill_gaps,
            competencies,
            readiness_score,
        )

    warmup_round = InterviewRound(
        type="Warm-up",
        focus_areas=["Candidate Introduction", "Experience Overview", "Career Aspirations"],
        stress_test_areas=[],
        competencies=[],
    )

    model_difficulty = str(parsed.get("difficulty", _difficulty_from_readiness(readiness_score))).upper()
    if model_difficulty not in {"BEGINNER", "INTERMEDIATE", "ADVANCED"}:
        model_difficulty = _difficulty_from_readiness(readiness_score)

    final_plan = InterviewPlan(
        domain=normalize_domain(parsed.get("domain", resolved_domain)),
        difficulty=model_difficulty,
        rounds=[warmup_round] + validated_rounds,
    )
    return final_plan.model_dump()
