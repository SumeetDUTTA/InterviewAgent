from core.domain_profiles import normalize_domain
from utils.llm import generate_with_mode
import re


def _to_normalized_set(values):
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _extract_required_years(experience_required: str) -> float | None:
    if not experience_required:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(experience_required))
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _find_partial_matches(requirements: set[str], candidates: set[str]) -> list[str]:
    partial = set()
    for requirement in requirements:
        for candidate in candidates:
            if requirement == candidate:
                continue
            if requirement in candidate or candidate in requirement:
                partial.add(requirement)
    return sorted(list(partial))


def _compute_readiness_score(
    strong_count: int,
    partial_count: int,
    missing_count: int,
    resume_experience_years: float,
    required_experience_years: float | None,
) -> int:
    total_signal = strong_count + partial_count + missing_count
    if total_signal == 0:
        return 0

    coverage = (strong_count + (0.5 * partial_count)) / total_signal
    score = coverage * 100

    if required_experience_years is not None:
        if resume_experience_years >= required_experience_years:
            score += 8
        elif resume_experience_years >= max(0, required_experience_years - 1):
            score += 2
        else:
            score -= min(20, (required_experience_years - resume_experience_years) * 6)

    return max(0, min(100, int(round(score))))


def _coerce_skill_list(payload: dict, key: str) -> list[str]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _validate_gap_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Gap payload must be a dictionary")

    readiness = payload.get("readiness_score", 0)
    try:
        readiness = int(readiness)
    except Exception as exc:
        raise ValueError("readiness_score must be an integer") from exc

    return {
        "strong_skills": _coerce_skill_list(payload, "strong_skills"),
        "partial_match_skills": _coerce_skill_list(payload, "partial_match_skills"),
        "skill_gaps": _coerce_skill_list(payload, "skill_gaps"),
        "resume_experience_years": float(payload.get("resume_experience_years", 0) or 0),
        "experience_required": str(payload.get("experience_required", "Not specified") or "Not specified"),
        "readiness_score": max(0, min(100, readiness)),
    }


def _refine_gap_with_llm(
    *,
    candidate_pool: set[str],
    requirement_pool: set[str],
    resume_data: dict,
    jd_data: dict,
    baseline: dict,
) -> dict | None:
    system_prompt = """
    You are a hiring analyst.

    Compare candidate resume skills with job description requirements.

    Identify:
    - strong_skills
    - partial_match_skills
    - skill_gaps
    - resume_experience_years
    - experience_required
    - readiness_score (0-100 integer)

    SCORING LOGIC:
    - Strong alignment -> higher score
    - Missing core skills -> reduce score
    - Experience mismatch -> reduce score
    - Be realistic and conservative

    RULES:
    - Do not hallucinate
    - If unsure, classify conservatively
    - Return structured output only
    - JSON only
    """

    user_prompt = f"""
    Candidate pool (normalized): {sorted(list(candidate_pool))}
    Requirement pool (normalized): {sorted(list(requirement_pool))}

    Resume skills: {resume_data.get("skills", [])}
    Resume tools: {resume_data.get("tools", [])}
    Resume research areas: {resume_data.get("research_areas", [])}
    Resume policy experience: {resume_data.get("policy_experience", [])}
    Resume experience years: {resume_data.get("experience_years", 0)}

    JD required skills: {jd_data.get("required_skills", [])}
    JD required tools: {jd_data.get("required_tools", [])}
    JD required competencies: {jd_data.get("required_competencies", [])}
    JD experience required: {jd_data.get("experience_required", "")}

    Deterministic baseline:
    - strong_skills: {baseline.get("strong_skills", [])}
    - partial_match_skills: {baseline.get("partial_match_skills", [])}
    - skill_gaps: {baseline.get("skill_gaps", [])}
    - readiness_score: {baseline.get("readiness_score", 0)}
    """

    llm_result = generate_with_mode(
        mode="planner",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expect_json=True,
        retries=2,
        validator=_validate_gap_payload,
    )

    if not isinstance(llm_result, dict) or llm_result.get("error"):
        return None

    llm_strong = _to_normalized_set(llm_result.get("strong_skills", []))
    llm_partial = _to_normalized_set(llm_result.get("partial_match_skills", []))
    llm_gaps = _to_normalized_set(llm_result.get("skill_gaps", []))

    valid_strong = (llm_strong & candidate_pool) & requirement_pool
    valid_partial = {
        skill
        for skill in llm_partial
        if skill in requirement_pool and skill not in valid_strong
    }
    valid_gaps = {
        skill
        for skill in llm_gaps
        if skill in requirement_pool and skill not in valid_strong
    }

    baseline_score = int(baseline.get("readiness_score", 0))
    llm_score = int(llm_result.get("readiness_score", baseline_score))
    blended_score = max(0, min(100, int(round((0.6 * baseline_score) + (0.4 * llm_score)))))

    return {
        "strong_skills": sorted(list(valid_strong)),
        "partial_match_skills": sorted(list(valid_partial)),
        "skill_gaps": sorted(list(valid_gaps)),
        "resume_experience_years": baseline.get("resume_experience_years", 0),
        "experience_required": baseline.get("experience_required", "Not specified"),
        "readiness_score": blended_score,
    }


def analyze_gap(resume_data, jd_data, domain: str | None = None, use_llm_refinement: bool = True):
    resolved_domain = normalize_domain(domain or jd_data.get("domain") or resume_data.get("domain"))

    resume_skills = _to_normalized_set(resume_data.get("skills", []))
    resume_tools = _to_normalized_set(resume_data.get("tools", []))
    resume_research = _to_normalized_set(resume_data.get("research_areas", []))
    resume_policy = _to_normalized_set(resume_data.get("policy_experience", []))

    jd_skills = _to_normalized_set(jd_data.get("required_skills", []))
    jd_tools = _to_normalized_set(jd_data.get("required_tools", []))
    jd_competencies = _to_normalized_set(jd_data.get("required_competencies", []))

    candidate_pool = resume_skills | resume_tools | resume_research | resume_policy
    core_requirement_pool = jd_skills | jd_tools

    strong = sorted(list(candidate_pool & core_requirement_pool))
    gaps = sorted(list(core_requirement_pool - candidate_pool))
    partial = _find_partial_matches(core_requirement_pool - set(strong), candidate_pool)

    resume_experience_years = float(resume_data.get("experience_years", 0) or 0)
    experience_required = str(jd_data.get("experience_required", "Not specified") or "Not specified")
    required_years = _extract_required_years(experience_required)

    readiness_score = _compute_readiness_score(
        strong_count=len(strong),
        partial_count=len(partial),
        missing_count=len(gaps),
        resume_experience_years=resume_experience_years,
        required_experience_years=required_years,
    )

    baseline = {
        "strong_skills": strong,
        "partial_match_skills": partial,
        "skill_gaps": gaps,
        "resume_experience_years": resume_experience_years,
        "experience_required": experience_required,
        "readiness_score": readiness_score,
    }

    refined = None
    if use_llm_refinement:
        try:
            refined = _refine_gap_with_llm(
                candidate_pool=candidate_pool,
                requirement_pool=core_requirement_pool,
                resume_data=resume_data,
                jd_data=jd_data,
                baseline=baseline,
            )
        except Exception:
            refined = None

    final = refined or baseline
    final_strong = _to_normalized_set(final.get("strong_skills", []))

    return {
        "domain": resolved_domain,
        "strong_skills": final.get("strong_skills", []),
        "partial_match_skills": final.get("partial_match_skills", []),
        "skill_gaps": final.get("skill_gaps", []),
        "resume_experience_years": float(final.get("resume_experience_years", resume_experience_years) or 0),
        "experience_required": str(final.get("experience_required", experience_required) or experience_required),
        "readiness_score": int(final.get("readiness_score", readiness_score)),
        "matched_competencies": sorted(list(jd_competencies & candidate_pool)),
        "missing_competencies": sorted(list(_to_normalized_set(jd_data.get("required_competencies", [])) - candidate_pool)),
    }
