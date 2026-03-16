import sys
import os
from typing import List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.llm import generate_with_mode, extract_json
import json
from pydantic import BaseModel, Field
from core.domain_profiles import infer_domain_from_text, normalize_domain


class JDRequirements(BaseModel):
    domain: str = "industry"
    required_skills: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    required_competencies: List[str] = Field(default_factory=list)
    experience_required: str = "Not specified"
    evaluation_focus: List[str] = Field(default_factory=list)


def _validate_jd_payload(payload: Any) -> dict:
    validated = JDRequirements.model_validate(payload)
    return validated.model_dump()


def parse_jd(jd_text: str, domain: str | None = None, use_reviewer: bool = False):
    inferred_domain = normalize_domain(domain) if domain else infer_domain_from_text(jd_text)
    system_prompt = f"""
    Extract job description requirements.
    Return JSON only with:
    - domain (industry/government/phd)
    - required_skills (list)
    - required_tools (list)
    - required_competencies (list)
    - experience_required (string)
    - evaluation_focus (list)

    STRICT RULES:
    - Output valid JSON only
    - No explanation
    - No markdown
    - No text before or after JSON
    - If information is missing, return empty lists or "Not specified" for experience_required
    - Avoid duplicates
    - Focus on extracting actionable information that can guide interview question generation
    - Do not hallucinate information not present in the Job Description
    - Prioritize extracting information relevant to the inferred domain: {inferred_domain}
    """

    print("Parsed JD Text:\n", jd_text[:500], "...\n")  # Debug print

    user_prompt = f"Domain hint: {inferred_domain}\n\nJD:\n{jd_text}"
    response = generate_with_mode(
        "planner",
        system_prompt,
        user_prompt,
        expect_json=True,
        retries=3,
        validator=_validate_jd_payload,
    )

    print("Raw Model Output:\n", response)  # Debug print

    try:
        raw_domain = None

        if isinstance(response, dict) and response.get("error") and response.get("raw"):
            recovered = extract_json(response.get("raw", ""))
            if recovered is not None:
                response = recovered

        if isinstance(response, dict):
            raw_domain = response.get("domain")
            parsed = _validate_jd_payload(response)
        else:
            payload = json.loads(response)
            raw_domain = payload.get("domain") if isinstance(payload, dict) else None
            parsed = _validate_jd_payload(payload)
        parsed["domain"] = normalize_domain(raw_domain or inferred_domain)

        if use_reviewer:
            parsed = jd_reviewer_pass(
                jd_text=jd_text,
                planner_output=parsed,
                domain=parsed["domain"],
            )

        return parsed
    except Exception as e:
        return {
            "error": f"Invalid or incompatible JSON from model: {str(e)}",
            "domain": inferred_domain,
            "required_skills": [],
            "required_tools": [],
            "required_competencies": [],
            "experience_required": "Not specified",
            "evaluation_focus": [],
        }


def jd_reviewer_pass(jd_text: str, planner_output: dict, domain: str | None = None):
    resolved_domain = normalize_domain(domain or planner_output.get("domain"))
    review_prompt = f"""
    You are a strict validation system for JD extraction.

    Validate and correct extracted job requirements conservatively.

    Rules:
    - Remove hallucinated skills/tools not mentioned or clearly implied.
    - Remove duplicates.
    - Ensure experience_required aligns with the JD text.
    - Keep only realistic, actionable competencies.
    - Preserve domain relevance for: {resolved_domain}
    - Output valid JSON only and match the required structure exactly.
    """

    user_prompt = f"""
    Domain: {resolved_domain}

    JOB DESCRIPTION:
    {jd_text}

    EXTRACTED:
    {json.dumps(planner_output, ensure_ascii=False)}
    """

    reviewed = generate_with_mode(
        mode="reviewer",
        system_prompt=review_prompt,
        user_prompt=user_prompt,
        expect_json=True,
        retries=2,
        validator=_validate_jd_payload,
    )

    if isinstance(reviewed, dict):
        reviewed["domain"] = resolved_domain
        return reviewed

    try:
        parsed = json.loads(reviewed)
        parsed["domain"] = resolved_domain
        return _validate_jd_payload(parsed)
    except Exception:
        planner_output["domain"] = resolved_domain
        return planner_output


def robust_parse_jd(jd_text: str, domain: str | None = None):
    inferred_domain = normalize_domain(domain) if domain else infer_domain_from_text(jd_text)
    try:
        planner_output = parse_jd(
            jd_text=jd_text,
            domain=inferred_domain,
            use_reviewer=False,
        )
        reviewed_output = jd_reviewer_pass(
            jd_text=jd_text,
            planner_output=planner_output,
            domain=inferred_domain,
        )
        return reviewed_output
    except Exception as e:
        raise RuntimeError(f"JD parsing pipeline failed: {e}")
