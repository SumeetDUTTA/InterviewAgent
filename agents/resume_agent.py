import sys
import os
from typing import List, Optional, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.llm import generate_with_mode, extract_json
import json
from pydantic import BaseModel, Field
from core.domain_profiles import normalize_domain


class Project(BaseModel):
    name: str = ""
    description: Optional[str] = ""


class EducationEntry(BaseModel):
    institution: str = ""
    degree: str = ""
    duration: str = ""
    cgpa: Optional[str] = ""
    percentage: Optional[str] = ""


class CertificationEntry(BaseModel):
    name: str = ""
    duration: Optional[str] = ""
    provider: Optional[str] = ""


class ResumeData(BaseModel):
    domain: str = "industry"
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    experience_years: float = 0
    projects: List[Project] = Field(default_factory=list)
    education: List[str | EducationEntry] = Field(default_factory=list)
    certifications: List[str | CertificationEntry] = Field(default_factory=list)
    publications: List[str | dict] = Field(default_factory=list)
    research_areas: List[str | dict] = Field(default_factory=list)
    policy_experience: List[str | dict] = Field(default_factory=list)
    summary: str = ""


def _validate_resume_payload(payload: Any) -> dict:
    validated = ResumeData.model_validate(payload)
    return validated.model_dump()


def parse_resume(resume_text: str, domain: str = "industry", use_reviewer: bool = False):
    normalized_domain = normalize_domain(domain)
    system_prompt = f"""
    You are an expert Resume Scrutinizer.
    Extract structured resume information.
    Return JSON only with:
    - skills (list)
    - tools (list)
    - experience_years (number)
    - projects (list)
    - education (list)
    - certifications (list)
    - publications (list)
    - research_areas (list)
    - policy_experience (list)
    - summary (string)

    STRICT RULES:
    - Output valid JSON only
    - No explanation
    - No markdown
    - No text before or after JSON
    - If information is missing, return empty lists or 0 for experience_years
    - Avoid duplicates
    - Do not hallucinate information not present in the resume
    - Focus on extracting information relevant to the domain: {normalized_domain}
    """
    print("Parsed Resume Text:\n", resume_text[:500], "...\n")  # Debug print

    user_prompt = f"Domain: {normalized_domain}\n\nResume:\n{resume_text}"
    response = generate_with_mode(
        "planner",
        system_prompt,
        user_prompt,
        expect_json=True,
        retries=3,
        validator=_validate_resume_payload,
    )

    print("Raw Model Output:\n", response)  # Debug print

    try:
        if isinstance(response, dict) and response.get("error") and response.get("raw"):
            recovered = extract_json(response.get("raw", ""))
            if recovered is not None:
                response = recovered

        if isinstance(response, dict):
            parsed = _validate_resume_payload(response)
        else:
            parsed = _validate_resume_payload(json.loads(response))
        parsed["domain"] = normalized_domain

        if use_reviewer:
            parsed = resume_reviewer_pass(
                resume_text=resume_text,
                planner_output=parsed,
                domain=normalized_domain,
            )

        return parsed
    except Exception as e:
        return {
            "error": f"Invalid or incompatible JSON from model: {str(e)}",
            "domain": normalized_domain,
            "skills": [],
            "tools": [],
            "experience_years": 0,
            "projects": [],
        }


def resume_reviewer_pass(resume_text: str, planner_output: dict, domain: str = "industry"):
    normalized_domain = normalize_domain(domain)
    review_prompt = f"""
    You are a strict validation system for resume extraction.

    Validate and correct extracted resume data conservatively.

    Rules:
    - Remove hallucinated skills/tools not clearly mentioned.
    - Remove duplicates.
    - Ensure experience_years is realistic.
    - Keep only actual projects mentioned.
    - Preserve domain relevance for: {normalized_domain}
    - Output valid JSON only and match the required structure exactly.
    """

    user_prompt = f"""
    Domain: {normalized_domain}

    RESUME:
    {resume_text}

    EXTRACTED:
    {json.dumps(planner_output, ensure_ascii=False)}
    """

    reviewed = generate_with_mode(
        mode="reviewer",
        system_prompt=review_prompt,
        user_prompt=user_prompt,
        expect_json=True,
        retries=2,
        validator=_validate_resume_payload,
    )

    if isinstance(reviewed, dict):
        reviewed["domain"] = normalized_domain
        return reviewed

    try:
        parsed = json.loads(reviewed)
        parsed["domain"] = normalized_domain
        return _validate_resume_payload(parsed)
    except Exception:
        planner_output["domain"] = normalized_domain
        return planner_output


def robust_parse_resume(resume_text: str, domain: str = "industry"):
    normalized_domain = normalize_domain(domain)
    try:
        planner_output = parse_resume(
            resume_text=resume_text,
            domain=normalized_domain,
            use_reviewer=False,
        )
        reviewed_output = resume_reviewer_pass(
            resume_text=resume_text,
            planner_output=planner_output,
            domain=normalized_domain,
        )
        return reviewed_output
    except Exception as e:
        raise RuntimeError(f"Resume parsing pipeline failed: {e}")
