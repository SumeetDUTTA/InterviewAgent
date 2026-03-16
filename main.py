import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from agents.resume_agent import parse_resume
from agents.jd_agent import parse_jd
from agents.planner_agent import generate_interview_plan
from agents.feedback_agent import generate_feedback
from core.gap_analysis import analyze_gap
from core.state import InterviewState
from core.interview_runtime import run_interview
from core.domain_profiles import infer_domain_from_text, normalize_domain
from core.rubrics import RUBRIC_VERSION
from core.safety import SAFETY_VERSION

app = FastAPI()


class InterviewRequest(BaseModel):
    resume_text: str
    jd_text: str
    domain: str | None = None
    use_llm_gap_refinement: bool = True


@app.post("/generate-plan")
def generate_plan(request: InterviewRequest):

    requested_domain = normalize_domain(request.domain) if request.domain else infer_domain_from_text(request.jd_text)

    resume_data = parse_resume(request.resume_text, domain=requested_domain)
    jd_data = parse_jd(request.jd_text, domain=requested_domain)
    resolved_domain = normalize_domain(jd_data.get("domain", requested_domain))

    gap_data = analyze_gap(
        resume_data,
        jd_data,
        domain=resolved_domain,
        use_llm_refinement=request.use_llm_gap_refinement,
    )

    plan = generate_interview_plan(
        gap_data["strong_skills"],
        gap_data["skill_gaps"],
        resume_data.get("experience_years", 0),
        domain=resolved_domain,
        competencies=jd_data.get("required_competencies", []),
        partial_skills=gap_data.get("partial_match_skills", []),
        readiness_score=gap_data.get("readiness_score", 0),
    )

    state = InterviewState(
        resume_data,
        jd_data,
        gap_data,
        plan,
        domain=resolved_domain,
        audit_trace={
            "requested_domain": requested_domain,
            "resolved_domain": resolved_domain,
            "use_llm_gap_refinement": request.use_llm_gap_refinement,
            "schema_version": "v1-domain-aware",
            "rubric_version": RUBRIC_VERSION,
            "safety_version": SAFETY_VERSION,
        },
    )

    return {
        "resume_data": resume_data,
        "jd_data": jd_data,
        "gap_analysis": gap_data,
        "interview_plan": plan,
        "state": state.to_dict(),
    }


@app.post("/start-interview")
def start_interview(request: InterviewRequest):

    requested_domain = normalize_domain(request.domain) if request.domain else infer_domain_from_text(request.jd_text)
    resume_data = parse_resume(request.resume_text, domain=requested_domain)
    jd_data = parse_jd(request.jd_text, domain=requested_domain)
    resolved_domain = normalize_domain(jd_data.get("domain", requested_domain))
    gap_data = analyze_gap(
        resume_data,
        jd_data,
        domain=resolved_domain,
        use_llm_refinement=request.use_llm_gap_refinement,
    )

    plan = generate_interview_plan(
        gap_data["strong_skills"],
        gap_data["skill_gaps"],
        resume_data.get("experience_years", 0),
        domain=resolved_domain,
        competencies=jd_data.get("required_competencies", []),
        partial_skills=gap_data.get("partial_match_skills", []),
        readiness_score=gap_data.get("readiness_score", 0),
    )

    scores = run_interview(plan)

    feedback = generate_feedback(scores, domain=resolved_domain)

    return {
        "domain": resolved_domain,
        "interview_plan": plan,
        "scores": scores,
        "feedback": feedback,
        "audit": {
            "schema_version": "v1-domain-aware",
            "rubric_version": RUBRIC_VERSION,
            "safety_version": SAFETY_VERSION,
            "requested_domain": requested_domain,
            "use_llm_gap_refinement": request.use_llm_gap_refinement,
        },
    }
