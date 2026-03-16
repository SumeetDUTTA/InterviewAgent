import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.file_loader import load_resume
from agents.resume_agent import parse_resume
from agents.jd_agent import parse_jd
from core.gap_analysis import analyze_gap
from agents.planner_agent import generate_interview_plan
from core.interview_runtime import run_interview
from agents.feedback_agent import generate_feedback
from core.domain_profiles import infer_domain_from_text


def main():

    print("=== AI Mock Interview Platform ===")

    resume_path = input("Enter resume file path (.pdf or .txt): ")
    jd_path = input("Enter job description file path (.txt): ")

    print("\nLoading files...")
    resume_text = load_resume(resume_path)
    jd_text = load_resume(jd_path)

    print("\nParsing resume...")
    inferred_domain = infer_domain_from_text(jd_text)
    resume_data = parse_resume(resume_text, domain=inferred_domain)

    print("Parsing job description...")
    jd_data = parse_jd(jd_text, domain=inferred_domain)
    resolved_domain = jd_data.get("domain", inferred_domain)

    gap_mode = input("Use LLM refinement for gap analysis? (Y/n): ").strip().lower()
    use_llm_gap_refinement = gap_mode not in {"n", "no", "0", "false"}

    print("Analyzing skill gaps...")
    gap_data = analyze_gap(
        resume_data,
        jd_data,
        domain=resolved_domain,
        use_llm_refinement=use_llm_gap_refinement,
    )

    print("\nGenerating interview plan...")
    plan = generate_interview_plan(
        gap_data["strong_skills"],
        gap_data["skill_gaps"],
        resume_data.get("experience_years", 0),
        domain=resolved_domain,
        competencies=jd_data.get("required_competencies", []),
        partial_skills=gap_data.get("partial_match_skills", []),
        readiness_score=gap_data.get("readiness_score", 0),
    )

    print("\nInterview Plan Generated:")
    print(plan)

    print("\nStarting Interview...\n")

    scores = run_interview(plan)

    print("\nGenerating Final Feedback...\n")
    feedback = generate_feedback(scores, domain=resolved_domain)

    print("\n===== FINAL REPORT =====")
    print(feedback)


if __name__ == "__main__":
    main()
