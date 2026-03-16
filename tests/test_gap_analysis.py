import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import core.gap_analysis as gap_analysis


def test_analyze_gap_deterministic_mode():
    resume_data = {
        "domain": "industry",
        "skills": ["Python", "SQL", "Data Analysis"],
        "tools": ["Git"],
        "experience_years": 3,
    }
    jd_data = {
        "domain": "industry",
        "required_skills": ["Python", "Machine Learning"],
        "required_tools": ["Git"],
        "required_competencies": ["problem solving"],
        "experience_required": "2+ years",
    }

    result = gap_analysis.analyze_gap(
        resume_data,
        jd_data,
        use_llm_refinement=False,
    )

    assert result["domain"] == "industry"
    assert "python" in result["strong_skills"]
    assert "machine learning" in result["skill_gaps"]
    assert "problem solving" not in result["skill_gaps"]
    assert "problem solving" in result["missing_competencies"]
    assert 0 <= result["readiness_score"] <= 100


def test_analyze_gap_hybrid_mode_uses_llm_and_blends_score(monkeypatch):
    resume_data = {
        "domain": "industry",
        "skills": ["Python", "Communication"],
        "tools": ["Git"],
        "experience_years": 4,
    }
    jd_data = {
        "domain": "industry",
        "required_skills": ["Python", "Leadership"],
        "required_tools": ["Git"],
        "required_competencies": ["leadership"],
        "experience_required": "5 years",
    }

    def _fake_generate_with_mode(**kwargs):
        return {
            "strong_skills": ["python", "git"],
            "partial_match_skills": ["lead"],
            "skill_gaps": ["leadership"],
            "resume_experience_years": 4,
            "experience_required": "5 years",
            "readiness_score": 90,
        }

    monkeypatch.setattr(gap_analysis, "generate_with_mode", _fake_generate_with_mode)

    result = gap_analysis.analyze_gap(
        resume_data,
        jd_data,
        use_llm_refinement=True,
    )

    assert "python" in result["strong_skills"]
    assert "git" in result["strong_skills"]
    assert result["readiness_score"] <= 100
    assert result["readiness_score"] >= 0
    assert "leadership" in result["skill_gaps"]


def test_analyze_gap_keeps_responsibility_competencies_out_of_skill_gaps():
    resume_data = {
        "domain": "industry",
        "skills": ["Node.js", "MongoDB"],
        "tools": [],
        "experience_years": 0,
    }
    jd_data = {
        "domain": "industry",
        "required_skills": ["Node.js", "Express", "MongoDB", "REST APIs"],
        "required_tools": [],
        "required_competencies": [
            "Scalable system design",
            "API performance optimization",
            "Clean and maintainable code",
            "Collaboration",
        ],
        "experience_required": "2+ years",
    }

    result = gap_analysis.analyze_gap(
        resume_data,
        jd_data,
        use_llm_refinement=False,
    )

    assert "node.js" in result["strong_skills"]
    assert "mongodb" in result["strong_skills"]
    assert "rest apis" in result["skill_gaps"]
    assert "scalable system design" not in result["skill_gaps"]
    assert "api performance optimization" in result["missing_competencies"]
