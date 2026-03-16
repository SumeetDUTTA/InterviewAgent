import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import agents.jd_agent as jd_agent


def test_parse_jd_recovers_valid_raw_json_government(monkeypatch):
    jd_text = "Public sector policy analyst role focused on regulation and governance."
    raw_payload = """
    {
      "domain": "government",
      "required_skills": ["Policy Analysis", "Stakeholder Management"],
      "required_tools": ["MS Excel"],
      "required_competencies": ["Public communication", "Ethical judgement"],
      "experience_required": "3 years",
      "evaluation_focus": ["Policy implementation", "Citizen impact"]
    }
    """

    def _fake_generate_with_mode(*args, **kwargs):
        return {"error": "Invalid JSON", "mode": "planner", "raw": raw_payload}

    monkeypatch.setattr(jd_agent, "generate_with_mode", _fake_generate_with_mode)

    result = jd_agent.parse_jd(jd_text)

    assert "error" not in result
    assert result["domain"] == "government"
    assert "Policy Analysis" in result["required_skills"]


def test_parse_jd_infers_phd_domain_without_technical_bias(monkeypatch):
    jd_text = "PhD admission in computational social science with research proposal and methodology focus."

    def _fake_generate_with_mode(*args, **kwargs):
        return {
            "required_skills": ["Research design", "Statistical analysis"],
            "required_tools": ["Python"],
            "required_competencies": ["Methodological rigor"],
            "experience_required": "Not specified",
            "evaluation_focus": ["Novelty", "Feasibility"],
        }

    monkeypatch.setattr(jd_agent, "generate_with_mode", _fake_generate_with_mode)

    result = jd_agent.parse_jd(jd_text)

    assert "error" not in result
    assert result["domain"] == "phd"
    assert "Research design" in result["required_skills"]
