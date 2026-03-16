import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import agents.resume_agent as resume_agent


def test_parse_resume_recovers_valid_raw_json(monkeypatch):
    raw_payload = """
    {
      "skills": ["Python", "Node.js"],
      "tools": ["Git", "MongoDB"],
      "experience_years": 0,
      "projects": [{"name": "ExpenseKeeper", "description": "Full stack app"}],
      "education": [
        {
          "institution": "School Of Open Learning, University Of Delhi",
          "degree": "B.A. Programme, Computer Application",
          "duration": "2023 - 2027",
          "cgpa": "7.32/10"
        }
      ],
      "certifications": [
        {
          "name": "Data Analytics",
          "duration": "Aug 2024 - Feb 2025",
          "provider": "Trainity"
        }
      ],
      "publications": [],
      "research_areas": [],
      "policy_experience": [],
      "summary": "Motivated learner"
    }
    """

    def _fake_generate_with_mode(*args, **kwargs):
        return {"error": "Invalid JSON", "mode": "planner", "raw": raw_payload}

    monkeypatch.setattr(resume_agent, "generate_with_mode", _fake_generate_with_mode)

    result = resume_agent.parse_resume("Sample resume text", domain="industry")

    assert "error" not in result
    assert result["domain"] == "industry"
    assert "Python" in result["skills"]
    assert isinstance(result["education"], list)
    assert isinstance(result["certifications"], list)
