import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.safety import guardrail_question, guardrail_answer


def test_guardrail_question_blocks_sensitive_content():
    blocked = guardrail_question(
        question="What is your age and marital status?",
        domain="industry",
        round_type="Role Fit",
        topic="team collaboration",
    )
    assert blocked["safe"] is False
    assert blocked["reason"] == "contains_protected_or_sensitive_topic"
    assert blocked["safe_question"] != "What is your age and marital status?"


def test_guardrail_question_allows_professional_content():
    allowed = guardrail_question(
        question="How do you prioritize tasks under deadline pressure?",
        domain="government",
        round_type="Case Scenario",
        topic="operations",
    )
    assert allowed["safe"] is True
    assert allowed["safe_question"] == "How do you prioritize tasks under deadline pressure?"


def test_guardrail_answer_flags_risky_patterns():
    flagged = guardrail_answer("I would hack the system and exploit a loophole.")
    assert flagged["flagged"] is True
    assert flagged["reason"] == "potential_risky_content"
