import re
from core.domain_profiles import normalize_domain


SAFETY_VERSION = "v1.0.0-interview-guardrails"


QUESTION_BLOCK_PATTERNS = [
    r"\bage\b",
    r"\bgender\b",
    r"\breligion\b",
    r"\bcaste\b",
    r"\bmarital\s+status\b",
    r"\bpregnan(t|cy)\b",
    r"\bpolitical\s+affiliation\b",
    r"\bvote\b",
    r"\bsexual\s+orientation\b",
    r"\bdisability\b",
    r"\bmedical\s+condition\b",
    r"\bcriminal\s+record\b",
]


ANSWER_RISK_PATTERNS = [
    r"\bhack\b",
    r"\bexploit\b",
    r"\bbribe\b",
    r"\bfalsify\b",
    r"\bfraud\b",
    r"\billegal\b",
    r"\bviolence\b",
]


def _matches_any_pattern(text: str, patterns: list[str]) -> list[str]:
    if not text:
        return []
    matches = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)
    return matches


def safe_fallback_question(domain: str, round_type: str, topic: str) -> str:
    resolved_domain = normalize_domain(domain)
    if resolved_domain == "government":
        return f"How would you approach a {round_type} situation on {topic} while ensuring fairness, transparency, and public accountability?"
    if resolved_domain == "phd":
        return f"Can you explain your approach to {topic} in terms of method choice, assumptions, and limitations?"
    return f"Can you walk through your approach to {topic}, including tradeoffs and implementation considerations?"


def guardrail_question(question: str, domain: str, round_type: str, topic: str) -> dict:
    patterns = _matches_any_pattern(question, QUESTION_BLOCK_PATTERNS)
    if patterns:
        return {
            "safe": False,
            "reason": "contains_protected_or_sensitive_topic",
            "matched_patterns": patterns,
            "safe_question": safe_fallback_question(domain, round_type, topic),
            "safety_version": SAFETY_VERSION,
        }
    return {
        "safe": True,
        "reason": "ok",
        "matched_patterns": [],
        "safe_question": question,
        "safety_version": SAFETY_VERSION,
    }


def guardrail_answer(answer: str) -> dict:
    patterns = _matches_any_pattern(answer, ANSWER_RISK_PATTERNS)
    return {
        "flagged": bool(patterns),
        "reason": "potential_risky_content" if patterns else "ok",
        "matched_patterns": patterns,
        "safety_version": SAFETY_VERSION,
    }
