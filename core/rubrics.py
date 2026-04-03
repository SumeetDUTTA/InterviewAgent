from core.domain_profiles import normalize_domain


RUBRIC_VERSION = "v1.0.0-domain-rubrics"


DEFAULT_DIMENSIONS = {
    "depth": "Quality and correctness of core content.",
    "clarity": "How clearly the candidate communicates the answer.",
    "confidence": "How decisively and coherently the candidate presents reasoning.",
}


DOMAIN_RUBRICS = {
    "industry": {
        "default": {
            "rubric_id": "industry.default",
            "dimensions": {
                "depth": "Role-relevant technical or functional depth with practical tradeoffs.",
                "clarity": "Structured, concise, business-relevant explanation.",
                "confidence": "Ownership mindset and execution confidence.",
            },
            "followup_threshold": 2,
        },
        "technical": {
            "rubric_id": "industry.technical",
            "dimensions": {
                "depth": "Correctness, implementation detail, and complexity awareness.",
                "clarity": "Clear articulation of approach and rationale.",
                "confidence": "Ability to defend tradeoffs and decisions.",
            },
            "followup_threshold": 3,
        },
    },
    "government": {
        "default": {
            "rubric_id": "government.default",
            "dimensions": {
                "depth": "Policy/regulation understanding and practical public-sector application.",
                "clarity": "Accessible communication for cross-stakeholder settings.",
                "confidence": "Sound judgement under accountability constraints.",
            },
            "followup_threshold": 2,
        },
        "case scenario": {
            "rubric_id": "government.case",
            "dimensions": {
                "depth": "Ethical reasoning, risk handling, and procedural correctness.",
                "clarity": "Stepwise and transparent justification.",
                "confidence": "Balanced confidence without overstatement.",
            },
            "followup_threshold": 3,
        },
    },
    "phd": {
        "default": {
            "rubric_id": "phd.default",
            "dimensions": {
                "depth": "Research rigor, methodological grounding, and evidence awareness.",
                "clarity": "Scientific communication quality and logical flow.",
                "confidence": "Research ownership with honest uncertainty.",
            },
            "followup_threshold": 2,
        },
        "methodology": {
            "rubric_id": "phd.methodology",
            "dimensions": {
                "depth": "Method selection, assumptions, validity, and limitations.",
                "clarity": "Precise explanation of procedures and evaluation.",
                "confidence": "Reasoned defence of methods and alternatives.",
            },
            "followup_threshold": 3,
        },
    },
}


def _normalize_round_type(round_type: str | None) -> str:
    if not round_type:
        return "default"
    return str(round_type).strip().lower()


def get_rubric(domain: str | None, round_type: str | None) -> dict:
    resolved_domain = normalize_domain(domain)
    resolved_round_type = _normalize_round_type(round_type)

    domain_map = DOMAIN_RUBRICS.get(resolved_domain, {})
    rubric = domain_map.get(resolved_round_type) or domain_map.get("default")

    if not rubric:
        rubric = {
            "rubric_id": f"{resolved_domain}.fallback",
            "dimensions": DEFAULT_DIMENSIONS,
            "followup_threshold": 2,
        }

    return {
        "domain": resolved_domain,
        "round_type": resolved_round_type,
        "rubric_version": RUBRIC_VERSION,
        **rubric,
    }


def rubric_prompt_block(domain: str | None, round_type: str | None) -> str:
    rubric = get_rubric(domain, round_type)
    dimensions = rubric.get("dimensions", DEFAULT_DIMENSIONS)
    return (
        f"Rubric Version: {rubric['rubric_version']}\n"
        f"Rubric ID: {rubric['rubric_id']}\n"
        "Scoring Dimensions:\n"
        f"- depth: {dimensions.get('depth', DEFAULT_DIMENSIONS['depth'])}\n"
        f"- clarity: {dimensions.get('clarity', DEFAULT_DIMENSIONS['clarity'])}\n"
        f"- confidence: {dimensions.get('confidence', DEFAULT_DIMENSIONS['confidence'])}\n"
        f"Follow-up threshold: trigger follow-up when any score <= {rubric.get('followup_threshold', 2)}."
    )


def _clamp_score(value) -> int:
    try:
        value = int(value)
    except Exception:
        value = 3
    return max(1, min(5, value))


def apply_rubric_post_rules(
    parsed: dict, answer: str, domain: str, round_type: str
) -> dict:
    rubric = get_rubric(domain, round_type)
    threshold = rubric.get("followup_threshold", 2)

    parsed["depth"] = _clamp_score(parsed.get("depth", 3))
    parsed["clarity"] = _clamp_score(parsed.get("clarity", 3))
    parsed["confidence"] = _clamp_score(parsed.get("confidence", 3))

    if len(answer.split()) < 5:
        parsed["depth"] = 1
        parsed["clarity"] = 1
        parsed["confidence"] = 1

    low_dimensions = []
    for dimension in ["depth", "clarity", "confidence"]:
        if parsed[dimension] <= threshold:
            low_dimensions.append(dimension)

    parsed["needs_followup"] = bool(low_dimensions)

    if low_dimensions:
        weakest = min(low_dimensions, key=lambda d: parsed[d])
        if weakest == "depth":
            parsed["followup_type"] = "depth_probe"
        elif weakest == "clarity":
            parsed["followup_type"] = "clarification"
        else:
            parsed["followup_type"] = "tradeoff"
    else:
        parsed["followup_type"] = parsed.get("followup_type", "depth_probe")

    parsed["rubric_id"] = rubric["rubric_id"]
    parsed["rubric_version"] = rubric["rubric_version"]
    return parsed
