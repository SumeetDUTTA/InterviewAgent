from typing import Dict


DOMAIN_INDUSTRY = "industry"
DOMAIN_GOVERNMENT = "government"
DOMAIN_PHD = "phd"


DOMAIN_ALIASES = {
    "industry": DOMAIN_INDUSTRY,
    "private": DOMAIN_INDUSTRY,
    "private_sector": DOMAIN_INDUSTRY,
    "corporate": DOMAIN_INDUSTRY,
    "government": DOMAIN_GOVERNMENT,
    "public_sector": DOMAIN_GOVERNMENT,
    "gov": DOMAIN_GOVERNMENT,
    "phd": DOMAIN_PHD,
    "phd_admissions": DOMAIN_PHD,
    "academia": DOMAIN_PHD,
    "research": DOMAIN_PHD,
}


DOMAIN_PROFILES: Dict[str, dict] = {
    DOMAIN_INDUSTRY: {
        "label": "Industry / Private Sector",
        "default_focus_areas": ["problem solving", "role fit"],
        "score_labels": {
            "depth": "Technical/Role Depth",
            "clarity": "Communication",
            "confidence": "Confidence",
        },
        "round_templates": [
            {"type": "Role Fit", "focus_areas": ["experience", "impact"]},
            {"type": "Technical", "focus_areas": ["core skills", "tools"]},
            {"type": "Scenario", "focus_areas": ["tradeoffs", "execution"]},
        ],
    },
    DOMAIN_GOVERNMENT: {
        "label": "Government / Public Sector",
        "default_focus_areas": ["policy awareness", "public service judgement"],
        "score_labels": {
            "depth": "Policy/Domain Understanding",
            "clarity": "Communication",
            "confidence": "Judgement Confidence",
        },
        "round_templates": [
            {"type": "Policy Knowledge", "focus_areas": ["regulations", "governance"]},
            {"type": "Case Scenario", "focus_areas": ["ethics", "public impact"]},
            {"type": "Role Fit", "focus_areas": ["service mindset", "stakeholder handling"]},
        ],
    },
    DOMAIN_PHD: {
        "label": "PhD Admissions",
        "default_focus_areas": ["research depth", "methodology"],
        "score_labels": {
            "depth": "Research Depth",
            "clarity": "Scientific Communication",
            "confidence": "Research Confidence",
        },
        "round_templates": [
            {"type": "Research Background", "focus_areas": ["publications", "prior work"]},
            {"type": "Methodology", "focus_areas": ["methods", "validation"]},
            {"type": "Research Proposal", "focus_areas": ["novelty", "feasibility"]},
        ],
    },
}


def normalize_domain(domain: str | None) -> str:
    if not domain:
        return DOMAIN_INDUSTRY
    key = str(domain).strip().lower().replace(" ", "_")
    return DOMAIN_ALIASES.get(key, DOMAIN_INDUSTRY)


def get_domain_profile(domain: str | None) -> dict:
    normalized = normalize_domain(domain)
    return DOMAIN_PROFILES[normalized]


def infer_domain_from_text(text: str) -> str:
    if not text:
        return DOMAIN_INDUSTRY

    t = text.lower()
    scores = {
        DOMAIN_INDUSTRY: 0,
        DOMAIN_GOVERNMENT: 0,
        DOMAIN_PHD: 0,
    }

    for keyword in ["government", "public sector", "civil service", "regulation", "policy", "administration"]:
        if keyword in t:
            scores[DOMAIN_GOVERNMENT] += 1

    for keyword in ["phd", "doctoral", "research proposal", "supervisor", "publication", "thesis", "methodology"]:
        if keyword in t:
            scores[DOMAIN_PHD] += 1

    for keyword in ["product", "engineering", "software", "startup", "stakeholder", "delivery", "business"]:
        if keyword in t:
            scores[DOMAIN_INDUSTRY] += 1

    return max(scores, key=scores.get)
