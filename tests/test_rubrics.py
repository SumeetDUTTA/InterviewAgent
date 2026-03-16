import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.rubrics import get_rubric, apply_rubric_post_rules, RUBRIC_VERSION


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "benchmark_cases.json"


def load_fixtures():
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_fixture_domains_have_rubrics():
    for case in load_fixtures():
        rubric = get_rubric(case["domain"], case["round_type"])
        assert rubric["rubric_version"] == RUBRIC_VERSION
        assert rubric["rubric_id"]
        assert isinstance(rubric.get("dimensions"), dict)


def test_apply_rubric_post_rules_adds_metadata_and_clamps_scores():
    parsed = {"depth": 8, "clarity": 0, "confidence": "bad"}
    normalized = apply_rubric_post_rules(
        parsed,
        answer="Short",
        domain="phd",
        round_type="Methodology",
    )

    assert 1 <= normalized["depth"] <= 5
    assert 1 <= normalized["clarity"] <= 5
    assert 1 <= normalized["confidence"] <= 5
    assert normalized["rubric_id"].startswith("phd")
    assert normalized["rubric_version"] == RUBRIC_VERSION
    assert normalized["needs_followup"] is True
    assert normalized["followup_type"] in {"depth_probe", "clarification", "tradeoff"}
