import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.domain_profiles import normalize_domain
from core.safety import guardrail_question


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "benchmark_cases.json"


def load_fixtures():
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_benchmark_fixture_coverage():
    fixtures = load_fixtures()
    domains = {normalize_domain(item["domain"]) for item in fixtures}
    assert {"industry", "government", "phd"}.issubset(domains)


def test_fixture_questions_pass_safety_guardrails():
    for case in load_fixtures():
        safety = guardrail_question(
            question=case["question"],
            domain=case["domain"],
            round_type=case["round_type"],
            topic=case["topic"],
        )
        assert safety["safe"] is True
