import json
from types import SimpleNamespace

from utils import llm


def test_extract_json_from_fenced_block():
    text = """```json
    {"a": 1, "b": [2, 3]}
    ```"""
    parsed = llm.extract_json(text)
    assert parsed == {"a": 1, "b": [2, 3]}


def test_generate_with_mode_repairs_invalid_json(monkeypatch):
    calls = {"count": 0}

    class FakeCompletions:
        def create(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                content = "{\"depth\": 4, \"clarity\": 5"
            else:
                content = json.dumps(
                    {
                        "depth": 4,
                        "clarity": 5,
                        "confidence": 4,
                        "needs_followup": False,
                        "followup_type": "depth_probe",
                    }
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(llm, "_get_groq_client", lambda: fake_client)

    result = llm.generate_with_mode(
        mode="scoring",
        system_prompt="test",
        user_prompt="test",
        expect_json=True,
        retries=1,
    )

    assert result["depth"] == 4
    assert result["clarity"] == 5


def test_generate_with_mode_validator_rejects_then_returns_error(monkeypatch):
    class FakeCompletions:
        def create(self, *args, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps({"depth": 4, "clarity": 4})
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(llm, "_get_groq_client", lambda: fake_client)

    def validator(payload):
        required = ["depth", "clarity", "confidence"]
        for key in required:
            if key not in payload:
                raise ValueError("missing")
        return payload

    result = llm.generate_with_mode(
        mode="scoring",
        system_prompt="test",
        user_prompt="test",
        expect_json=True,
        retries=1,
        validator=validator,
    )

    assert result["error"] == "Invalid JSON"
    assert result["mode"] == "scoring"
