import json
import re
import os
from typing import Callable, Any
from dotenv import load_dotenv

load_dotenv()

MODEL_PRESETS = {
    "planner": {"temperature": 0.2, "max_tokens": 700},
    "question": {"temperature": 0.6, "max_tokens": 200},
    "scoring": {"temperature": 0.1, "max_tokens": 200},
    "feedback": {"temperature": 0.3, "max_tokens": 700},
    "reviewer": {"temperature": 0.0, "max_tokens": 500},
}

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def extract_json(text):
    if not text:
        return None

    cleaned = str(text).strip()

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def _get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")

    from groq import Groq

    return Groq(api_key=api_key)


def _call_groq(
    model: str,
    system_prompt: str,
    user_prompt: str,
    config: dict,
    expect_json: bool = False,
) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=model,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"} if expect_json else None,
    )
    return (response.choices[0].message.content or "").strip()


def _repair_json(mode: str, broken_content: str, model: str, config: dict):
    repair_system = "You are a strict JSON repair assistant. Return valid JSON only."
    repair_user = f"""
The following model output should be valid JSON but is malformed.

Mode: {mode}
Broken output:
{broken_content}

Return corrected JSON only.
"""
    repaired_text = _call_groq(
        model=model,
        system_prompt=repair_system,
        user_prompt=repair_user,
        config={"temperature": 0.0, "max_tokens": config["max_tokens"]},
        expect_json=True,
    )
    return extract_json(repaired_text)


def _is_valid_with_validator(
    candidate: Any, validator: Callable[[Any], Any] | None
) -> tuple[bool, Any]:
    if validator is None:
        return True, candidate
    try:
        validated = validator(candidate)
        return True, validated
    except Exception:
        return False, candidate


def generate_with_mode(
    mode: str,
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = False,
    retries: int = 2,
    model: str | None = None,
    validator: Callable[[Any], Any] | None = None,
):

    config = MODEL_PRESETS.get(mode)

    if not config:
        raise ValueError(f"Unknown mode: {mode}")

    resolved_model = model or DEFAULT_MODEL
    last_content = ""

    for attempt in range(retries):

        content = _call_groq(
            model=resolved_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
            expect_json=expect_json,
        )
        last_content = content

        if not expect_json:
            return content

        parsed = extract_json(content)
        if parsed:
            valid, validated = _is_valid_with_validator(parsed, validator)
            if valid:
                return validated

        repaired = _repair_json(mode, content, resolved_model, config)
        if repaired is not None:
            valid, validated = _is_valid_with_validator(repaired, validator)
            if valid:
                return validated

    if expect_json:
        return {"error": "Invalid JSON", "mode": mode, "raw": last_content}
    return last_content
