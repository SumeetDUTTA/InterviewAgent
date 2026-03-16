import sys
import os
import re
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.llm import generate_with_mode
import json
from core.domain_profiles import normalize_domain
from core.rubrics import rubric_prompt_block, apply_rubric_post_rules


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _sanitize_answer_text(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    transcript_markers = [
        r"\buser\s*:",
        r"\bgithub\s*copilot\s*:",
        r"\bai\s*question\s*:",
        r"\bai\s*follow-up\s*:",
        r"\bcandidate\s*:",
    ]

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in transcript_markers):
        copilot_splits = re.split(r"(?i)\bgithub\s*copilot\s*:", text)
        if len(copilot_splits) > 1:
            text = copilot_splits[-1].strip()
        text = re.sub(r"(?im)^\s*(user|candidate|ai question|ai follow-up)\s*:\s*", "", text)

    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_meta_or_assisted_answer(answer: str) -> bool:
    text = (answer or "").lower()
    markers = [
        "github copilot:",
        "user:",
        "ai question:",
        "ai follow-up:",
        "answer this question for me",
        "here's a strong sample response",
        "ai overview",
    ]
    hits = sum(1 for marker in markers if marker in text)
    return hits >= 2


def _contains_security_red_flags(answer: str) -> bool:
    text = (answer or "").lower()
    patterns = [
        r"\b(bypass|disable|turn off|remove)\b.{0,30}\b(auth|authentication|authorization|2fa|mfa|security)\b",
        r"\bstore\b.{0,30}\b(password|credentials?|api key|token|secret)\b.{0,30}\b(plain\s*text|openly|unencrypted)\b",
        r"\bhardcod(e|ing)\b.{0,25}\b(password|credentials?|api key|token|secret)\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _has_concrete_evidence(answer: str) -> bool:
    text = answer or ""
    patterns = [
        r"\bfor example\b",
        r"\bfor instance\b",
        r"\bin my (last|previous|recent)\b",
        r"\bin a project\b",
        r"\bwe (implemented|built|reduced|improved|launched|delivered)\b",
        r"\bresult(ed)? in\b",
        r"\b\d+\s*(%|ms|sec|seconds|minutes|hours|days|x)\b",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _vagueness_signals(answer: str) -> dict:
    normalized = _normalize_text(answer)
    word_count = len(normalized.split())

    vague_phrases = [
        "at a high level",
        "in general",
        "it depends",
        "best practices",
        "i would focus on",
        "i would aim to",
        "generally speaking",
        "overall approach",
    ]
    vague_hits = sum(1 for phrase in vague_phrases if phrase in normalized)
    has_concrete = _has_concrete_evidence(answer)

    is_vague = (
        (word_count < 45 and vague_hits >= 1 and not has_concrete)
        or (word_count >= 45 and vague_hits >= 3 and not has_concrete)
    )

    return {
        "word_count": word_count,
        "vague_hits": vague_hits,
        "has_concrete": has_concrete,
        "is_vague": is_vague,
    }


def _question_needs_concrete_example(question: str) -> bool:
    q = _normalize_text(question)
    triggers = [
        "how would you",
        "walk me through",
        "describe a real project",
        "what trade offs",
        "what trade-offs",
        "challenge you faced",
    ]
    return any(trigger in q for trigger in triggers)


def _followup_depth_threshold(difficulty: str, round_type: str) -> int:
    base = 3
    if str(difficulty).upper() == "BEGINNER":
        base = 2
    elif str(difficulty).upper() == "ADVANCED":
        base = 4

    if round_type == "Warm-up":
        base = max(2, base - 1)
    elif round_type in ["Case Study", "Technical", "Domain-Specific"]:
        base = min(5, base + 1)
    return max(1, min(5, base))


def _should_force_depth_followup(vagueness: dict, question: str, round_type: str, difficulty: str, evaluation: dict) -> bool:
    if round_type == "Warm-up":
        return False

    depth_threshold = _followup_depth_threshold(difficulty, round_type)
    depth_score = int(evaluation.get("depth", 3))

    lacks_concrete = _question_needs_concrete_example(question) and not vagueness["has_concrete"]
    strong_vagueness = vagueness["is_vague"] and vagueness["vague_hits"] >= 2
    return strong_vagueness or (lacks_concrete and depth_score < min(5, depth_threshold + 1)) or (vagueness["is_vague"] and depth_score < depth_threshold)


def _jaccard_similarity(a: str, b: str) -> float:
    a_set = set(_normalize_text(a).split())
    b_set = set(_normalize_text(b).split())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def _extract_single_question(raw_text: str, fallback: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return fallback

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("question"), str) and parsed["question"].strip():
                text = parsed["question"].strip()
            else:
                for value in parsed.values():
                    if isinstance(value, str) and "?" in value:
                        text = value.strip()
                        break
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str) and "?" in item:
                    text = item.strip()
                    break
    except Exception:
        pass

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    selected = ""
    for ln in lines:
        candidate = ln.lstrip("-*• ")
        if ". " in candidate and candidate.split(". ", 1)[0].isdigit():
            candidate = candidate.split(". ", 1)[1]
        if candidate.endswith("?"):
            selected = candidate
            break
        if "?" in candidate and not selected:
            selected = candidate.split("?")[0].strip() + "?"

    selected = (selected or fallback).strip().strip('"').strip("'")
    if not selected.endswith("?"):
        selected = selected.rstrip(".") + "?"
    return selected


def _is_generic_question(question: str, topic: str) -> bool:
    q = _normalize_text(question)
    topic_norm = _normalize_text(topic)
    generic_patterns = [
        f"can you explain your understanding of {topic_norm}",
        "can you explain your understanding",
        "with one practical example",
    ]
    return any(pattern in q for pattern in generic_patterns)


def _is_repetitive_question(question: str, previous_questions: List[str], threshold: float = 0.72) -> bool:
    for previous in previous_questions:
        if _jaccard_similarity(question, previous) >= threshold:
            return True
    return False


def _clamp5(value: int) -> int:
    return max(1, min(5, int(round(value))))


def _heuristic_scores(answer: str) -> dict:
    text = (answer or "").strip()
    word_count = len(text.split())
    has_example = _has_concrete_evidence(text)

    depth = 2
    clarity = 3
    confidence = 3
    if word_count >= 25:
        depth += 1
    if word_count >= 50:
        depth += 1
    if has_example:
        depth += 1
        clarity += 1
    if _is_meta_or_assisted_answer(text):
        confidence -= 1
        clarity -= 1

    return {
        "depth": _clamp5(depth),
        "clarity": _clamp5(clarity),
        "confidence": _clamp5(confidence),
    }


def _merge_scores(llm_eval: dict, heuristic_eval: dict) -> dict:
    merged = {}
    for key in ["depth", "clarity", "confidence"]:
        llm_value = int(llm_eval.get(key, 3))
        heur_value = int(heuristic_eval.get(key, 3))
        merged[key] = _clamp5((0.75 * llm_value) + (0.25 * heur_value))
    return merged


def generate_question(
    topic,
    difficulty,
    previous_questions,
    domain="industry",
    round_type="General",
):
    resolved_domain = normalize_domain(domain)
    topic_lower = (topic or "").lower()
    is_warmup = round_type == "Warm-up" or "candidate introduction" in topic_lower

    warmup_questions = [
        "Could you please introduce yourself and briefly walk me through your background?",
        "Tell me about your recent experience and the kind of work you have been focusing on.",
        "What role are you currently targeting, and why does it fit your strengths?",
    ]

    if is_warmup:
        fallback = warmup_questions[min(len(previous_questions), len(warmup_questions) - 1)]
        system_prompt = f"""
        You are a professional interviewer.
        Domain: {resolved_domain}
        Round type: {round_type}
        Ask one concise warm-up question.
        Return exactly one plain-text question.
        Avoid repeats from: {previous_questions}
        """
    else:
        fallback_options = [
            f"Can you describe a real project or task where you applied {topic}?",
            f"What trade-offs did you consider while applying {topic} in a real work situation?",
            f"Can you walk me through one challenge you faced with {topic} and how you solved it?",
        ]
        fallback = fallback_options[len(previous_questions) % len(fallback_options)]
        system_prompt = f"""
        You are a professional interviewer.
        Domain: {resolved_domain}
        Round type: {round_type}
        Difficulty: {difficulty}
        Topic: {topic}

        Return exactly one plain-text question.
        Do not return multiple questions, lists, or headings.
        Avoid generic 'explain your understanding' phrasing.
        Keep it practical and role-relevant.
        Avoid repeats from: {previous_questions}
        """

    question = fallback
    for _ in range(3):
        response = generate_with_mode("question", system_prompt, "")
        candidate = _extract_single_question(response, fallback)
        if _is_generic_question(candidate, topic) and not is_warmup:
            continue
        if _is_repetitive_question(candidate, previous_questions):
            continue
        question = candidate
        break

    return question


def analyze_answer(
    question,
    answer,
    domain="industry",
    round_type="General",
    topic="general competency",
    difficulty="INTERMEDIATE",
    previous_answers=None,
):
    previous_answers = previous_answers or []
    sanitized_answer = _sanitize_answer_text(answer)
    scoring_answer = sanitized_answer or (answer or "")

    resolved_domain = normalize_domain(domain)
    rubric_block = rubric_prompt_block(resolved_domain, round_type)
    system_prompt = f"""
Evaluate the candidate answer.

Return JSON:
- depth (1-5)
- clarity (1-5)
- confidence (1-5)
- needs_followup (true/false)
- followup_type (depth_probe/clarification/tradeoff)

Use this rubric strictly:
{rubric_block}
"""

    user_prompt = f"""
Question: {question}
Topic: {topic}
Difficulty: {difficulty}
Answer: {scoring_answer}
Domain: {resolved_domain}
Round type: {round_type}
"""
    
    print("Analyzing answer with:")
    print("Question:", question)
    print("Answer:", scoring_answer)

    response = generate_with_mode(
        "scoring", system_prompt, user_prompt, expect_json=True, retries=3
    )

    print("Raw Model Output:\n", response)  # Debug print

    try:
        parsed = response if isinstance(response, dict) else json.loads(response)
        parsed = apply_rubric_post_rules(parsed, scoring_answer, domain=resolved_domain, round_type=round_type)

        heuristics = _heuristic_scores(scoring_answer)
        blended = _merge_scores(parsed, heuristics)
        parsed.update(blended)

        normalized_previous = [_sanitize_answer_text(a) for a in previous_answers[-3:]]
        for previous_answer in normalized_previous:
            if previous_answer and _jaccard_similarity(scoring_answer, previous_answer) >= 0.82:
                parsed["depth"] = _clamp5(parsed.get("depth", 3) - 1)
                parsed["clarity"] = _clamp5(parsed.get("clarity", 3) - 1)
                parsed["confidence"] = _clamp5(parsed.get("confidence", 3) - 1)
                parsed["needs_followup"] = True
                parsed["followup_type"] = "clarification"
                break

        if _is_meta_or_assisted_answer(answer):
            parsed["depth"] = min(parsed.get("depth", 3), 2)
            parsed["clarity"] = min(parsed.get("clarity", 3), 3)
            parsed["confidence"] = min(parsed.get("confidence", 3), 2)
            parsed["needs_followup"] = True
            parsed["followup_type"] = "clarification"

        vagueness = _vagueness_signals(scoring_answer)
        if vagueness["is_vague"]:
            parsed["depth"] = _clamp5(parsed.get("depth", 3) - 1)
            if difficulty == "ADVANCED":
                parsed["confidence"] = _clamp5(parsed.get("confidence", 3) - 1)

        if _should_force_depth_followup(vagueness, question, round_type, difficulty, parsed):
            parsed["needs_followup"] = True
            parsed["followup_type"] = "depth_probe"

        if _contains_security_red_flags(scoring_answer):
            parsed["depth"] = min(parsed.get("depth", 3), 1)
            parsed["confidence"] = min(parsed.get("confidence", 3), 1)
            parsed["clarity"] = min(parsed.get("clarity", 3), 2)
            parsed["needs_followup"] = True
            parsed["followup_type"] = "clarification"

        if any(parsed.get(metric, 5) <= 2 for metric in ["depth", "clarity", "confidence"]):
            parsed["needs_followup"] = True
            parsed["followup_type"] = "depth_probe" if parsed.get("depth", 3) <= 2 else "clarification"

        parsed["domain"] = resolved_domain
        parsed["round_type"] = round_type

        print("Parsed Analysis:\n", parsed)  # Debug print
        return parsed
    except Exception as e:
        print("Error parsing response:", e)
        fallback = {
            "depth": 3,
            "clarity": 3,
            "confidence": 3,
            "needs_followup": True,
            "followup_type": "clarification",
            "domain": resolved_domain,
            "round_type": round_type,
        }
        return apply_rubric_post_rules(
            fallback, scoring_answer, domain=resolved_domain, round_type=round_type
        )


def generate_followup(
    question,
    answer,
    followup_type,
    domain="industry",
    round_type="General",
):
    resolved_domain = normalize_domain(domain)
    system_prompt = f"""
The candidate answer was weak.
Domain: {resolved_domain}
Round type: {round_type}
Ask one {followup_type} follow-up question.
Return exactly one plain-text question.
Do not return headings, bullets, or multiple questions.
"""

    user_prompt = f"""
Original question: {question}
Candidate answer: {_sanitize_answer_text(answer)}
"""

    response = generate_with_mode("question", system_prompt, user_prompt)
    fallback_map = {
        "depth_probe": "Can you give one specific example with context, action, and measurable result?",
        "clarification": "Can you answer directly in your own words without quoting external text?",
        "tradeoff": "What trade-offs did you consider, and why did you choose that approach?",
    }
    fallback = fallback_map.get(followup_type, "Can you answer the previous question with one concrete example from your experience?")
    return _extract_single_question(response, fallback)
