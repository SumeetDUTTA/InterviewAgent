import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.interview_conductor import (
    generate_question,
    analyze_answer,
    generate_followup,
)

from core.rubrics import RUBRIC_VERSION
from datetime import datetime
from core.safety import guardrail_question, guardrail_answer, SAFETY_VERSION


def run_interview(plan):

    all_scores = []
    questions_per_round = 3
    domain = plan.get("domain", "industry")

    for round_data in plan.get("rounds", []):

        focus_areas = round_data.get("focus_areas") or [round_data.get("type", "General")]
        topic = focus_areas[0] if focus_areas else "general competency"
        round_type = round_data.get("type", "General")
        base_difficulty = plan.get("difficulty", "BEGINNER")
        current_difficulty = base_difficulty

        print(f"\n=== Round: {round_data['type']} | Topic: {topic} | Difficulty: {current_difficulty} ===\n")

        previous_questions = []

        for i in range(questions_per_round):

            if len(all_scores) >= 2:
                recent = all_scores[-2:]
                if all(float(s.get("depth", 0)) >= 4 for s in recent):
                    current_difficulty = "ADVANCED"
                elif all(float(s.get("depth", 5)) <= 2 for s in recent):
                    current_difficulty = "BEGINNER"
                else:
                    current_difficulty = base_difficulty

            question = generate_question(
                topic,
                current_difficulty,
                previous_questions,
                domain=domain,
                round_type=round_type,
            )
            question_safety = guardrail_question(question, domain, round_type, topic)
            question = question_safety["safe_question"]
            previous_questions.append(question)
            print("\nAI Question:", question)

            answer = input("Your Answer: ")

            attempts = 0
            while (not answer or len(answer.strip()) < 10) and attempts < 2:
                print("Answer too short. Please elaborate.")
                attempts += 1
                answer = input("Your Answer: ")

            print("Candidate:", answer)

            if not answer or len(answer.strip()) < 10:
                print("Skipping question due to insufficient response.")
                continue

            answer_safety = guardrail_answer(answer)

            analysis = analyze_answer(
                question,
                answer,
                domain=domain,
                round_type=round_type,
                topic=topic,
                difficulty=current_difficulty,
                previous_answers=[s.get("answer", "") for s in all_scores[-5:] if s.get("answer")],
            )
            analysis["topic"] = topic
            analysis["difficulty"] = current_difficulty
            analysis["question"] = question
            analysis["answer"] = answer
            analysis["answer_text"] = answer
            analysis["answer_length"] = len(answer.split())
            analysis["is_followup"] = False
            analysis["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
            analysis["rubric_version"] = analysis.get("rubric_version", RUBRIC_VERSION)
            analysis["safety"] = {
                "question": question_safety,
                "answer": answer_safety,
                "safety_version": SAFETY_VERSION,
            }
            all_scores.append(analysis)

            if analysis.get("needs_followup"):
                followup = generate_followup(
                    question,
                    answer,
                    analysis["followup_type"],
                    domain=domain,
                    round_type=round_type,
                )
                followup_safety = guardrail_question(followup, domain, round_type, topic)
                followup = followup_safety["safe_question"]

                print("\nAI Follow-up:", followup)

                followup_answer = input("Your Answer: ")

                followup_answer_safety = guardrail_answer(followup_answer)

                analysis = analyze_answer(
                    followup,
                    followup_answer,
                    domain=domain,
                    round_type=round_type,
                    topic=topic,
                    difficulty=current_difficulty,
                    previous_answers=[s.get("answer", "") for s in all_scores[-5:] if s.get("answer")],
                )
                analysis["topic"] = topic
                analysis["difficulty"] = current_difficulty
                analysis["question"] = followup
                analysis["answer"] = followup_answer
                analysis["answer_text"] = followup_answer
                analysis["answer_length"] = len(followup_answer.split())
                analysis["is_followup"] = True
                analysis["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"
                analysis["rubric_version"] = analysis.get("rubric_version", RUBRIC_VERSION)
                analysis["safety"] = {
                    "question": followup_safety,
                    "answer": followup_answer_safety,
                    "safety_version": SAFETY_VERSION,
                }
                all_scores.append(analysis)

    return all_scores


def summarize_interview(scores):
    if not scores:
        return {
            "average_depth": 0,
            "average_clarity": 0,
            "average_confidence": 0,
            "average_speech_clarity": 0,
            "total_questions": 0,
            "followups_triggered": 0,
        }

    total = len(scores)
    avg_depth = sum(float(s.get("depth", 0)) for s in scores) / total
    avg_clarity = sum(float(s.get("clarity", 0)) for s in scores) / total
    avg_confidence = sum(float(s.get("confidence", 0)) for s in scores) / total
    avg_speech = sum(float(s.get("speech_clarity", 0)) for s in scores) / total
    followups_triggered = sum(1 for s in scores if s.get("is_followup"))

    return {
        "average_depth": round(avg_depth, 2),
        "average_clarity": round(avg_clarity, 2),
        "average_confidence": round(avg_confidence, 2),
        "average_speech_clarity": round(avg_speech, 2),
        "total_questions": total,
        "followups_triggered": followups_triggered,
    }
