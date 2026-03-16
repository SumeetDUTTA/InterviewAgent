# InterviewAgent

AI-powered mock interview system that:

- Parses resume and job description text into structured data
- Detects skill/experience gaps and computes readiness score
- Builds a domain-aware interview plan
- Runs a terminal-based interview loop with adaptive difficulty
- Produces structured final feedback with deterministic + LLM-assisted signals

Supported domains:

- industry
- government
- phd

## Features

- Dual interfaces:
  - FastAPI app for plan generation and interview execution
  - Terminal interactive workflow for local practice
- Domain adaptation via profile templates in core logic
- Safety guardrails for questions and answer risk flags
- JSON-focused LLM pipeline with retry + repair behavior
- Test suite covering parsing, gap analysis, safety, and fixture quality

## Project Layout

```text
.
|- main.py                        # FastAPI entrypoint
|- terminal_interview.py          # Terminal interview entrypoint
|- agents/
|  |- resume_agent.py             # Resume extraction
|  |- jd_agent.py                 # JD extraction
|  |- planner_agent.py            # Interview plan generation
|  |- interview_conductor.py      # Question, scoring, follow-up generation
|  |- feedback_agent.py           # Final feedback synthesis
|- core/
|  |- gap_analysis.py             # Deterministic + optional LLM gap refinement
|  |- interview_runtime.py        # Interactive interview runtime
|  |- domain_profiles.py          # Domain normalization/inference/profiles
|  |- rubrics.py                  # Scoring rubric logic
|  |- safety.py                   # Guardrails
|  |- state.py                    # Serializable interview state wrapper
|- utils/
|  |- llm.py                      # Groq client + response parsing
|  |- file_loader.py              # TXT/PDF loading helpers
|- tests/
|  |- test_*.py
|  |- fixtures/
```

## Requirements

- Python 3.10+
- A Groq API key

Core Python packages used by the project:

- fastapi
- uvicorn
- pydantic
- python-dotenv
- groq
- PyPDF2
- pytest

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

1. Install dependencies.

```powershell
pip install fastapi uvicorn pydantic python-dotenv groq PyPDF2 pytest
```

1. Create a .env file in the project root.

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Notes:

- GROQ_API_KEY is required.
- GROQ_MODEL is optional; defaults to llama-3.1-8b-instant.

## Run: FastAPI Server

```powershell
uvicorn main:app --reload
```

Default local URL:

- <http://127.0.0.1:8000>

Interactive API docs:

- <http://127.0.0.1:8000/docs>

### Endpoint: POST /generate-plan

Builds structured resume/JD data, gap analysis, interview plan, and returns serialized state.

Request body:

```json
{
 "resume_text": "<resume plain text>",
 "jd_text": "<job description plain text>",
 "domain": "industry",
 "use_llm_gap_refinement": true
}
```

### Endpoint: POST /start-interview

Runs the full interview flow and returns scores + feedback payload.

Request body is the same as /generate-plan.

## Run: Terminal Interview

```powershell
python terminal_interview.py
```

You will be prompted for:

- Resume file path (.pdf or .txt)
- JD file path (.txt)
- Whether to enable LLM gap refinement

Then the app runs multi-round Q and A in terminal and prints final report.

## Testing

Run all tests:

```powershell
pytest -q
```

Run a single test file:

```powershell
pytest tests/test_gap_analysis.py -q
```

## Implementation Notes

- Domain is inferred from JD text when not provided explicitly.
- Gap analysis blends deterministic matching with optional LLM refinement.
- Interview difficulty is adapted from readiness score and recent answer quality.
- Feedback combines LLM output with deterministic safeguards to avoid unsupported claims.
- Safety guardrails block sensitive/protected-topic interview questions and flag risky answer content.

## Current Limitations

- No pinned dependency file is committed yet (requirements.txt/pyproject.toml).
- Some modules include debug prints that may be noisy in production logs.
- The long file terminal_interview_1.py appears to be legacy notebook-derived code and is not required for standard usage.

## Suggested Next Improvements

- Add a pinned requirements.txt and optional Makefile or task runner.
- Add API integration tests for /generate-plan and /start-interview.
- Add response examples in docs for each domain profile.
- Add configurable logging level and remove debug prints from normal runs.
