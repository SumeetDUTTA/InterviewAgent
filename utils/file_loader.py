import os
import PyPDF2

DEBUG_LOGS = os.getenv('INTERVIEW_AGENT_DEBUG', "0") == "1"

def _debug_log(*args,):
    if DEBUG_LOGS:
        print(*args)

def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(path):
    chunks = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
            _debug_log("Extracted Page Text:\n", chunks[-1][:500], "...\n")  # Debug print
    return "\n".join(chunks)


def load_resume(path):
    if path.endswith(".pdf"):
        return load_pdf_file(path)
    elif path.endswith(".txt"):
        return load_text_file(path)
    else:
        raise ValueError("Unsupported file format")
