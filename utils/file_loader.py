import PyPDF2


def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
            print("Extracted Page Text:\n", text[:500], "...\n")  # Debug print
    return text


def load_resume(path):
    if path.endswith(".pdf"):
        return load_pdf_file(path)
    elif path.endswith(".txt"):
        return load_text_file(path)
    else:
        raise ValueError("Unsupported file format")
