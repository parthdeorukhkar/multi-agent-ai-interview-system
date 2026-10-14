"""
Reads a candidate's resume file (PDF, DOCX, or TXT) and extracts plain text
to use as background context for the Interviewer agent.
"""
from pathlib import Path


def read_resume(path: str) -> str:
    file_path = Path(path.strip().strip('"'))
    if not file_path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(file_path)
    elif ext == ".docx":
        return _read_docx(file_path)
    elif ext == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported resume format: {ext} (use .pdf, .docx, or .txt)")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()


def _read_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs).strip()