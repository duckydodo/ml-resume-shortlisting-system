import pdfplumber
from docx import Document
from pathlib import Path

def extract_text(file) -> str:
    """
    Extract text from PDF or DOCX.
    Accepts file path OR file-like object.
    """
    if isinstance(file, (str, Path)):
        path = Path(file)
        suffix = path.suffix.lower()
    else:
        # Streamlit uploaded file
        path = None
        suffix = Path(file.name).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file)
    elif suffix == ".docx":
        return _extract_docx(file)
    else:
        raise ValueError("Unsupported file format")


def _extract_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def _extract_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
