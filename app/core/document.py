"""Document loading and chunking.

Chunking strategy and rationale
-------------------------------
The provided document is not free-flowing prose: it is a set of short,
self-contained paragraphs, each describing one topic/entity (Zara, Alex, the
"Luz de Luna" flower, Emma, "Sombra Silenciosa"). Every test question is
answered by exactly ONE of those paragraphs.

Therefore we split on blank lines: one paragraph == one chunk (~5 chunks).
This keeps each answer fully inside a single chunk, so top-1 retrieval returns
complete context. A fixed-size splitter (e.g. 200 chars) would cut a paragraph
in half and risk retrieving only part of the answer.

Per the challenge hint, we may adjust structure (whitespace) but never content.
"""

import re
from pathlib import Path
from typing import List


def load_document(path: str) -> str:
    """Return the document text. Supports .docx (via python-docx) and .txt."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found at: {file_path}")

    if file_path.suffix.lower() == ".docx":
        from docx import Document  # imported lazily so .txt runs need no docx

        document = Document(str(file_path))
        paragraphs = [
            para.text.strip()
            for para in document.paragraphs
            if para.text and para.text.strip()
        ]
        return "\n\n".join(paragraphs)

    return file_path.read_text(encoding="utf-8")


def split_into_chunks(text: str) -> List[str]:
    """Split the document into one chunk per thematic paragraph."""
    blocks = re.split(r"\n\s*\n", text)  # split on one or more blank lines
    return [block.strip() for block in blocks if block.strip()]