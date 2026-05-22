from typing import List
import os
import re

from pypdf import PdfReader


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end == length:
            chunks.append(text[start:end].strip())
            break

        # Try to split at a sentence boundary inside the window
        window = text[start:end]
        last_period = window.rfind('. ')
        if last_period != -1 and last_period > int(chunk_size * 0.25):
            split_at = start + last_period + 1
            chunks.append(text[start:split_at].strip())
            start = split_at + 1
            continue

        # Fallback: split on last space
        last_space = window.rfind(' ')
        if last_space != -1:
            split_at = start + last_space
            chunks.append(text[start:split_at].strip())
            start = split_at + 1
            continue

        # Hard cut
        chunks.append(text[start:end].strip())
        start = end

    return [c for c in chunks if c]


def load_document(path: str, chunk_size: int = 1000) -> List[str]:
    """Load text from a PDF or text file and split into chunks.

    Returns a list of string chunks (may be empty on failure).
    """
    ext = os.path.splitext(path)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            reader = PdfReader(path)
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    # ignore page-level extraction errors
                    pages.append("")
            text = "\n\n".join(pages)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception:
        return []

    return _chunk_text(text, chunk_size)
