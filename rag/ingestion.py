import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Document:
    """A single document loaded from source file"""
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A single retrievable chunk with provenance tracking"""
    text: str
    source:str
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def load_text_file(path: str) -> Document:
    """Load a plain text file as a Document"""
    text = Path(path).read_text(encoding="utf-8")
    return Document(content=text, source=path)

def load_pdf(path: str) -> Document:
        """Load a PDF file, extracting text from all pages """
        try:
            import PyPDF2
            text_parts = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return Document(
                 content= "\n\n".join(text_parts),
                 source= path,
                 metadata={"num_pages": len(reader.pages)}
            )

        except ImportError:
            raise ImportError("PyPDF2 required for PDF loading: pip install PyPDF2")


def recursive_chunk(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[list[str]] = None,
) -> list[Chunk]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
        # Removed "" (character-level) — causes list-of-chars bug

    def _split_text(text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators in priority order."""
        if not separators:
            # Base case: no more separators to try, return text as-is
            return [text] if text.strip() else []

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator) if separator else [text]

        good_splits: list[str] = []
        current: list[str] = []
        current_len = 0

        for split in splits:
            if not split.strip():
                continue  # skip empty splits

            split_len = len(split.split())

            if current_len + split_len <= chunk_size:
                current.append(split)
                current_len += split_len
            else:
                if current:
                    # Join accumulated splits into one chunk string
                    good_splits.append(separator.join(current))
                # If single split is too large and we have more separators,
                # recursively split it further
                if split_len > chunk_size and remaining_separators:
                    sub_splits = _split_text(split, remaining_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Accept oversized chunk rather than splitting further
                    good_splits.append(split)
                current = []
                current_len = 0

        # Don't forget remaining accumulated splits
        if current:
            good_splits.append(separator.join(current))

        return [s for s in good_splits if isinstance(s, str) and s.strip()]

    raw_chunks = _split_text(document.content, separators)

    # Apply overlap
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            prev_words = raw_chunks[i - 1].split()
            overlap_text = " ".join(prev_words[-chunk_overlap:])
            chunk_text = overlap_text + " " + chunk_text

        chunks.append(Chunk(
            text=chunk_text.strip(),
            source=document.source,
            chunk_id=i,
            metadata={**document.metadata, "chunk_index": i}
        ))

    return chunks