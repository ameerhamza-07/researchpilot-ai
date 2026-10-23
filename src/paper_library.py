
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymupdf

from src.chunk_text import chunk_pages
from src.semantic_search import build_search_index


LIBRARY_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "library"
)


def get_paper_id(pdf_bytes: bytes) -> str:
    """Generate a stable ID based on PDF contents."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def add_paper(
    pdf_bytes: bytes,
    filename: str,
    embedding_model,
    library_dir: Path = LIBRARY_DIR,
) -> dict:
    """Process a PDF and save its index permanently."""

    if not pdf_bytes:
        raise ValueError("PDF is empty.")

    paper_id = get_paper_id(pdf_bytes)
    paper_dir = library_dir / paper_id
    metadata_path = paper_dir / "metadata.json"

    if metadata_path.exists():
        with metadata_path.open(
            "r", encoding="utf-8"
        ) as file:
            return json.load(file)

    pages = []

    try:
        with pymupdf.open(
            stream=pdf_bytes, filetype="pdf"
        ) as document:
            for page_number, page in enumerate(
                document, start=1
            ):
                text = page.get_text(
                    "text", sort=True
                ).strip()

                if text:
                    pages.append({
                        "page": page_number,
                        "text": text,
                    })
    except Exception as error:
        raise ValueError(
            f"Could not read PDF: {error}"
        ) from error

    if not pages:
        raise ValueError(
            "No extractable text found in this PDF."
        )

    chunks = chunk_pages(pages)
    embeddings = build_search_index(
        chunks, embedding_model
    )

    paper_dir.mkdir(parents=True, exist_ok=True)

    with (paper_dir / "chunks.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(chunks, file, ensure_ascii=False)

    np.save(
        paper_dir / "embeddings.npy",
        embeddings,
        allow_pickle=False,
    )

    metadata = {
        "paper_id": paper_id,
        "filename": Path(filename).name,
        "pages": len(pages),
        "chunks": len(chunks),
        "embedding_model": (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with metadata_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    return metadata


def list_papers(
    library_dir: Path = LIBRARY_DIR,
) -> list[dict]:
    """List all successfully indexed papers."""

    if not library_dir.exists():
        return []

    papers = []

    for metadata_path in library_dir.glob(
        "*/metadata.json"
    ):
        try:
            with metadata_path.open(
                "r", encoding="utf-8"
            ) as file:
                papers.append(json.load(file))
        except (OSError, json.JSONDecodeError):
            continue

    return sorted(
        papers,
        key=lambda paper: paper["created_at"],
        reverse=True,
    )


def load_paper(
    paper_id: str,
    library_dir: Path = LIBRARY_DIR,
) -> tuple[list[dict], np.ndarray]:
    """Load a saved paper's chunks and embeddings."""

    paper_dir = library_dir / paper_id

    with (paper_dir / "chunks.json").open(
        "r", encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    embeddings = np.load(
        paper_dir / "embeddings.npy",
        allow_pickle=False,
    )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Stored index is inconsistent."
        )

    return chunks, embeddings
