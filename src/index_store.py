
import json
from pathlib import Path

import numpy as np


INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"


def save_index(
    chunks: list[dict],
    embeddings: np.ndarray,
    index_dir: Path = INDEX_DIR,
) -> None:
    """Save chunks and embedding vectors to disk."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    index_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"

    with chunks_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    np.save(embeddings_path, embeddings, allow_pickle=False)

    print(f"Saved chunks to: {chunks_path}")
    print(f"Saved embeddings to: {embeddings_path}")


def load_index(
    index_dir: Path = INDEX_DIR,
) -> tuple[list[dict], np.ndarray]:
    """Load chunks and embeddings from disk."""

    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"

    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Index files not found in {index_dir}"
        )

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(embeddings_path, allow_pickle=False)

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Index is inconsistent: chunk and embedding counts differ."
        )

    return chunks, embeddings
