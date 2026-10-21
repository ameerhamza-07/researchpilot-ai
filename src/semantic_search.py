
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunk_text import chunk_pages
from src.read_pdf import extract_pdf_text, PDF_PATH


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_search_index(
    chunks: list[dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """Convert chunk text into normalized embedding vectors."""

    if not chunks:
        return np.empty((0, 0), dtype=np.float32)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings


def search_chunks(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    """Return the most semantically relevant chunks."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not query.strip():
        raise ValueError("Query must not be empty.")

    if not chunks:
        return []

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Normalized vectors: dot product equals cosine similarity.
    scores = embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        result = dict(chunks[int(index)])
        result["score"] = float(scores[index])
        results.append(result)

    return results


def main():
    pages = extract_pdf_text(PDF_PATH)
    chunks = chunk_pages(pages)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    embeddings = build_search_index(chunks, model)

    print(f"\nIndexed {len(chunks)} chunks.")
    print(f"Embedding shape: {embeddings.shape}")

    while True:
        query = input(
            "\nAsk a question about the paper (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        try:
            results = search_chunks(
                query, chunks, embeddings, model, top_k=3
            )
        except ValueError as error:
            print(error)
            continue

        for rank, result in enumerate(results, start=1):
            print(
                f"\n#{rank} | Page {result['page']} "
                f"| Similarity: {result['score']:.3f}"
            )
            print(result["text"][:700])


if __name__ == "__main__":
    main()
