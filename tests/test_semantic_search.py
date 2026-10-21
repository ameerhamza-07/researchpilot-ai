
import numpy as np
import pytest

from src.semantic_search import search_chunks


class FakeModel:
    def encode(self, text, **kwargs):
        vectors = {
            "attention": np.array([1.0, 0.0], dtype=np.float32),
            "question": np.array([0.9, 0.1], dtype=np.float32),
        }
        return vectors[text]


def test_search_ranks_most_similar_chunk_first():
    chunks = [
        {"chunk_id": 0, "page": 1, "text": "Attention passage"},
        {"chunk_id": 1, "page": 2, "text": "Other passage"},
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    results = search_chunks(
        "question", chunks, embeddings, FakeModel(), top_k=2
    )

    assert results[0]["chunk_id"] == 0
    assert results[0]["score"] > results[1]["score"]


def test_empty_chunks_return_empty_results():
    results = search_chunks(
        "question",
        [],
        np.empty((0, 0), dtype=np.float32),
        FakeModel(),
    )

    assert results == []


def test_blank_query_is_rejected():
    with pytest.raises(ValueError):
        search_chunks(
            "  ",
            [{"chunk_id": 0, "page": 1, "text": "Example"}],
            np.array([[1.0, 0.0]], dtype=np.float32),
            FakeModel(),
        )

