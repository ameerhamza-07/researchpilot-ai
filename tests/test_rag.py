
import numpy as np

from src import rag


class FakeEmbeddingModel:
    """Deterministic embeddings for testing retrieval."""

    def encode(self, text, **kwargs):
        if text == "attention question":
            return np.array([1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 1.0], dtype=np.float32)


def sample_chunks():
    return [
        {
            "chunk_id": 0,
            "page": 2,
            "text": "Self-attention relates words to other words.",
        },
        {
            "chunk_id": 1,
            "page": 5,
            "text": "The training procedure uses optimization.",
        },
    ]


def sample_embeddings():
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_answer_question_returns_answer_and_sources(monkeypatch):
    def fake_chat(**kwargs):
        # Verify that retrieved evidence is passed to the LLM.
        messages = kwargs["messages"]
        assert "Self-attention relates words" in messages[1]["content"]

        class Message:
            content = "Self-attention relates words to other words. [S1, Page 2]"

        class Response:
            message = Message()

        return Response()

    monkeypatch.setattr(rag, "chat", fake_chat)

    result = rag.answer_question(
        question="attention question",
        chunks=sample_chunks(),
        embeddings=sample_embeddings(),
        embedding_model=FakeEmbeddingModel(),
        top_k=1,
    )

    assert "Self-attention" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_id"] == "S1"
    assert result["sources"][0]["page"] == 2


def test_blank_question_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="must not be empty"):
        rag.answer_question(
            question="   ",
            chunks=sample_chunks(),
            embeddings=sample_embeddings(),
            embedding_model=FakeEmbeddingModel(),
        )


def test_no_chunks_returns_empty_sources(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("LLM should not be called without sources")

    monkeypatch.setattr(rag, "chat", fail_if_called)

    result = rag.answer_question(
        question="attention question",
        chunks=[],
        embeddings=np.empty((0, 0), dtype=np.float32),
        embedding_model=FakeEmbeddingModel(),
    )

    assert result["sources"] == []
    assert "No indexed paper passages" in result["answer"]
