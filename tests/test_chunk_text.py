
import pytest

from src.chunk_text import chunk_pages


def test_chunks_long_text_with_overlap():
    pages = [{"page": 4, "text": "A" * 2500}]

    chunks = chunk_pages(pages, chunk_size=1000, overlap=150)

    assert len(chunks) == 3
    assert chunks[0]["text"] == "A" * 1000
    assert chunks[1]["text"] == "A" * 1000
    assert chunks[2]["text"] == "A" * 800
    assert all(chunk["page"] == 4 for chunk in chunks)


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([]) == []


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages(
            [{"page": 1, "text": "Some text"}],
            chunk_size=100,
            overlap=100,
        )


def test_chunk_ids_are_sequential():
    pages = [
        {"page": 1, "text": "A" * 120},
        {"page": 2, "text": "B" * 120},
    ]

    chunks = chunk_pages(pages, chunk_size=100, overlap=10)

    assert [chunk["chunk_id"] for chunk in chunks] == list(
        range(len(chunks))
    )
