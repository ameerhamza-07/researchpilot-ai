
import json

import numpy as np
import pytest

from src.index_store import load_index, save_index


def test_save_and_load_index(tmp_path):
    chunks = [
        {"chunk_id": 0, "page": 1, "text": "Attention is useful."},
        {"chunk_id": 1, "page": 2, "text": "Transformers process sequences."},
    ]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )

    save_index(chunks, embeddings, tmp_path)
    loaded_chunks, loaded_embeddings = load_index(tmp_path)

    assert loaded_chunks == chunks
    np.testing.assert_array_equal(loaded_embeddings, embeddings)


def test_missing_index_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path)


def test_mismatched_counts_are_rejected(tmp_path):
    chunks = [{"chunk_id": 0, "page": 1, "text": "Example"}]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )

    with pytest.raises(ValueError):
        save_index(chunks, embeddings, tmp_path)
