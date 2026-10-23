
import numpy as np

from src.paper_library import (
    get_paper_id,
    list_papers,
    load_paper,
    add_paper,
)


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.ones(
            (len(texts), 3),
            dtype=np.float32,
        )


def make_pdf():
    import pymupdf

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Transformers use attention mechanisms.",
        )
        return document.tobytes()


def test_paper_id_is_deterministic():
    content = b"sample pdf bytes"

    assert get_paper_id(content) == get_paper_id(content)


def test_add_and_load_paper(tmp_path):
    pdf_bytes = make_pdf()

    metadata = add_paper(
        pdf_bytes,
        "sample.pdf",
        FakeEmbeddingModel(),
        tmp_path,
    )

    chunks, embeddings = load_paper(
        metadata["paper_id"],
        tmp_path,
    )

    assert len(chunks) > 0
    assert len(chunks) == len(embeddings)
    assert metadata["filename"] == "sample.pdf"


def test_duplicate_upload(tmp_path):
    pdf_bytes = make_pdf()
    model = FakeEmbeddingModel()

    first = add_paper(
        pdf_bytes, "paper.pdf", model, tmp_path
    )

    second = add_paper(
        pdf_bytes, "paper.pdf", model, tmp_path
    )

    assert first["paper_id"] == second["paper_id"]
    assert len(list_papers(tmp_path)) == 1
