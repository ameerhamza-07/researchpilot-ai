
from pathlib import Path

import pymupdf
import pytest

from src.read_pdf import extract_pdf_text


def test_extracts_text_and_page_numbers(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"

    # Create a small test PDF without needing an external file.
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "ResearchPilot test paper")
        document.save(pdf_path)

    pages = extract_pdf_text(pdf_path)

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "ResearchPilot test paper" in pages[0]["text"]


def test_missing_pdf_raises_error(tmp_path: Path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf_text(missing_path)
