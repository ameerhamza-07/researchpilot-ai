
from pathlib import Path

import pymupdf


PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "papers"
    / "attention_is_all_you_need.pdf"
)


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """Extract non-empty page text with original page numbers."""

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    extracted_pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(
            document, start=1
        ):
            text = page.get_text("text", sort=True).strip()

            if text:
                extracted_pages.append({
                    "page": page_number,
                    "text": text,
                })

    if not extracted_pages:
        raise ValueError(
            "No extractable text found in the PDF."
        )

    return extracted_pages


def main():
    pages = extract_pdf_text(PDF_PATH)

    print(f"PDF: {PDF_PATH.name}")
    print(f"Pages containing text: {len(pages)}")

    for page in pages[:2]:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:700])


if __name__ == "__main__":
    main()
