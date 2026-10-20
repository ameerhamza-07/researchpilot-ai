
from src.read_pdf import extract_pdf_text, PDF_PATH


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """Split extracted page text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size.")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"].strip()
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": chunk_text,
                    }
                )
                chunk_id += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks


def main():
    pages = extract_pdf_text(PDF_PATH)
    chunks = chunk_pages(pages)

    print(f"Paper: {PDF_PATH.name}")
    print(f"Extracted pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print(
            f"\n--- Chunk {chunk['chunk_id']} "
            f"(page {chunk['page']}) ---"
        )
        print(chunk["text"][:250])


if __name__ == "__main__":
    main()
