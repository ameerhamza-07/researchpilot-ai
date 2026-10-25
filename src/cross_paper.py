
from ollama import chat
from sentence_transformers import SentenceTransformer

from src.paper_library import list_papers, load_paper
from src.semantic_search import search_chunks


OLLAMA_MODEL = "qwen3:4b"


def answer_across_papers(
    question: str,
    paper_ids: list[str],
    embedding_model: SentenceTransformer,
    top_k_per_paper: int = 2,
) -> dict:
    """Retrieve evidence from multiple papers and synthesize an answer."""

    if not question.strip():
        raise ValueError("Question must not be empty.")

    if not paper_ids:
        raise ValueError("Select at least one paper.")

    if top_k_per_paper <= 0:
        raise ValueError("top_k_per_paper must be greater than zero.")

    # Map stored IDs to paper metadata.
    papers_by_id = {
        paper["paper_id"]: paper
        for paper in list_papers()
    }

    unknown_ids = [
        paper_id
        for paper_id in paper_ids
        if paper_id not in papers_by_id
    ]

    if unknown_ids:
        raise ValueError(
            f"Unknown paper IDs: {', '.join(unknown_ids)}"
        )

    evidence = []

    for paper_id in paper_ids:
        paper = papers_by_id[paper_id]
        chunks, embeddings = load_paper(paper_id)

        retrieved = search_chunks(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            model=embedding_model,
            top_k=top_k_per_paper,
        )

        for item in retrieved:
            evidence.append({
                "source_id": f"S{len(evidence) + 1}",
                "paper_id": paper_id,
                "paper_name": paper["filename"],
                "chunk_id": item["chunk_id"],
                "page": item["page"],
                "score": item["score"],
                "text": item["text"],
            })

    if not evidence:
        return {
            "answer": "No relevant passages were found in the selected papers.",
            "sources": [],
        }

    context_parts = []

    for item in evidence:
        context_parts.append(
            f"[{item['source_id']} | Paper: {item['paper_name']} "
            f"| Page {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are ResearchPilot, a research literature "
                    "analysis assistant. Answer using only the supplied "
                    "excerpts. Compare the papers when relevant, "
                    "distinguishing their methods, findings, and "
                    "limitations. Support factual claims with citations "
                    "using the exact source ID, paper name, and page, "
                    "such as [S1, Page 4]. Use only the source IDs and "
                    "pages provided. Never invent findings or citations. "
                    "If the evidence is insufficient, say so. Do not "
                    "assume that a missing detail means a paper lacks "
                    "that feature."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved evidence:\n{context}\n\n"
                    "Answer clearly. When comparing papers, explain "
                    "similarities and differences supported by the "
                    "provided excerpts."
                ),
            },
        ],
        options={"temperature": 0},
    )

    return {
        "answer": response.message.content,
        "sources": evidence,
    }
