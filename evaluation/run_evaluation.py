
import csv
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.paper_library import load_paper
from src.rag import answer_question
from src.semantic_search import MODEL_NAME


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "evaluation" / "evaluation_dataset.csv"
RESULTS_PATH = ROOT_DIR / "evaluation" / "evaluation_results.csv"
SUMMARY_PATH = ROOT_DIR / "evaluation" / "evaluation_summary.json"

TOP_K = 3


def load_dataset():
    with DATASET_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    required = {
        "question_id",
        "paper_id",
        "question",
        "expected_answer",
        "expected_page",
        "question_type",
    }

    if not rows:
        raise ValueError("The evaluation dataset is empty.")

    if not required.issubset(rows[0].keys()):
        raise ValueError(
            f"Dataset must contain these columns: {sorted(required)}"
        )

    return rows


def load_question_index(paper_ids, cache):
    all_chunks = []
    all_embeddings = []

    for paper_id in paper_ids:
        paper_id = paper_id.strip()

        if not paper_id:
            continue

        if paper_id not in cache:
            cache[paper_id] = load_paper(paper_id)

        chunks, embeddings = cache[paper_id]

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding mismatch for paper {paper_id}"
            )

        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    if not all_chunks:
        raise ValueError("No paper chunks were loaded.")

    embeddings = np.concatenate(all_embeddings, axis=0)

    if len(all_chunks) != len(embeddings):
        raise ValueError("Combined index is inconsistent.")

    return all_chunks, embeddings


def main():
    rows = load_dataset()
    paper_cache = {}

    print(f"Evaluation questions: {len(rows)}")
    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    results = []

    for number, row in enumerate(rows, start=1):
        question_id = row["question_id"].strip()
        question = row["question"].strip()
        paper_ids = row["paper_id"].split(";")
        question_type = row["question_type"].strip().lower()

        print(
            f"\n[{number}/{len(rows)}] "
            f"{question_id}: {question}"
        )

        result_row = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "expected_answer": row["expected_answer"],
            "expected_page": row["expected_page"],
            "generated_answer": "",
            "retrieved_pages": "",
            "retrieval_scores": "",
            "retrieved_chunks": "",
            "latency_seconds": "",
            "status": "ok",
            "error": "",
        }

        try:
            chunks, embeddings = load_question_index(
                paper_ids,
                paper_cache,
            )

            start_time = time.perf_counter()

            answer_result = answer_question(
                question=question,
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=embedding_model,
                top_k=TOP_K,
            )

            elapsed = time.perf_counter() - start_time
            sources = answer_result.get("sources", [])

            result_row.update({
                "generated_answer": answer_result.get("answer", ""),
                "retrieved_pages": ";".join(
                    str(source["page"]) for source in sources
                ),
                "retrieval_scores": ";".join(
                    f"{source['score']:.4f}" for source in sources
                ),
                "retrieved_chunks": json.dumps(
                    [
                        {
                            "source_id": source["source_id"],
                            "page": source["page"],
                            "chunk_id": source["chunk_id"],
                            "score": source["score"],
                            "text": source["text"],
                        }
                        for source in sources
                    ],
                    ensure_ascii=False,
                ),
                "latency_seconds": f"{elapsed:.2f}",
            })

            print(f"Answer: {result_row['generated_answer'][:250]}")
            print(f"Retrieved pages: {result_row['retrieved_pages']}")
            print(f"Latency: {elapsed:.2f}s")

        except Exception as error:
            result_row["status"] = "error"
            result_row["error"] = str(error)
            print(f"ERROR: {error}")

        results.append(result_row)

    fieldnames = list(results[0].keys())

    with RESULTS_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    successful = [
        result for result in results if result["status"] == "ok"
    ]

    summary = {
        "total_questions": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "question_types": {},
        "average_latency_seconds": (
            round(
                sum(float(r["latency_seconds"]) for r in successful)
                / len(successful),
                2,
            )
            if successful else None
        ),
        "note": (
            "Answer correctness and retrieval relevance require "
            "separate human review; successful execution does not "
            "mean an answer is correct."
        ),
    }

    for result in results:
        kind = result["question_type"]
        summary["question_types"].setdefault(
            kind, {"total": 0, "successful": 0}
        )
        summary["question_types"][kind]["total"] += 1

        if result["status"] == "ok":
            summary["question_types"][kind]["successful"] += 1

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n--- Evaluation complete ---")
    print(f"Results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Successful: {summary['successful']}/{summary['total_questions']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average latency: {summary['average_latency_seconds']} seconds")


if __name__ == "__main__":
    main()
