
import json
import re
import time

import ollama

from src.paper_library import load_paper, list_papers
from src.semantic_search import search_chunks


OLLAMA_MODEL = "qwen3:4b"

DEFAULT_TOP_K_PER_PAPER = 2
MAX_CONTEXT_CHARS_PER_PAPER = 3000
MAX_OUTPUT_TOKENS = 1536

FIELD_NAMES = [
    "research_objective",
    "methodology",
    "dataset",
    "main_findings",
    "limitations",
]

NOT_FOUND = "Not established from retrieved passages"


def _get_value(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _get_paper_metadata(paper_id):
    for paper in list_papers():
        if paper.get("paper_id") == paper_id:
            return paper

    raise ValueError(f"Paper ID '{paper_id}' was not found.")


def _extract_json(raw_content):
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ValueError("Ollama returned empty response content.")

    content = raw_content.strip()

    content = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"\s*```\s*$", "", content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Ollama returned invalid JSON. "
            f"Error: {exc.msg}, line {exc.lineno}, "
            f"column {exc.colno}. "
            f"Response preview: {content[:250]!r}"
        ) from exc


def _retrieve_paper_evidence(
    paper_id,
    embedding_model,
    top_k_per_paper,
):
    metadata = _get_paper_metadata(paper_id)
    paper_name = metadata.get("filename", str(paper_id))

    chunks, embeddings = load_paper(paper_id)

    if not chunks:
        raise ValueError(f"No indexed chunks found for '{paper_name}'.")

    query = (
        "research objective methodology model experimental setup "
        "dataset benchmark results findings limitations"
    )

    # Correct signature: semantic_search.search_chunks uses model=.
    results = search_chunks(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        model=embedding_model,
        top_k=top_k_per_paper,
    )

    passages = []
    remaining = MAX_CONTEXT_CHARS_PER_PAPER

    for index, result in enumerate(results, start=1):
        text = str(result.get("text", "")).strip()

        if not text or remaining <= 0:
            continue

        excerpt = text[:remaining]
        remaining -= len(excerpt)

        passages.append(
            {
                "source_id": f"S{index}",
                "page": result.get(
                    "page",
                    result.get("page_number", "?"),
                ),
                "text": excerpt,
            }
        )

    if not passages:
        raise ValueError(
            f"No searchable evidence found for '{paper_name}'."
        )

    evidence_text = "\n\n".join(
        f"[{p['source_id']} | PAGE {p['page']}]\n{p['text']}"
        for p in passages
    )

    return {
        "paper_id": paper_id,
        "paper_name": paper_name,
        "passages": passages,
        "evidence_text": evidence_text,
    }


def _build_prompt(papers):
    paper_blocks = []

    for paper in papers:
        paper_blocks.append(
            f"FILENAME: {paper['paper_name']}\n"
            f"PAPER ID: {paper['paper_id']}\n"
            f"EVIDENCE:\n{paper['evidence_text']}"
        )

    evidence = "\n\n---\n\n".join(paper_blocks)

    schema = {
        "papers": [
            {
                "paper_name": "exact filename",
                "research_objective": "concise string",
                "methodology": "concise string",
                "dataset": "concise string",
                "main_findings": "concise string",
                "limitations": "concise string",
                "evidence_refs": {
                    "research_objective": ["S1"],
                    "methodology": ["S1"],
                    "dataset": [],
                    "main_findings": ["S2"],
                    "limitations": [],
                },
            }
        ]
    }

    return f"""
/no_think

Extract a concise structured summary for EACH supplied research paper.
Use only that paper's evidence. Do not mix evidence between papers.
Do not invent facts. Keep each field to one short sentence.
If information is absent, use exactly:
"{NOT_FOUND}"

For evidence_refs, use only source IDs belonging to that paper.
Use [] when there is no supporting evidence.

Return only a valid JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False)}

There must be exactly {len(papers)} paper objects.
Copy each filename exactly as supplied.

PAPER EVIDENCE:
{evidence}
""".strip()


def _normalise_comparison(comparison, papers):
    if not isinstance(comparison, dict):
        raise ValueError("The JSON root must be an object.")

    rows = comparison.get("papers")

    if not isinstance(rows, list):
        raise ValueError("JSON is missing the 'papers' list.")

    rows_by_name = {}

    for row in rows:
        if isinstance(row, dict):
            name = row.get("paper_name")
            if isinstance(name, str):
                rows_by_name[name.strip().casefold()] = row

    normalised_rows = []
    evidence_by_name = {}

    for paper in papers:
        name = paper["paper_name"]
        row = rows_by_name.get(name.strip().casefold())

        if row is None:
            raise ValueError(
                f"The model omitted the result for '{name}'. "
                "Try comparing two papers."
            )

        allowed_ids = {
            passage["source_id"] for passage in paper["passages"]
        }

        raw_refs = row.get("evidence_refs", {})
        if not isinstance(raw_refs, dict):
            raw_refs = {}

        clean_row = {"paper_name": name}
        clean_refs = {}

        for field in FIELD_NAMES:
            value = row.get(field)

            if not isinstance(value, str) or not value.strip():
                value = NOT_FOUND

            clean_row[field] = value.strip()

            refs = raw_refs.get(field, [])
            if not isinstance(refs, list):
                refs = []

            clean_refs[field] = [
                str(ref)
                for ref in refs
                if str(ref) in allowed_ids
            ]

        clean_row["evidence_refs"] = clean_refs
        normalised_rows.append(clean_row)

        evidence_by_name[name] = [
            (
                f"[{p['source_id']}] Page {p['page']}\n"
                f"{p['text']}"
            )
            for p in paper["passages"]
        ]

    return {
        "papers": normalised_rows,
        "retrieved_evidence": [
            {
                "paper_name": paper["paper_name"],
                "passages": evidence_by_name[paper["paper_name"]],
            }
            for paper in papers
        ],
    }


def compare_papers_structured(
    paper_ids,
    embedding_model,
    top_k_per_paper=DEFAULT_TOP_K_PER_PAPER,
):
    """Retrieve evidence and compare papers using local Ollama."""

    started = time.perf_counter()

    if not paper_ids or len(paper_ids) < 2:
        raise ValueError("Select at least two papers.")

    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Duplicate paper IDs are not allowed.")

    if top_k_per_paper < 1:
        raise ValueError("top_k_per_paper must be at least 1.")

    papers = []

    for paper_id in paper_ids:
        paper_started = time.perf_counter()

        paper = _retrieve_paper_evidence(
            paper_id=paper_id,
            embedding_model=embedding_model,
            top_k_per_paper=top_k_per_paper,
        )
        papers.append(paper)

        print(
            f"[STRUCTURED] Retrieved {paper['paper_name']} in "
            f"{time.perf_counter() - paper_started:.2f}s",
            flush=True,
        )

    prompt = _build_prompt(papers)

    print(
        f"[STRUCTURED] Prompt length: {len(prompt)} characters; "
        f"papers: {len(papers)}",
        flush=True,
    )
    print("[STRUCTURED] Calling Ollama...", flush=True)

    llm_started = time.perf_counter()

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract concise evidence-grounded facts. "
                        "Return the requested JSON and nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            think=False,
            options={
                "temperature": 0,
                "num_predict": MAX_OUTPUT_TOKENS,
            },
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}. "
            "Check that Ollama is running and qwen3:4b is installed."
        ) from exc

    print(
        f"[STRUCTURED] Ollama finished in "
        f"{time.perf_counter() - llm_started:.2f}s",
        flush=True,
    )

    message = _get_value(response, "message")

    if message is None:
        raise ValueError("Ollama returned no message.")

    raw_content = _get_value(message, "content", "")
    done_reason = _get_value(response, "done_reason", "unknown")

    if not isinstance(raw_content, str) or not raw_content.strip():
        thinking = _get_value(message, "thinking", "")
        raise ValueError(
            "Ollama returned empty content. "
            f"Done reason: {done_reason}. "
            f"Thinking length: "
            f"{len(thinking) if isinstance(thinking, str) else 0}. "
            "If think=False is unsupported, upgrade the Ollama Python "
            "package and retry."
        )

    print(
        f"[STRUCTURED] Response length: {len(raw_content)}; "
        f"done reason: {done_reason}",
        flush=True,
    )

    comparison = _extract_json(raw_content)
    result = _normalise_comparison(comparison, papers)

    print(
        f"[STRUCTURED] Total time: "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )

    return result
