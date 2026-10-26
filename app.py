
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.paper_library import add_paper, list_papers, load_paper
from src.rag import answer_question
from src.cross_paper import answer_across_papers
from src.structured_compare import compare_papers_structured
from src.semantic_search import MODEL_NAME


st.set_page_config(
    page_title="ResearchPilot AI",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME, device="cpu")


def initialize_state():
    defaults = {
        "messages": [],
        "chat_context": None,
        "structured_comparison": None,
        "structured_comparison_ids": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sources(sources):
    if not sources:
        st.info("No supporting passages were retrieved.")
        return

    with st.expander(
        f"View retrieved evidence ({len(sources)} passages)"
    ):
        for source in sources:
            source_id = source.get("source_id", "Source")
            source_name = source.get("paper_name")
            page = source.get("page", "?")
            score = source.get("score")

            heading = f"**[{source_id}]"

            if source_name:
                heading += f" — {source_name}"

            heading += f" · Page {page}**"

            if score is not None:
                heading += f" · Similarity: {score:.3f}"

            st.markdown(heading)
            st.write(source.get("text", ""))
            st.divider()


def render_structured_comparison(result):
    rows = result.get("papers", [])

    if not rows:
        st.warning("No comparison results were returned.")
        return

    fields = [
        ("research_objective", "Research objective"),
        ("methodology", "Methodology"),
        ("dataset", "Dataset"),
        ("main_findings", "Main findings"),
        ("limitations", "Limitations"),
    ]

    table_rows = []

    for row in rows:
        table_row = {
            "Paper": row.get("paper_name", "Unknown paper")
        }

        for field, label in fields:
            table_row[label] = row.get(
                field,
                "Not established from retrieved passages",
            )

        table_rows.append(table_row)

    st.subheader("Comparison table")

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Model-generated summaries may be incomplete or incorrect. "
        "Verify important claims against the source passages and PDFs."
    )

    st.subheader("Supporting evidence")

    evidence_by_name = {
        paper.get("paper_name", "Unknown paper"): paper.get(
            "passages", []
        )
        for paper in result.get("retrieved_evidence", [])
    }

    for row in rows:
        paper_name = row.get("paper_name", "Unknown paper")

        with st.expander(f"Evidence: {paper_name}"):
            st.markdown("**Extracted details and evidence references**")

            refs = row.get("evidence_refs", {})

            for field, label in fields:
                field_refs = refs.get(field, [])

                st.markdown(f"**{label}**")

                if field_refs:
                    st.write(
                        "Model-provided references: "
                        + ", ".join(str(ref) for ref in field_refs)
                    )
                else:
                    st.caption("No evidence reference supplied.")

                st.write(
                    row.get(
                        field,
                        "Not established from retrieved passages",
                    )
                )

            st.divider()
            st.markdown("**Retrieved passages**")

            passages = evidence_by_name.get(paper_name, [])

            if not passages:
                st.info("No retrieved passages are available.")
            else:
                for passage in passages:
                    st.text(passage)
                    st.divider()


def main():
    initialize_state()

    st.title("📚 ResearchPilot AI")
    st.caption(
        "Explore research papers, ask evidence-grounded questions, "
        "and compare findings across documents."
    )

    # ------------------------------------------------------
    # Sidebar: upload papers and choose research mode
    # ------------------------------------------------------
    with st.sidebar:
        st.header("Research Library")

        uploaded_files = st.file_uploader(
            "Upload research papers",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button(
            "Add papers to library",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        ):
            try:
                embedding_model = load_embedding_model()
                added_count = 0

                for uploaded_file in uploaded_files:
                    try:
                        with st.spinner(
                            f"Processing {uploaded_file.name}..."
                        ):
                            metadata = add_paper(
                                pdf_bytes=uploaded_file.getvalue(),
                                filename=uploaded_file.name,
                                embedding_model=embedding_model,
                            )

                        st.success(f"Added: {metadata['filename']}")
                        added_count += 1

                    except Exception as error:
                        st.error(f"{uploaded_file.name}: {error}")

                if added_count:
                    st.rerun()

            except Exception as error:
                st.error(f"Could not load embedding model: {error}")

        papers = list_papers()

        if not papers:
            st.info("Upload a research PDF to get started.")
            st.stop()

        paper_options = {
            paper["paper_id"]: paper["filename"]
            for paper in papers
        }
        paper_ids = list(paper_options.keys())

        st.divider()
        st.header("Research mode")

        mode = st.radio(
            "Choose a workflow",
            [
                "Ask one paper",
                "Compare papers",
                "Structured comparison",
            ],
        )

        selected_ids = []

        if mode == "Ask one paper":
            selected_id = st.selectbox(
                "Select a paper",
                options=paper_ids,
                format_func=lambda pid: paper_options[pid],
            )
            selected_ids = [selected_id]

        else:
            default_ids = paper_ids[:2]

            selected_ids = st.multiselect(
                "Select papers to compare",
                options=paper_ids,
                default=default_ids,
                format_func=lambda pid: paper_options[pid],
                help="Select at least two papers for comparison.",
            )

        st.divider()
        st.caption("Language model: Qwen3 4B via Ollama")
        st.caption(f"Embedding model: {MODEL_NAME}")

        if st.button(
            "Clear conversation",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.session_state.chat_context = None
            st.rerun()

    # ------------------------------------------------------
    # Structured comparison mode
    # ------------------------------------------------------
    if mode == "Structured comparison":
        st.subheader("Structured research comparison")
        st.write(
            "Compare research objectives, methodologies, datasets, "
            "findings, and limitations across selected papers."
        )

        if len(selected_ids) < 2:
            st.warning(
                "Select at least two papers in the sidebar."
            )
            return

        current_comparison_ids = tuple(selected_ids)

        if st.session_state.structured_comparison_ids != current_comparison_ids:
            st.session_state.structured_comparison = None
            st.session_state.structured_comparison_ids = (
                current_comparison_ids
            )

        selected_names = [
            paper_options[paper_id] for paper_id in selected_ids
        ]

        with st.expander("Selected papers", expanded=True):
            for name in selected_names:
                st.write(f"• {name}")

        if st.button(
            "Generate comparison",
            type="primary",
            use_container_width=True,
        ):
            try:
                embedding_model = load_embedding_model()

                with st.spinner(
                    "Retrieving passages and generating comparison..."
                ):
                    result = compare_papers_structured(
                        paper_ids=selected_ids,
                        embedding_model=embedding_model,
                        top_k_per_paper=2,
                    )

                st.session_state.structured_comparison = result
                st.session_state.structured_comparison_ids = (
                    current_comparison_ids
                )

            except Exception as error:
                st.error(f"Comparison failed: {error}")
                st.caption(
                    "Check that Ollama is running, qwen3:4b is installed, "
                    "and the paper indexes are valid."
                )

        result = st.session_state.structured_comparison

        if (
            result
            and st.session_state.structured_comparison_ids
            == current_comparison_ids
        ):
            render_structured_comparison(result)

        return

    # ------------------------------------------------------
    # Validate selections for Q&A modes
    # ------------------------------------------------------
    if mode == "Compare papers" and len(selected_ids) < 2:
        st.warning(
            "Select at least two papers in the sidebar "
            "to use comparison mode."
        )
        return

    # Reset chat when mode or selected papers change.
    current_context = (mode, tuple(selected_ids))

    if st.session_state.chat_context != current_context:
        st.session_state.chat_context = current_context
        st.session_state.messages = []

    # ------------------------------------------------------
    # Display active research context
    # ------------------------------------------------------
    if mode == "Ask one paper":
        paper_name = paper_options[selected_ids[0]]

        try:
            chunks, embeddings = load_paper(selected_ids[0])
        except Exception as error:
            st.error(f"Could not load paper: {error}")
            return

        st.success(
            f"Active paper: {paper_name} · {len(chunks)} chunks"
        )

    else:
        selected_names = [
            paper_options[pid] for pid in selected_ids
        ]

        st.success(
            f"Comparison mode · {len(selected_ids)} papers selected"
        )

        with st.expander("Selected papers", expanded=True):
            for name in selected_names:
                st.write(f"• {name}")

    # ------------------------------------------------------
    # Render existing conversation
    # ------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    placeholder = (
        "Ask about this paper..."
        if mode == "Ask one paper"
        else "Ask a question comparing the selected papers..."
    )

    question = st.chat_input(placeholder)

    if not question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Retrieving evidence and generating an answer..."
            ):
                embedding_model = load_embedding_model()

                if mode == "Ask one paper":
                    result = answer_question(
                        question=question,
                        chunks=chunks,
                        embeddings=embeddings,
                        embedding_model=embedding_model,
                    )
                else:
                    result = answer_across_papers(
                        question=question,
                        paper_ids=selected_ids,
                        embedding_model=embedding_model,
                        top_k_per_paper=2,
                    )

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)
            render_sources(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            st.error(f"Could not generate an answer: {error}")
            st.caption(
                "Check that Ollama is running, qwen3:4b is installed, "
                "and the selected paper indexes are valid."
            )


if __name__ == "__main__":
    main()
