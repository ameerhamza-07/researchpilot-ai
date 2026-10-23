
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.paper_library import add_paper, list_papers, load_paper
from src.rag import answer_question
from src.semantic_search import MODEL_NAME


st.set_page_config(
    page_title="ResearchPilot AI",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_embedding_model():
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


def initialize_state():
    """Initialize Streamlit session state."""
    defaults = {
        "messages": [],
        "selected_paper_id": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sources(sources):
    """Display the passages retrieved for an answer."""
    if not sources:
        st.info("No supporting passages were retrieved.")
        return

    with st.expander(
        f"View retrieved evidence ({len(sources)} passages)"
    ):
        for source in sources:
            st.markdown(
                f"**[{source['source_id']}] "
                f"Page {source['page']}** · "
                f"Similarity: {source['score']:.3f}"
            )
            st.write(source["text"])
            st.divider()


def main():
    initialize_state()

    st.title("📚 ResearchPilot AI")
    st.caption(
        "Your local AI research assistant. Upload research papers, "
        "ask questions, and inspect the supporting passages."
    )

    # ---------------------------------------------------------
    # Sidebar: upload papers and manage the research library
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("Research Library")

        uploaded_files = st.file_uploader(
            "Upload research papers",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload text-based PDF research papers.",
        )

        if st.button(
            "Add papers to library",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        ):
            try:
                embedding_model = load_embedding_model()

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

                        st.success(
                            f"Added: {metadata['filename']}"
                        )

                    except Exception as error:
                        st.error(
                            f"Failed to process "
                            f"{uploaded_file.name}: {error}"
                        )

            except Exception as error:
                st.error(f"Could not load embedding model: {error}")

        st.divider()
        st.subheader("Your papers")

        papers = list_papers()

        if not papers:
            st.info(
                "Your library is empty. Upload a PDF and click "
                "'Add papers to library' to get started."
            )
            st.stop()

        paper_options = {
            paper["paper_id"]: paper["filename"]
            for paper in papers
        }

        selected_id = st.selectbox(
            "Select a paper",
            options=list(paper_options.keys()),
            format_func=lambda paper_id: paper_options[paper_id],
            key="paper_selector",
        )

        st.divider()
        st.subheader("Model configuration")
        st.write("**Language model:** Qwen3 4B")
        st.write("**Embeddings:** MiniLM")
        st.write("**Inference:** Local via Ollama")

        if st.button(
            "Clear conversation",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "Paper indexes are stored locally in data/library/."
        )

    # ---------------------------------------------------------
    # Load the selected paper and reset chat when switching
    # ---------------------------------------------------------
    if st.session_state.selected_paper_id != selected_id:
        st.session_state.selected_paper_id = selected_id
        st.session_state.messages = []

    paper_name = paper_options[selected_id]

    try:
        chunks, embeddings = load_paper(selected_id)
    except Exception as error:
        st.error(f"Could not load '{paper_name}': {error}")
        st.stop()

    st.success(
        f"Active paper: {paper_name} · {len(chunks)} chunks"
    )

    # ---------------------------------------------------------
    # Display previous conversation messages
    # ---------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    # ---------------------------------------------------------
    # Chat input and RAG question answering
    # ---------------------------------------------------------
    question = st.chat_input(
        f"Ask a question about {paper_name}..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching the paper and generating an answer..."
                ):
                    embedding_model = load_embedding_model()

                    result = answer_question(
                        question=question,
                        chunks=chunks,
                        embeddings=embeddings,
                        embedding_model=embedding_model,
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
                st.error(
                    f"Could not generate an answer: {error}"
                )
                st.caption(
                    "Check that Ollama is running, qwen3:4b is "
                    "installed, and the selected paper index is valid."
                )


if __name__ == "__main__":
    main()
