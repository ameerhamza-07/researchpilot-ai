
import hashlib

import pymupdf
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.chunk_text import chunk_pages
from src.rag import answer_question
from src.semantic_search import MODEL_NAME, build_search_index
from src.index_store import load_index


st.set_page_config(
    page_title="ResearchPilot AI",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource
def load_saved_index():
    return load_index()


def initialize_state():
    defaults = {
        "messages": [],
        "active_chunks": None,
        "active_embeddings": None,
        "active_paper_name": None,
        "active_pdf_hash": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_pdf(pdf_bytes: bytes):
    """Extract, chunk, and embed an uploaded PDF."""

    pages = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned "
            "or image-only; OCR support is not implemented yet."
        )

    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    model = load_embedding_model()
    embeddings = build_search_index(chunks, model)

    return chunks, embeddings, len(pages)


def main():
    initialize_state()

    st.title("📚 ResearchPilot AI")
    st.caption(
        "Upload a research paper, ask questions, and inspect "
        "the passages used to generate answers."
    )

    with st.sidebar:
        st.header("Research paper")

        uploaded_file = st.file_uploader(
            "Upload a PDF paper",
            type=["pdf"],
            help="Text-based PDFs are supported. Scanned PDFs need OCR.",
        )

        if uploaded_file is not None:
            pdf_bytes = uploaded_file.getvalue()
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

            if st.button(
                "Process PDF",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Extracting text and generating embeddings..."
                    ):
                        chunks, embeddings, page_count = process_pdf(
                            pdf_bytes
                        )

                    st.session_state.active_chunks = chunks
                    st.session_state.active_embeddings = embeddings
                    st.session_state.active_paper_name = uploaded_file.name
                    st.session_state.active_pdf_hash = pdf_hash
                    st.session_state.messages = []

                    st.success(
                        f"Processed {page_count} pages into "
                        f"{len(chunks)} chunks."
                    )

                except Exception as error:
                    st.error(f"PDF processing failed: {error}")

        st.divider()
        st.header("Configuration")
        st.write(f"**LLM:** Qwen3 4B")
        st.write(f"**Embeddings:** {MODEL_NAME}")
        st.write("**Inference:** Local Ollama")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("Use saved index", use_container_width=True):
            st.session_state.active_chunks = None
            st.session_state.active_embeddings = None
            st.session_state.active_paper_name = None
            st.session_state.active_pdf_hash = None
            st.session_state.messages = []
            st.rerun()

    # Use the uploaded paper if it has been processed.
    if st.session_state.active_chunks is not None:
        chunks = st.session_state.active_chunks
        embeddings = st.session_state.active_embeddings
        paper_name = st.session_state.active_paper_name

    else:
        try:
            chunks, embeddings = load_saved_index()
            paper_name = "Saved research paper index"
        except FileNotFoundError:
            st.info(
                "Upload a PDF and select Process PDF, or create a saved "
                "index first using `python -m src.semantic_search`."
            )
            st.stop()
        except Exception as error:
            st.error(f"Could not load the saved index: {error}")
            st.stop()

    st.success(f"Active paper: {paper_name} · {len(chunks)} chunks")

    # Load the embedding model once and reuse it.
    try:
        embedding_model = load_embedding_model()
    except Exception as error:
        st.error(f"Could not load the embedding model: {error}")
        st.stop()

    # Display conversation history and evidence.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View retrieved evidence"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**[{source['source_id']}] "
                            f"Page {source['page']}** · "
                            f"Similarity: {source['score']:.3f}"
                        )
                        st.write(source["text"])
                        st.divider()

    question = st.chat_input(
        f"Ask a question about {paper_name}..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and asking Qwen3 4B..."):
                try:
                    result = answer_question(
                        question=question,
                        chunks=chunks,
                        embeddings=embeddings,
                        embedding_model=embedding_model,
                    )

                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander(
                            f"View retrieved evidence ({len(sources)} passages)",
                            expanded=True,
                        ):
                            for source in sources:
                                st.markdown(
                                    f"**[{source['source_id']}] "
                                    f"Page {source['page']}** · "
                                    f"Similarity: {source['score']:.3f}"
                                )
                                st.write(source["text"])
                                st.divider()
                    else:
                        st.info("No evidence passages were retrieved.")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")


if __name__ == "__main__":
    main()
