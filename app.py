
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.index_store import load_index
from src.rag import answer_question
from src.semantic_search import MODEL_NAME


st.set_page_config(
    page_title="ResearchPilot AI",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_embedding_model():
    """Load the embedding model once per app session."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource
def load_paper_index():
    """Load the saved chunks and embeddings."""
    return load_index()


def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.title("📚 ResearchPilot AI")
    st.caption(
        "Your local AI research assistant — ask questions "
        "about a research paper and inspect the supporting evidence."
    )

    with st.sidebar:
        st.header("About ResearchPilot")
        st.write("**Language model:** Qwen3 4B")
        st.write("**Retrieval:** Semantic search")
        st.write("**Embeddings:** Sentence Transformers")
        st.write("**Inference:** Local through Ollama")

        st.divider()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "Current version: one pre-indexed research paper."
        )

    initialize_chat()

    try:
        chunks, embeddings = load_paper_index()
        embedding_model = load_embedding_model()
    except FileNotFoundError:
        st.error(
            "No saved paper index was found. Build your index first "
            "by running `python -m src.semantic_search`."
        )
        st.stop()
    except Exception as error:
        st.error(f"Could not initialize ResearchPilot: {error}")
        st.stop()

    st.success(f"Research index loaded: {len(chunks)} text chunks")

    # Display previous messages and their evidence.
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
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

    question = st.chat_input(
        "Ask a question about the research paper..."
    )

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching the paper and generating an answer..."
            ):
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
                        st.info("No supporting passages were retrieved.")

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
                        "Check that Ollama is running and the "
                        "qwen3:4b model is available."
                    )


if __name__ == "__main__":
    main()
