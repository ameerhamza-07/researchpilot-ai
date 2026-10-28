# ResearchPilot AI

An AI-powered research paper assistant that helps users explore, question, and compare research papers using local Large Language Models (LLMs), semantic search, and Retrieval-Augmented Generation (RAG).

## Features

- **PDF ingestion** — Extract text from research papers.
- **Text chunking** — Split paper content into smaller passages for retrieval.
- **Semantic search** — Retrieve relevant passages using Sentence Transformers embeddings.
- **RAG question answering** — Generate answers grounded in retrieved paper content.
- **Paper library** — Load and work with stored research papers.
- **Multi-paper comparison** — Explore similarities and differences between papers.
- **Structured comparison** — Compare papers using a dedicated comparison mode.
- **Evaluation pipeline** — Run a question-based evaluation dataset and save results and summary metrics.

## Tech Stack

- **Language:** Python
- **PDF processing:** PyMuPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector search:** FAISS
- **Local LLM:** Ollama with Qwen3 4B
- **User interface:** Streamlit
- **Testing:** pytest

## How It Works

1. Load a research paper PDF.
2. Extract its text and split it into chunks.
3. Convert text chunks into embeddings.
4. Retrieve relevant passages for a user's question.
5. Provide the retrieved context to the local LLM.
6. Display the generated answer and available source information.

## Application Modes

- **Ask one paper:** Ask questions about an individual research paper.
- **Compare papers:** Ask questions involving multiple research papers.
- **Structured comparison:** Compare papers in a structured format.

## Evaluation

The project includes a question-based evaluation dataset and an evaluation runner.

The current baseline contains **11 evaluation questions** covering factual, methodological, unanswerable, and comparative questions.

The evaluation pipeline records model answers, execution status, latency, and summary information.

Initial evaluation identified areas for improvement, including retrieval of relevant passages, cross-paper comparisons, and handling questions that cannot be answered from the available sources.

See the `evaluation/` directory for the dataset, runner, results, and summary metrics.

## Prerequisites

- Python 3.11
- Git
- Ollama
- A compatible local LLM pulled through Ollama

## Running Locally

Clone the repository:

```bash
git clone https://github.com/ameerhamza-07/researchpilot-ai.git
cd researchpilot-ai
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Ensure Ollama is installed and the required model is available. For the current setup, the model is `qwen3:4b`.

```bash
ollama pull qwen3:4b
```

Start the application:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## Running Tests

Run the test suite from the project root:

```bash
python -m pytest tests/ -v
```

## Project Structure

```text
researchpilot-ai/
├── app.py
├── src/
│   ├── read_pdf.py
│   ├── chunk_text.py
│   ├── semantic_search.py
│   ├── index_store.py
│   ├── rag.py
│   ├── paper_library.py
│   ├── cross_paper.py
│   └── structured_compare.py
├── tests/
└── evaluation/
    ├── evaluation_dataset.csv
    ├── run_evaluation.py
    ├── evaluation_results.csv
    └── evaluation_summary.json
```

## Current Limitations

- Answer quality depends on retrieval relevance and the local LLM.
- Some comparative questions may not retrieve sufficient evidence from both papers.
- Questions that cannot be answered from the source documents require careful handling.
- Local inference can be slow on hardware without a dedicated GPU.

## Future Improvements

- Improve retrieval quality and ensure evidence is retrieved from each paper during comparisons.
- Strengthen handling of unanswerable questions.
- Improve response latency.
- Expand the evaluation dataset and track answer quality over time.
- Add screenshots and a short project demonstration.

## Project Status

The core application, paper library, question-answering pipeline, comparison modes, automated tests, and initial evaluation pipeline are implemented. Further work is focused on retrieval reliability, answer quality, performance, and portfolio presentation.
