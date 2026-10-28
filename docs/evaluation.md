# ResearchPilot AI — Evaluation Report

## 1. Objective

Evaluate the initial ResearchPilot AI question-answering pipeline on factual, methodological, unanswerable, and comparative questions about research papers.

## 2. Evaluation Dataset

The baseline dataset contains 11 questions:

- Factual questions
- Methodological questions
- Unanswerable questions
- Comparative questions

The dataset is stored in `evaluation/evaluation_dataset.csv`.

## 3. Evaluation Method

The evaluation runner loads the questions, retrieves relevant paper content, generates answers using the local language model, and records execution results.

Run the evaluation from the project root:

```bash
python evaluation/run_evaluation.py
```

Check the script's configured output paths if your local command or output filenames differ.

## 4. Initial Baseline Findings

The initial run completed all 11 questions without execution failures.

The average recorded latency was approximately 207.9 seconds per question.

Qualitative observations:

- Several factual and methodological questions received broadly correct answers.
- A question about positional encodings did not retrieve the expected evidence in the initial evaluation.
- A cross-paper comparison question did not adequately compare both papers.
- An unanswerable question produced a response that required further verification.

## 5. Limitations

- The dataset is small and is not sufficient to establish general answer quality.
- Successful execution does not necessarily mean an answer is correct.
- Retrieval relevance affects the quality of generated answers.
- Comparative questions need evidence from each selected paper.
- Local inference latency is high on the current hardware.

## 6. Future Work

- Improve retrieval and evidence selection.
- Evaluate whether answers are supported by retrieved passages.
- Improve handling of unanswerable questions.
- Ensure comparative answers use evidence from both papers.
- Expand the dataset and measure answer quality consistently.
- Reduce inference latency without sacrificing answer quality.

## 7. Reproducibility

The dataset, evaluation runner, results, and summary are maintained in the `evaluation/` directory.

The recorded baseline reflects the current local setup and should not be interpreted as a benchmark against other systems.