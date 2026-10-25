
import pytest

from src.cross_paper import answer_across_papers


def test_blank_question_is_rejected():
    with pytest.raises(ValueError, match="Question must not be empty"):
        answer_across_papers(
            question="   ",
            paper_ids=["paper1"],
            embedding_model=None,
        )


def test_empty_paper_selection_is_rejected():
    with pytest.raises(ValueError, match="Select at least one paper"):
        answer_across_papers(
            question="Compare the methods",
            paper_ids=[],
            embedding_model=None,
        )


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k_per_paper"):
        answer_across_papers(
            question="Compare the methods",
            paper_ids=["paper1"],
            embedding_model=None,
            top_k_per_paper=0,
        )
