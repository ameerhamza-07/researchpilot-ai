
import pytest

from src.structured_compare import compare_papers_structured


def test_requires_at_least_two_papers():
    with pytest.raises(ValueError, match="at least two"):
        compare_papers_structured(
            paper_ids=["paper-one"],
            embedding_model=None,
        )


def test_rejects_duplicate_paper_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        compare_papers_structured(
            paper_ids=["paper-one", "paper-one"],
            embedding_model=None,
        )
