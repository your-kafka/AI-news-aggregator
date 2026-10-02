"""Tests for the IR metrics. No database, no LLM, no corpus.

These functions decide whether every later retrieval change counts as an
improvement, so a bug here would silently mislead every experiment that
follows. They are pure functions over small lists, which makes them cheap to
test exhaustively - and worth testing against hand-computed values rather
than against themselves.
"""

from __future__ import annotations

import math

import pytest

from Evals.metrics import (
    dcg,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# Three relevant documents with different grades, plus irrelevant noise.
GRADES = {"a": 3, "b": 2, "c": 1}


def test_dcg_discounts_by_log_of_rank() -> None:
    """Hand-computed: 3/log2(2) + 2/log2(3) = 3.0 + 1.2618..."""
    assert dcg([3, 2]) == pytest.approx(3.0 + 2 / math.log2(3))


def test_perfect_ranking_scores_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], GRADES, 10) == pytest.approx(1.0)


def test_reversed_ranking_scores_below_one() -> None:
    """Same documents, worse order. nDCG must notice - that is its whole job,
    and the reason precision@k alone is not enough."""
    assert ndcg_at_k(["c", "b", "a"], GRADES, 10) < 1.0


def test_relevant_documents_beyond_k_do_not_count() -> None:
    """A result at rank 11 is invisible to a user looking at ten."""
    ranked = ["x", "y", "z", "a"]
    assert ndcg_at_k(ranked, GRADES, 3) == 0.0
    assert ndcg_at_k(ranked, GRADES, 4) > 0.0


def test_no_relevant_documents_scores_zero_not_nan() -> None:
    """Guard against dividing by an ideal DCG of zero."""
    assert ndcg_at_k(["x", "y"], {}, 10) == 0.0


def test_empty_ranking_scores_zero() -> None:
    assert ndcg_at_k([], GRADES, 10) == 0.0
    assert mrr_at_k([], GRADES, 10) == 0.0
    assert precision_at_k([], GRADES, 10) == 0.0
    assert hit_rate_at_k([], GRADES, 10) == 0.0


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        (["a", "x", "y"], 1.0),        # first result relevant
        (["x", "a", "y"], 0.5),        # second
        (["x", "y", "a"], 1 / 3),      # third
        (["x", "y", "z"], 0.0),        # none
    ],
)
def test_mrr_is_one_over_the_first_relevant_rank(
    ranked: list[str], expected: float
) -> None:
    assert mrr_at_k(ranked, GRADES, 10) == pytest.approx(expected)


def test_mrr_ignores_grade_only_position(ranked: None = None) -> None:
    """A grade-1 document at rank 1 beats a grade-3 at rank 2, because MRR
    measures how far the user had to look, not how good the find was."""
    assert mrr_at_k(["c", "a"], GRADES, 10) == 1.0


@pytest.mark.parametrize(
    ("ranked", "k", "expected"),
    [
        (["a", "b", "c"], 50, 1.0),       # all three found
        (["a", "b"], 50, 2 / 3),
        (["a"], 50, 1 / 3),
        (["x"], 50, 0.0),
        (["x"] * 49 + ["a"], 50, 1 / 3),  # scraping in at rank 50 still counts
        (["x"] * 50 + ["a"], 50, 0.0),    # rank 51 does not
    ],
)
def test_recall_counts_relevant_documents_inside_k(
    ranked: list[str], k: int, expected: float
) -> None:
    assert recall_at_k(ranked, GRADES, k) == pytest.approx(expected)


def test_recall_with_no_relevant_documents_is_zero() -> None:
    assert recall_at_k(["a"], {}, 50) == 0.0


def test_precision_is_the_relevant_fraction_of_the_top_k() -> None:
    assert precision_at_k(["a", "x", "b", "y"], GRADES, 4) == pytest.approx(0.5)


def test_precision_divides_by_results_returned_not_by_k() -> None:
    """Returning 2 results of which 1 is relevant is precision 0.5, not 0.1.
    Dividing by k would punish a retriever for being asked for more results
    than the corpus contains."""
    assert precision_at_k(["a", "x"], GRADES, 10) == pytest.approx(0.5)


def test_hit_rate_is_binary() -> None:
    assert hit_rate_at_k(["x", "y", "a"], GRADES, 10) == 1.0
    assert hit_rate_at_k(["x", "y", "z"], GRADES, 10) == 0.0


def test_recall_and_precision_can_diverge_sharply() -> None:
    """The exact situation measured on the real corpus: the lexical baseline
    reached recall@50 of 0.93 while precision@10 fell to 0.09. Finding
    everything and ordering it badly are different failures, and the metrics
    have to be able to say so."""
    ranked = ["x"] * 9 + ["a", "b", "c"]

    assert recall_at_k(ranked, GRADES, 50) == pytest.approx(1.0)
    assert precision_at_k(ranked, GRADES, 10) == pytest.approx(0.1)
