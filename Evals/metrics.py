"""Information-retrieval metrics. Pure functions, no I/O, no LLM.

These are the backbone of the evaluation: deterministic, free, and identical
on every run, so they can gate CI. DeepEval's LLM-judged metrics sit on top
and add a different kind of signal - but an LLM judge is non-deterministic
and costs a call per test, so it cannot be the thing a build depends on.

Every function takes:
    ranked  - document keys in the order the retriever returned them
    grades  - {document key: relevance grade}, where a key absent from the
              dict is irrelevant (grade 0)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain.

    Each result's gain is divided by log2(rank + 1), so a relevant document
    at rank 1 is worth more than the same document at rank 10. That is the
    whole point: retrieval quality is about ORDER, not just membership.
    """
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """Normalised DCG - the headline retrieval metric.

    Divides the DCG of what we returned by the DCG of the best possible
    ordering, so 1.0 means "perfect ranking" and the number is comparable
    across queries that have different numbers of relevant documents.
    """
    actual = dcg([grades.get(key, 0) for key in ranked[:k]])
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant result.

    Answers "how far down did the user have to look?" - 1.0 if the first
    result was relevant, 0.5 if the second was, and so on. Harsher than nDCG
    about the top position, which is what matters for a digest that only
    shows ten items.
    """
    for i, key in enumerate(ranked[:k]):
        if grades.get(key, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 50) -> float:
    """Fraction of relevant documents that made it into the top k.

    This is the metric for stage 1 of the pipeline. A re-ranker can only
    reorder what it is given, so anything recall misses is lost for good -
    which is why candidate generation is measured at a much larger k than
    the final ranking.
    """
    relevant = {key for key, grade in grades.items() if grade > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """Fraction of the top k that are relevant."""
    if not ranked[:k]:
        return 0.0
    return sum(1 for key in ranked[:k] if grades.get(key, 0) > 0) / len(ranked[:k])


def hit_rate_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """1.0 if anything relevant appeared in the top k, else 0.0.

    Crude, but the easiest number to explain to a non-specialist: "how often
    did it find something useful at all?"
    """
    return 1.0 if any(grades.get(key, 0) > 0 for key in ranked[:k]) else 0.0
