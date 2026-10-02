"""Evaluate a retriever against the golden dataset.

    uv run python Evals/retrieval_eval.py              # deterministic metrics
    uv run python Evals/retrieval_eval.py --deepeval   # + LLM-judged metrics
    uv run python Evals/retrieval_eval.py --show G004  # inspect one query

TWO LAYERS, on purpose:

  Deterministic IR metrics (nDCG, MRR, Recall, Precision) come from the
  graded relevance judgements in the golden file. They are free, identical
  on every run, and therefore safe to gate CI on.

  DeepEval's metrics (ContextualPrecision, ContextualRecall,
  ContextualRelevancy) use an LLM as a judge. They catch things the grades
  cannot - whether the retrieved passages actually CONTAIN the answer, not
  merely whether the right document was returned - but they cost an API call
  per test and vary between runs, so they inform rather than gate.

Chunks are collapsed to documents before scoring: the golden file grades
documents, and a retriever returning three chunks of the same article has
found one document, not three.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from Evals.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from Evals.retrievers import (
    Hit,
    PostgresFullTextRetriever,
    PostgresLexicalOrRetriever,
    Retriever,
    TitleOnlyRetriever,
)

GOLDENS = Path(__file__).parent / "golden" / "retrieval_goldens.jsonl"

K = 10          # the ranking depth users actually see
RECALL_K = 50   # candidate-generation depth; a re-ranker can only reorder this


def load_goldens() -> list[dict[str, Any]]:
    with GOLDENS.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def grades_of(golden: dict[str, Any]) -> dict[str, int]:
    """{"source:external_id": grade} for one golden."""
    return {
        f"{d['source']}:{d['external_id']}": int(d["grade"])
        for d in golden["relevant_docs"]
    }


def to_document_ranking(hits: list[Hit]) -> list[str]:
    """Collapse chunk-level hits to a document ranking, keeping best rank.

    Without this, a retriever that returns five chunks of one article looks
    like five results. The golden file grades documents, so the comparison
    has to be document-level.
    """
    seen: list[str] = []
    for hit in hits:
        if hit.key not in seen:
            seen.append(hit.key)
    return seen


def evaluate(retriever: Retriever, goldens: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one retriever over every golden."""
    rows: list[dict[str, Any]] = []

    for golden in goldens:
        grades = grades_of(golden)
        hits = retriever.search(golden["question"], k=RECALL_K)
        ranking = to_document_ranking(hits)

        rows.append({
            "id": golden["id"],
            "difficulty": golden["difficulty"],
            "query_type": golden["query_type"],
            f"ndcg@{K}": ndcg_at_k(ranking, grades, K),
            f"mrr@{K}": mrr_at_k(ranking, grades, K),
            f"precision@{K}": precision_at_k(ranking, grades, K),
            f"recall@{RECALL_K}": recall_at_k(ranking, grades, RECALL_K),
            f"hit@{K}": hit_rate_at_k(ranking, grades, K),
        })

    metric_names = [
        f"ndcg@{K}", f"mrr@{K}", f"precision@{K}",
        f"recall@{RECALL_K}", f"hit@{K}",
    ]
    means = {
        name: sum(r[name] for r in rows) / len(rows) if rows else 0.0
        for name in metric_names
    }
    return {"retriever": retriever.name, "means": means, "rows": rows, "metrics": metric_names}


def print_report(results: list[dict[str, Any]]) -> None:
    metrics = results[0]["metrics"]
    width = max(len(r["retriever"]) for r in results) + 2

    print()
    print("  " + "RETRIEVAL QUALITY".ljust(width) + "  ".join(f"{m:>14}" for m in metrics))
    print("  " + "-" * (width + 16 * len(metrics)))
    for result in results:
        cells = "  ".join(f"{result['means'][m]:>14.4f}" for m in metrics)
        print(f"  {result['retriever']:<{width}}{cells}")
    print()

    # Per-difficulty breakdown for the last (presumably best) retriever.
    best = results[-1]
    print(f"  {best['retriever']} broken down by difficulty")
    for level in ("easy", "medium", "hard"):
        subset = [r for r in best["rows"] if r["difficulty"] == level]
        if not subset:
            continue
        ndcg = sum(r[f"ndcg@{K}"] for r in subset) / len(subset)
        hit = sum(r[f"hit@{K}"] for r in subset) / len(subset)
        print(f"    {level:<8} n={len(subset):<3} ndcg@{K}={ndcg:.4f}  hit@{K}={hit:.4f}")
    print()

    print("  weakest queries (where to look next)")
    worst = sorted(best["rows"], key=lambda r: r[f"ndcg@{K}"])[:6]
    for row in worst:
        print(f"    {row['id']}  ndcg={row[f'ndcg@{K}']:.3f}  "
              f"[{row['query_type']}/{row['difficulty']}]")
    print()


def show_one(retriever: Retriever, goldens: list[dict[str, Any]], golden_id: str) -> int:
    """Print what the retriever returned for one golden, next to the truth."""
    match = next((g for g in goldens if g["id"] == golden_id), None)
    if match is None:
        print(f"no golden with id {golden_id!r}")
        return 1

    grades = grades_of(match)
    hits = retriever.search(match["question"], k=K)

    print(f"\n  {match['id']}  [{match['query_type']}/{match['difficulty']}]")
    print(f"  Q: {match['question']}")
    print(f"  ideal: {match['ideal_answer'][:160]}...")
    print("\n  expected documents:")
    for key, grade in sorted(grades.items(), key=lambda kv: -kv[1]):
        print(f"    grade {grade}  {key[:88]}")
    print(f"\n  retrieved by {retriever.name}:")
    for i, hit in enumerate(hits, 1):
        mark = f"HIT grade {grades[hit.key]}" if hit.key in grades else "   -"
        print(f"    {i:>2}. {mark:<12} {hit.score:.4f}  {hit.key[:72]}")
    print()
    return 0


def run_deepeval(retriever: Retriever, goldens: list[dict[str, Any]], limit: int) -> int:
    """LLM-judged retrieval metrics via DeepEval.

    Needs an LLM judge configured (OPENAI_API_KEY, or a custom model). These
    metrics ask questions the grades cannot: does the retrieved text actually
    contain the answer, and are the useful passages ranked above the useless
    ones?
    """
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import (
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        print("  deepeval is not installed. Run:  uv sync --group evals")
        return 1

    cases = []
    for golden in goldens[:limit]:
        hits = retriever.search(golden["question"], k=K)
        cases.append(LLMTestCase(
            input=golden["question"],
            # NOTE: there is no generator yet, so the ideal answer stands in
            # here. The three metrics below judge expected_output against
            # retrieval_context, so actual_output does not drive the verdict -
            # but this must be replaced with the real generated answer once a
            # generator exists, or end-to-end metrics would be meaningless.
            actual_output=golden["ideal_answer"],
            expected_output=golden["ideal_answer"],
            retrieval_context=[hit.text for hit in hits],
        ))

    print(f"\n  running DeepEval on {len(cases)} cases (LLM judge; this costs calls)\n")
    deepeval_evaluate(
        test_cases=cases,
        metrics=[
            ContextualRecallMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
            ContextualRelevancyMetric(threshold=0.5),
        ],
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepeval", action="store_true", help="also run LLM-judged metrics")
    parser.add_argument("--deepeval-limit", type=int, default=10, help="cases to judge")
    parser.add_argument("--show", metavar="GOLDEN_ID", help="inspect one query")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args()

    goldens = load_goldens()
    # Ordered weakest to strongest, so the report reads as a progression
    # and the last row is the current best.
    retrievers: list[Retriever] = [
        TitleOnlyRetriever(),
        PostgresFullTextRetriever(),
        PostgresLexicalOrRetriever(),
    ]

    if args.show:
        return show_one(retrievers[-1], goldens, args.show)

    results = [evaluate(r, goldens) for r in retrievers]

    if args.json:
        print(json.dumps([{"retriever": r["retriever"], "means": r["means"]} for r in results], indent=2))
    else:
        print(f"\n  {len(goldens)} goldens, "
              f"{sum(len(g['relevant_docs']) for g in goldens)} relevance judgements")
        print_report(results)

    if args.deepeval:
        return run_deepeval(retrievers[-1], goldens, args.deepeval_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
