"""Check the golden dataset against the live corpus.

Run this after any edit to the goldens, and in CI. It catches the failure
mode that quietly ruins a retrieval benchmark: a ground-truth document id
that does not exist, which makes a perfectly good retriever look broken
because the "correct" answer is unreachable.

    uv run python Evals/validate_goldens.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select, tuple_

from lodestar.storage.models import Article, Source
from lodestar.storage.session import session_scope

GOLDENS = Path(__file__).parent / "golden" / "retrieval_goldens.jsonl"
VALID_GRADES = {1, 2, 3}


def load() -> list[dict[str, Any]]:
    with GOLDENS.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    goldens = load()
    problems: list[str] = []
    seen_ids: Counter[str] = Counter()

    wanted: set[tuple[str, str]] = set()
    for g in goldens:
        seen_ids[str(g["id"])] += 1
        if not str(g["question"]).strip():
            problems.append(f"{g['id']}: empty question")
        if len(str(g["ideal_answer"]).strip()) < 20:
            problems.append(f"{g['id']}: ideal_answer too short to be useful")
        docs = g["relevant_docs"]
        assert isinstance(docs, list)
        if not docs:
            problems.append(f"{g['id']}: no relevant documents")
        if not any(d["grade"] == 3 for d in docs):
            # Without at least one perfectly-relevant document, nDCG has no
            # achievable ceiling and the question cannot be scored fairly.
            problems.append(f"{g['id']}: no grade-3 document")
        for d in docs:
            if d["grade"] not in VALID_GRADES:
                problems.append(f"{g['id']}: grade {d['grade']} outside 1-3")
            if d["source"] not in {s.value for s in Source}:
                problems.append(f"{g['id']}: unknown source {d['source']!r}")
            wanted.add((str(d["source"]), str(d["external_id"])))

    for gid, count in seen_ids.items():
        if count > 1:
            problems.append(f"duplicate golden id {gid} ({count} times)")

    # One query for every referenced document.
    with session_scope() as session:
        rows = session.execute(
            select(Article.source, Article.external_id, Article.content).where(
                tuple_(Article.source, Article.external_id).in_(
                    [(Source(s), e) for s, e in wanted]
                )
            )
        ).all()

    present = {(r[0].value, r[1]) for r in rows}
    no_body = {(r[0].value, r[1]) for r in rows if not r[2]}
    missing = wanted - present

    for source, ext in sorted(missing):
        problems.append(f"MISSING from corpus: {source}:{ext}")
    for source, ext in sorted(no_body):
        # Retrievable by title, but with no body there is nothing to chunk,
        # so it can never be found by passage-level search.
        problems.append(f"no content (unchunkable): {source}:{ext}")

    judgement_count = sum(len(g["relevant_docs"]) for g in goldens)
    print(f"goldens              : {len(goldens)}")
    print(f"relevance judgements : {judgement_count}")
    print(f"distinct documents   : {len(wanted)}")
    print(f"found in corpus      : {len(present)}")
    print()

    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("all referenced documents exist and have content")
    return 0


if __name__ == "__main__":
    sys.exit(main())
