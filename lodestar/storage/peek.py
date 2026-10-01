"""Browse the corpus from the terminal: `make peek`, `make show q="..."`.

A convenience for inspecting what ingestion and enrichment actually stored.
Read-only, and deliberately not part of the pipeline.
"""

from __future__ import annotations

import sys
import textwrap

from sqlalchemy import func, select

from lodestar.storage.models import Article
from lodestar.storage.session import session_scope


def _bar(value: int, biggest: int, width: int = 22) -> str:
    filled = 0 if biggest == 0 else round(width * value / biggest)
    return "#" * filled


def overview() -> None:
    with session_scope() as session:
        rows = session.execute(
            select(
                Article.source,
                func.count().label("total"),
                func.count(Article.content).label("with_body"),
                func.sum(func.length(Article.content)).label("chars"),
                func.round(func.avg(func.length(Article.content))).label("avg"),
                func.max(func.length(Article.content)).label("longest"),
            )
            .group_by(Article.source)
            .order_by(func.count().desc())
        ).all()

        total_docs = sum(r.total for r in rows)
        total_chars = sum(int(r.chars or 0) for r in rows)
        biggest = max((r.total for r in rows), default=0)

        print()
        print(f"  CORPUS: {total_docs} documents, {total_chars:,} characters of body text")
        print()
        print(f"  {'source':<12} {'docs':>5} {'bodies':>7} {'avg':>7} {'longest':>8}  ")
        print(f"  {'-' * 12} {'-' * 5} {'-' * 7} {'-' * 7} {'-' * 8}")
        for r in rows:
            print(
                f"  {r.source.value:<12} {r.total:>5} {r.with_body:>7} "
                f"{int(r.avg or 0):>7} {int(r.longest or 0):>8}  {_bar(r.total, biggest)}"
            )

        missing = session.execute(
            select(func.count()).select_from(Article).where(Article.content.is_(None))
        ).scalar_one()
        print()
        print(f"  without a body: {missing}")

        print()
        print("  LONGEST DOCUMENTS (these are why chunking matters)")
        longest = session.execute(
            select(Article)
            .where(Article.content.is_not(None))
            .order_by(func.length(Article.content).desc())
            .limit(5)
        ).scalars()
        for a in longest:
            print(f"    {len(a.content or ''):>7,} chars  {a.source.value:<11} {a.title[:52]}")

        print()
        print("  SHORTEST DOCUMENTS")
        shortest = session.execute(
            select(Article)
            .where(Article.content.is_not(None))
            .order_by(func.length(Article.content).asc())
            .limit(3)
        ).scalars()
        for a in shortest:
            print(f"    {len(a.content or ''):>7,} chars  {a.source.value:<11} {a.title[:52]}")
        print()


def show(query: str) -> None:
    """Print the first document whose title matches `query`."""
    with session_scope() as session:
        article = session.execute(
            select(Article)
            .where(Article.title.ilike(f"%{query}%"), Article.content.is_not(None))
            .order_by(func.length(Article.content).desc())
            .limit(1)
        ).scalar_one_or_none()

        if article is None:
            print(f"  nothing with a body matching {query!r}")
            return

        body = article.content or ""
        print()
        print(f"  {article.title}")
        print(f"  {article.source.value} | {article.published_at:%Y-%m-%d} | {len(body):,} chars")
        print(f"  {article.url}")
        print("  " + "-" * 72)
        for line in body[:3000].splitlines():
            print(textwrap.fill(line, width=76, initial_indent="  ", subsequent_indent="  "))
        if len(body) > 3000:
            print(f"\n  ... {len(body) - 3000:,} more characters")
        print()


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--show":
        show(sys.argv[2])
    else:
        overview()
    return 0


if __name__ == "__main__":
    sys.exit(main())
