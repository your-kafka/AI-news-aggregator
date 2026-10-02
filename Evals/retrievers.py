"""Retrievers under evaluation, behind one interface.

Anything with a .search() method can be evaluated, which is what lets the
same harness score today's trivial baseline and the full six-stage pipeline
later, with no change to the evaluation code.

PostgresFullTextRetriever is the BASELINE. It is deliberately simple - one
SQL query, no embeddings, no chunking, no re-ranking - and its whole purpose
is to produce the "before" number. A fancy retriever that cannot beat
full-text search is not earning its complexity, and without a baseline there
is no way to know.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlalchemy import text

from lodestar.storage.session import session_scope


@dataclass(frozen=True)
class Hit:
    """One retrieved result."""

    source: str
    external_id: str
    score: float
    text: str = ""

    @property
    def key(self) -> str:
        """Document identity used to match against the golden grades."""
        return f"{self.source}:{self.external_id}"


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


class PostgresFullTextRetriever:
    """Lexical baseline: Postgres full-text search over title + body.

    websearch_to_tsquery parses a plain query the way a search box would.
    ts_rank_cd is cover-density ranking, which rewards query terms appearing
    close together - not true BM25, but a genuine lexical baseline.

    Title is repeated twice to weight it, since a term in a title is a
    stronger signal than the same term buried in 90,000 characters of body.
    """

    name = "postgres-fulltext"

    SQL = text("""
        SELECT source, external_id, title, content,
               ts_rank_cd(
                   to_tsvector('english',
                       coalesce(title,'') || ' ' || coalesce(title,'') || ' ' ||
                       coalesce(content, coalesce(summary,''))),
                   query
               ) AS score
        FROM articles, websearch_to_tsquery('english', :q) AS query
        WHERE to_tsvector('english',
                  coalesce(title,'') || ' ' ||
                  coalesce(content, coalesce(summary,''))) @@ query
        ORDER BY score DESC
        LIMIT :k
    """)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        with session_scope() as session:
            rows = session.execute(self.SQL, {"q": query, "k": k}).all()
        return [
            Hit(
                source=row.source,
                external_id=row.external_id,
                score=float(row.score),
                # A window of body text, standing in for a chunk until real
                # chunking exists. DeepEval judges this as retrieval_context.
                text=f"{row.title}\n\n{(row.content or '')[:1200]}",
            )
            for row in rows
        ]


class TitleOnlyRetriever(PostgresFullTextRetriever):
    """Even weaker baseline: titles only, ignoring the body.

    Included to show how much the body text is actually worth. If full
    retrieval barely beats this, the bodies are not being used well.
    """

    name = "title-only"

    SQL = text("""
        SELECT source, external_id, title, content,
               ts_rank_cd(to_tsvector('english', coalesce(title,'')), query) AS score
        FROM articles, websearch_to_tsquery('english', :q) AS query
        WHERE to_tsvector('english', coalesce(title,'')) @@ query
        ORDER BY score DESC
        LIMIT :k
    """)


class PostgresLexicalOrRetriever(PostgresFullTextRetriever):
    """The FAIR lexical baseline: same index, OR semantics.

    websebsearch_to_tsquery joins terms with AND, so a natural-language
    question only matches documents containing EVERY content word. Asking
    "What percentage of Barclays developers..." therefore returns nothing,
    because "percentage" does not appear in the Barclays article - even though
    "Barclays" alone matches it immediately.

    That makes the AND variant a strawman, and beating a strawman proves
    nothing. This version lexemises the query with to_tsvector, then ORs the
    lexemes, so a document matching some query terms still scores and ranking
    decides the rest. Stop-word removal and stemming come free from
    to_tsvector rather than being hand-rolled.

    This is the number a hybrid retriever has to beat to justify its
    complexity.
    """

    name = "postgres-lexical-or"

    SQL = text("""
        WITH q AS (
            SELECT to_tsquery(
                'english',
                array_to_string(
                    tsvector_to_array(to_tsvector('english', :q)), ' | '
                )
            ) AS query
        )
        SELECT source, external_id, title, content,
               ts_rank_cd(
                   to_tsvector('english',
                       coalesce(title,'') || ' ' || coalesce(title,'') || ' ' ||
                       coalesce(content, coalesce(summary,''))),
                   q.query
               ) AS score
        FROM articles, q
        WHERE to_tsvector('english',
                  coalesce(title,'') || ' ' ||
                  coalesce(content, coalesce(summary,''))) @@ q.query
        ORDER BY score DESC
        LIMIT :k
    """)
