"""Every database operation on articles. The only code that writes them.

Nothing outside lodestar.storage imports SQLAlchemy. Callers pass in a
Session and get model objects back, which is what lets the retrieval engine
in P4 be tested without a database at all.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from lodestar.core.logging import get_logger
from lodestar.domain.article import ArticleIn
from lodestar.storage.models import Article, Source

log = get_logger(__name__)

#: First retry delay for a transient content failure; doubles each attempt.
RETRY_BASE_DELAY = timedelta(minutes=15)


class ArticleRepository:
    """Reads and writes articles.

    Takes a Session rather than creating one: the caller owns the
    transaction, so an ingest run can write articles and update its Run row
    atomically.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # -- writes ------------------------------------------------------------

    def upsert_many(self, articles: Sequence[ArticleIn]) -> list[uuid.UUID]:
        """Insert articles, skipping any we already have.

        Returns the ids of the rows ACTUALLY inserted, so the caller knows
        how many were new without counting before and after.

        One statement, not a loop. The old pattern -

            existing = session.query(Article).filter_by(...).first()
            if not existing:
                session.add(Article(...))

        - is a race: two workers can both find nothing and both insert. Here
        the UNIQUE (source, external_id) constraint from models.py decides,
        and ON CONFLICT DO NOTHING makes a re-run a no-op. That is what
        makes ingestion safely retryable, which Celery needs in P8.
        """
        if not articles:
            return []

        rows = [{"id": uuid.uuid4(), **a.to_row()} for a in articles]

        statement = (
            pg_insert(Article)
            .values(rows)
            .on_conflict_do_nothing(
                # Name the columns, not the constraint, so this keeps working
                # if the constraint is ever renamed.
                index_elements=["source", "external_id"],
            )
            .returning(Article.id)
        )
        inserted = list(self.session.execute(statement).scalars().all())

        log.info(
            "articles_upserted",
            offered=len(rows),
            inserted=len(inserted),
            skipped=len(rows) - len(inserted),
        )
        return inserted

    def set_content(self, article_id: uuid.UUID, content: str) -> bool:
        """Attach fetched body text or a transcript. Returns False if gone."""
        article = self.session.get(Article, article_id)
        if article is None:
            return False
        article.content = content
        article.content_fetched_at = datetime.now(UTC)
        return True

    # -- reads -------------------------------------------------------------

    def get(self, article_id: uuid.UUID) -> Article | None:
        return self.session.get(Article, article_id)

    def get_by_external_id(self, source: Source, external_id: str) -> Article | None:
        statement = select(Article).where(
            Article.source == source,
            Article.external_id == external_id,
        )
        return self.session.execute(statement).scalar_one_or_none()

    def list_recent(
        self,
        limit: int = 20,
        source: Source | None = None,
        since: datetime | None = None,
    ) -> list[Article]:
        """Newest first, optionally filtered by source or publication date.

        The single-table design is what makes this one query. With a table
        per source it would be one query per source plus a merge sort in
        Python, and LIMIT could not be pushed down to the database.
        """
        statement = select(Article).order_by(Article.published_at.desc()).limit(limit)
        if source is not None:
            statement = statement.where(Article.source == source)
        if since is not None:
            statement = statement.where(Article.published_at >= since)
        return list(self.session.execute(statement).scalars().all())

    def list_needing_content(
        self, limit: int = 50, max_attempts: int = 3
    ) -> list[Article]:
        """Articles whose body has not been fetched yet, and is still worth trying.

        This IS the work queue for enrichment: "content IS NULL" needs no
        extra table and no status column, so the queue cannot disagree with
        reality.

        The attempt limit is what makes it DRAIN. Without it a 404 or a
        transcript-disabled video is retried on every run, forever.
        """
        now = datetime.now(UTC)
        statement = (
            select(Article)
            .where(
                Article.content.is_(None),
                Article.content_attempts < max_attempts,
                # Not due yet -> skip. This is what stops a rate-limited
                # article from burning every attempt in one run.
                or_(
                    Article.content_next_attempt_at.is_(None),
                    Article.content_next_attempt_at <= now,
                ),
            )
            .order_by(Article.published_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(statement).scalars().all())

    def mark_content_failed(
        self,
        article_id: uuid.UUID,
        error: str,
        *,
        permanent: bool = False,
        max_attempts: int = 3,
    ) -> bool:
        """Record why a body could not be fetched.

        A permanent failure jumps straight to the attempt limit rather than
        counting up, so we stop asking immediately instead of after three
        pointless retries.
        """
        article = self.session.get(Article, article_id)
        if article is None:
            return False
        article.content_error = error[:2000]

        if permanent:
            # Jump to the limit so we stop asking immediately rather than
            # after three pointless retries.
            article.content_attempts = max_attempts
            article.content_next_attempt_at = None
            return True

        article.content_attempts += 1
        # Exponential backoff: 15 min, 30 min, 60 min. Long enough that a
        # rate limit has actually had a chance to clear.
        delay = RETRY_BASE_DELAY * (2 ** (article.content_attempts - 1))
        article.content_next_attempt_at = datetime.now(UTC) + delay
        return True

    def count(self, source: Source | None = None) -> int:
        statement = select(func.count()).select_from(Article)
        if source is not None:
            statement = statement.where(Article.source == source)
        return self.session.execute(statement).scalar_one()

    def count_by_source(self) -> dict[Source, int]:
        """How many articles per source. Feeds the Analytics page in P9."""
        statement = select(Article.source, func.count()).group_by(Article.source)
        return {row[0]: row[1] for row in self.session.execute(statement)}
