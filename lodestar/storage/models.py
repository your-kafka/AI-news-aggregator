"""Database tables, declared as Python classes (SQLAlchemy 2.0 style).

One table holds articles from every source, distinguished by a `source`
column. The alternative - a table per source - forces a near-identical
repository method per source and makes "newest across all sources" a
three-query stitch-up instead of one ORDER BY.

Every column's Python type hint generates its SQL type and nullability,
so mypy and Postgres can never disagree about the shape of a row.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Parent of every table. Alembic reads Base.metadata to spot changes."""


class Source(StrEnum):
    """Where an article came from.

    Adding a source here plus a scraper in P2 is the whole cost of a new
    feed. No new table, no new repository method.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    YOUTUBE = "youtube"
    ARXIV = "arxiv"
    HACKERNEWS = "hackernews"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Store enums as VARCHAR with a CHECK constraint rather than a native
# Postgres ENUM type. Adding a value to a native enum needs ALTER TYPE,
# which cannot run inside a transaction in older Postgres and makes
# migrations awkward. A CHECK constraint is just a normal migration.
# values_callable is also required. By default SQLAlchemy stores the enum
# MEMBER NAME ("OPENAI"), not its value ("openai") - so a hand-written
# `WHERE source = 'openai'` would silently match nothing.
def _enum_values(enum_cls: type[StrEnum]) -> list[str]:
    return [member.value for member in enum_cls]


def _in_clause(column: str, enum_cls: type[StrEnum]) -> str:
    """Build "column IN ('a', 'b')" from an enum, as SQL text.

    We write these CHECK constraints by hand rather than using
    Enum(create_constraint=True). That flag builds its constraint against a
    synthetic table at DDL time, so Alembic's autogenerate finds it in the
    database but not in the model metadata - and emits a drop_constraint for
    it in EVERY future migration. Declaring it here, with a name, lets
    Alembic match the two and leave it alone.

    The value list still comes from the enum, so there is one source of truth.
    """
    values = ", ".join(f"'{member.value}'" for member in enum_cls)
    return f"{column} IN ({values})"



_source_enum = Enum(
    Source,
    name="source_enum",
    native_enum=False,
    create_constraint=False,
    values_callable=_enum_values,
    length=32,
)
_run_status_enum = Enum(
    RunStatus,
    name="run_status_enum",
    native_enum=False,
    create_constraint=False,
    values_callable=_enum_values,
    length=16,
)


class Article(Base):
    """One item from any source: a blog post, a paper, or a video."""

    __tablename__ = "articles"

    # UUID rather than an auto-incrementing integer: a worker can generate
    # the id BEFORE inserting, so it can log and reference the article
    # without a database round trip. Costs slightly more storage.
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    source: Mapped[Source] = mapped_column(_source_enum)

    # The feed's own identifier (RSS guid, YouTube video id, arXiv id).
    # Paired with source it is unique - see __table_args__ below.
    external_id: Mapped[str] = mapped_column(String(512))

    title: Mapped[str] = mapped_column(String(1024))
    url: Mapped[str] = mapped_column(String(2048))

    # Mapped[X | None] is what makes a column nullable. Not every feed
    # gives an author, and transcripts have no author at all.
    author: Mapped[str | None] = mapped_column(String(256), default=None)

    # timezone=True gives TIMESTAMPTZ. Always. Naive timestamps force you
    # to re-attach a timezone later and guess which one was meant.
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Short text straight from the feed - usually present immediately.
    summary: Mapped[str | None] = mapped_column(Text, default=None)

    # Full body: article markdown or a video transcript. Filled in later
    # by a separate enrichment step, so it starts NULL. "content IS NULL"
    # is literally the work queue for that step.
    content: Mapped[str | None] = mapped_column(Text, default=None)
    content_fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # How many times we have tried to fetch the body, and why it last failed.
    #
    # Without these, "content IS NULL" retries a 404 or a transcript-disabled
    # video on every single run, forever. The reference project works around
    # this by writing the literal string "__UNAVAILABLE__" into the transcript
    # column, which then pollutes the text that gets embedded.
    content_attempts: Mapped[int] = mapped_column(default=0, server_default="0")
    content_error: Mapped[str | None] = mapped_column(Text, default=None)

    # WHEN to try again, not just how many times.
    #
    # A counter alone is not enough: each batch re-queries the work queue, so
    # a rate-limited article burned all three of its attempts within five
    # seconds and was then treated as permanently dead - even though a 429
    # would have cleared in an hour. This holds the next eligible time, set
    # to an exponentially growing delay on each transient failure.
    content_next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # Anything source-specific that does not deserve its own column:
    # channel_id, arXiv categories, HN score. JSONB is indexable and
    # queryable, unlike a plain JSON string.
    meta: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )

    # server_default=func.now() uses the DATABASE clock, not the app's.
    # Several worker containers have several slightly different clocks;
    # the database has one.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        # Makes ingestion idempotent IN THE DATABASE. Re-running a scraper
        # cannot create duplicates. A Python "SELECT then INSERT" check is
        # a race: two workers can both find nothing and both insert.
        UniqueConstraint("source", "external_id", name="uq_articles_source_external_id"),
        # Feeds are read newest-first and usually filtered by source.
        Index("ix_articles_published_at", published_at.desc()),
        Index("ix_articles_source_published_at", "source", published_at.desc()),
        # Cheap guard against junk rows.
        CheckConstraint("length(title) > 0", name="ck_articles_title_not_empty"),
        CheckConstraint(_in_clause("source", Source), name="ck_articles_source"),
    )

    def __repr__(self) -> str:
        return f"<Article {self.source}:{self.external_id} {self.title[:40]!r}>"


class Run(Base):
    """One execution of the pipeline.

    Its id is the run_id bound in core.logging, so a row here and every
    log line from that execution share one identifier.
    """

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    status: Mapped[RunStatus] = mapped_column(
        _run_status_enum, default=RunStatus.RUNNING
    )

    # "scheduled" or "manual" - lets you tell a cron run from a button press.
    trigger: Mapped[str] = mapped_column(String(32), default="manual")

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # Per-stage counts, e.g. {"fetched": 150, "new": 14, "skipped": 136}.
    # Free-form on purpose: stages get added over the next few phases and
    # none of them should require a migration.
    stats: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)

    __table_args__ = (
        Index("ix_runs_started_at", started_at.desc()),
        CheckConstraint(_in_clause("status", RunStatus), name="ck_runs_status"),
    )

    def __repr__(self) -> str:
        return f"<Run {self.id} {self.status}>"
