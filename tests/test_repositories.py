"""Tests for the repository layer, against a real PostgreSQL database.

Real Postgres, not SQLite or a mock, because the behaviour under test IS
Postgres behaviour: ON CONFLICT DO NOTHING, the composite unique index,
TIMESTAMPTZ handling and JSONB defaults. A mock would happily confirm code
that cannot work.

Every test runs in a transaction that is rolled back - see conftest.py.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from lodestar.domain.article import ArticleIn
from lodestar.storage.models import RunStatus, Source
from lodestar.storage.repositories import ArticleRepository, RunRepository

# Every test in this file needs a database.
pytestmark = pytest.mark.integration

NOW = datetime.now(UTC)


def article(
    external_id: str = "post-1",
    source: Source = Source.OPENAI,
    *,
    title: str = "A real title",
    published_at: datetime | None = None,
    **extra: object,
) -> ArticleIn:
    return ArticleIn(
        source=source,
        external_id=external_id,
        title=title,
        url=f"https://example.com/{external_id}",
        published_at=published_at or NOW,
        **extra,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
#  upsert_many - the method that makes ingestion safely retryable
# ---------------------------------------------------------------------------


def test_upsert_many_inserts_and_returns_new_ids(session: Session) -> None:
    repo = ArticleRepository(session)

    inserted = repo.upsert_many([article("a"), article("b"), article("c")])

    assert len(inserted) == 3
    assert all(isinstance(i, uuid.UUID) for i in inserted)
    assert repo.count() == 3


def test_upsert_many_is_idempotent(session: Session) -> None:
    """Re-running a scraper must not duplicate anything.

    This is the behaviour Celery depends on in P8: a retried task replays
    the same work and changes nothing.
    """
    repo = ArticleRepository(session)
    batch = [article("a"), article("b")]

    first = repo.upsert_many(batch)
    second = repo.upsert_many(batch)

    assert len(first) == 2
    assert second == []  # nothing new
    assert repo.count() == 2


def test_upsert_many_reports_only_the_genuinely_new(session: Session) -> None:
    repo = ArticleRepository(session)
    repo.upsert_many([article("a"), article("b")])

    inserted = repo.upsert_many([article("b"), article("c"), article("d")])

    assert len(inserted) == 2  # c and d; b already existed
    assert repo.count() == 4


def test_same_external_id_from_different_sources_coexists(session: Session) -> None:
    """Uniqueness is (source, external_id), not external_id alone.

    Feeds pick ids independently, so two sources can easily both use "1".
    A single-column unique index would silently drop one of them.
    """
    repo = ArticleRepository(session)

    inserted = repo.upsert_many([
        article("shared-id", source=Source.OPENAI),
        article("shared-id", source=Source.ANTHROPIC),
        article("shared-id", source=Source.ARXIV),
    ])

    assert len(inserted) == 3
    assert repo.count() == 3


def test_upsert_many_with_empty_list_does_nothing(session: Session) -> None:
    """Guard against building an INSERT with no VALUES, which is a SQL error."""
    repo = ArticleRepository(session)

    assert repo.upsert_many([]) == []
    assert repo.count() == 0


def test_stored_values_round_trip(session: Session) -> None:
    repo = ArticleRepository(session)
    repo.upsert_many([
        article("a", author="Someone", summary="Short", meta={"score": 42}),
    ])

    stored = repo.get_by_external_id(Source.OPENAI, "a")

    assert stored is not None
    assert stored.author == "Someone"
    assert stored.summary == "Short"
    assert stored.meta == {"score": 42}
    assert stored.source is Source.OPENAI
    assert stored.published_at.tzinfo is not None
    assert stored.created_at is not None  # set by the database clock
    assert stored.content is None


# ---------------------------------------------------------------------------
#  Reads
# ---------------------------------------------------------------------------


def test_get_by_external_id_returns_none_when_absent(session: Session) -> None:
    assert ArticleRepository(session).get_by_external_id(Source.OPENAI, "nope") is None


def test_list_recent_is_newest_first_and_respects_limit(session: Session) -> None:
    repo = ArticleRepository(session)
    repo.upsert_many([
        article("old", published_at=NOW - timedelta(days=3)),
        article("newest", published_at=NOW),
        article("middle", published_at=NOW - timedelta(days=1)),
    ])

    got = repo.list_recent(limit=2)

    assert [a.external_id for a in got] == ["newest", "middle"]


def test_list_recent_can_filter_by_source(session: Session) -> None:
    repo = ArticleRepository(session)
    repo.upsert_many([
        article("o1", source=Source.OPENAI),
        article("a1", source=Source.ANTHROPIC),
        article("a2", source=Source.ANTHROPIC),
    ])

    got = repo.list_recent(source=Source.ANTHROPIC)

    assert {a.external_id for a in got} == {"a1", "a2"}


def test_list_recent_can_filter_by_date(session: Session) -> None:
    repo = ArticleRepository(session)
    repo.upsert_many([
        article("ancient", published_at=NOW - timedelta(days=30)),
        article("fresh", published_at=NOW),
    ])

    got = repo.list_recent(since=NOW - timedelta(days=1))

    assert [a.external_id for a in got] == ["fresh"]


def test_count_by_source_groups_correctly(session: Session) -> None:
    repo = ArticleRepository(session)
    repo.upsert_many([
        article("o1", source=Source.OPENAI),
        article("a1", source=Source.ANTHROPIC),
        article("a2", source=Source.ANTHROPIC),
    ])

    assert repo.count_by_source() == {Source.OPENAI: 1, Source.ANTHROPIC: 2}


# ---------------------------------------------------------------------------
#  The enrichment work queue
# ---------------------------------------------------------------------------


def test_list_needing_content_finds_only_unfetched(session: Session) -> None:
    """"content IS NULL" is the work queue for P3. No status column needed,
    so the queue cannot disagree with reality."""
    repo = ArticleRepository(session)
    repo.upsert_many([article("has-body", content="Full text"), article("no-body")])

    pending = repo.list_needing_content()

    assert [a.external_id for a in pending] == ["no-body"]


def test_set_content_marks_the_article_fetched(session: Session) -> None:
    repo = ArticleRepository(session)
    repo.upsert_many([article("a")])
    stored = repo.get_by_external_id(Source.OPENAI, "a")
    assert stored is not None

    assert repo.set_content(stored.id, "The full body text") is True
    session.flush()

    assert stored.content == "The full body text"
    assert stored.content_fetched_at is not None
    assert repo.list_needing_content() == []  # left the queue


def test_set_content_on_a_missing_article_returns_false(session: Session) -> None:
    assert ArticleRepository(session).set_content(uuid.uuid4(), "x") is False


# ---------------------------------------------------------------------------
#  Runs
# ---------------------------------------------------------------------------


def test_start_records_a_running_run_immediately(session: Session) -> None:
    """Written up front, so a crashed run still leaves evidence - it stays
    RUNNING, which is visible. A row written only on success makes a crash
    indistinguishable from never having started."""
    run = RunRepository(session).start(trigger="scheduled")

    assert run.status is RunStatus.RUNNING
    assert run.trigger == "scheduled"
    assert run.started_at is not None
    assert run.finished_at is None


def test_finish_marks_success_and_stores_stats(session: Session) -> None:
    repo = RunRepository(session)
    run = repo.start()

    assert repo.finish(run.id, {"fetched": 150, "new": 14}) is True
    session.flush()

    assert run.status is RunStatus.SUCCEEDED
    assert run.finished_at is not None
    assert run.stats == {"fetched": 150, "new": 14}
    assert run.error is None


def test_fail_records_the_error(session: Session) -> None:
    repo = RunRepository(session)
    run = repo.start()

    assert repo.fail(run.id, "ConnectionError: feed unreachable") is True
    session.flush()

    assert run.status is RunStatus.FAILED
    assert run.error is not None
    assert "feed unreachable" in run.error


def test_fail_truncates_an_enormous_traceback(session: Session) -> None:
    repo = RunRepository(session)
    run = repo.start()

    repo.fail(run.id, "x" * 20_000)
    session.flush()

    assert run.error is not None
    assert len(run.error) == 8000


def test_finish_on_a_missing_run_returns_false(session: Session) -> None:
    assert RunRepository(session).finish(uuid.uuid4(), {}) is False


def test_last_successful_ignores_running_and_failed(session: Session) -> None:
    """Ingestion uses this as its "fetch since" point, so it must never
    return a run that did not actually complete."""
    repo = RunRepository(session)
    good = repo.start()
    repo.finish(good.id, {})
    bad = repo.start()
    repo.fail(bad.id, "boom")
    repo.start()  # still running
    session.flush()

    last = repo.last_successful()

    assert last is not None
    assert last.id == good.id
