"""Run every registered source and store what they return.

Compare with the reference project's runner, which names all three scrapers
explicitly and calls each with a different method:

    videos   = youtube_scraper.get_latest_videos(channel_id, hours=hours)
    articles = openai_scraper.get_articles(hours=hours)
    anth     = anthropic_scraper.get_articles(hours=hours)

Adding a fourth source means editing that function. Here the runner asks the
registry for whatever is registered and calls one method, so adding arXiv was
adding a file.

Two behaviours worth knowing about:

  PARTIAL SUCCESS. One unreachable feed must not lose the other four, so
  per-source failures are caught, recorded, and the run continues. Only an
  unexpected error fails the whole run.

  INCREMENTAL. `since` defaults to when the last successful run started, so
  a normal run fetches only what is new. The very first run looks back a
  fixed window instead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lodestar.core.logging import bind_context, clear_context, get_logger
from lodestar.ingestion import sources as _sources  # noqa: F401  (registers them)
from lodestar.ingestion.base import SourceFetchError
from lodestar.ingestion.registry import build_all
from lodestar.storage.repositories import ArticleRepository, RunRepository
from lodestar.storage.session import session_scope

log = get_logger(__name__)

# How far back the FIRST ever run looks. A tuning constant, not an
# environment setting, so it lives here rather than in .env.
BOOTSTRAP_WINDOW = timedelta(days=7)

# Overlap re-fetched on every run. Feeds sometimes publish slightly out of
# order, and ON CONFLICT DO NOTHING makes re-reading an item free - so a
# small overlap costs nothing and avoids missing late arrivals.
OVERLAP = timedelta(hours=2)


def ingest(
    since: datetime | None = None,
    trigger: str = "manual",
) -> dict[str, Any]:
    """Fetch from every source and upsert the results. Returns the run stats."""
    with session_scope() as session:
        articles_repo = ArticleRepository(session)
        runs_repo = RunRepository(session)

        run = runs_repo.start(trigger=trigger)
        # Every log line from here on carries this run_id, across every
        # module, without being passed as an argument.
        bind_context(run_id=str(run.id))

        window_start = since or _default_since(runs_repo)
        log.info("ingest_started", trigger=trigger, since=window_start.isoformat())

        per_source: dict[str, Any] = {}
        fetched_total = 0
        new_total = 0
        failures = 0

        try:
            for source in build_all():
                key = source.name.value
                try:
                    articles = source.fetch(window_start)
                except SourceFetchError as exc:
                    # Expected: a feed is down or returned junk. Record and
                    # keep going - four good sources beat zero.
                    failures += 1
                    per_source[key] = {"error": str(exc)}
                    log.warning("source_failed", source=key, error=str(exc))
                    continue
                finally:
                    source.close()

                inserted = articles_repo.upsert_many(articles)
                fetched_total += len(articles)
                new_total += len(inserted)
                per_source[key] = {
                    "fetched": len(articles),
                    "new": len(inserted),
                }

            stats = {
                "since": window_start.isoformat(),
                "fetched": fetched_total,
                "new": new_total,
                "failed_sources": failures,
                "sources": per_source,
            }

            # A run where every source failed is a failed run, not a quiet
            # success that silently ingests nothing for a week.
            if failures and failures == len(per_source):
                runs_repo.fail(run.id, "every source failed", stats)
                log.error("ingest_failed", **stats)
            else:
                runs_repo.finish(run.id, stats)
                log.info("ingest_complete", **stats)

            return {"run_id": str(run.id), **stats}

        except Exception as exc:
            # Unexpected: a bug, not a flaky feed. Record it on the run so
            # the row does not sit in RUNNING forever, then re-raise.
            runs_repo.fail(run.id, f"{type(exc).__name__}: {exc}")
            log.error("ingest_crashed", error=str(exc), exc_info=True)
            raise
        finally:
            clear_context()


def _default_since(runs_repo: RunRepository) -> datetime:
    """Start from the last successful run, minus a small overlap."""
    last = runs_repo.last_successful()
    if last is None:
        return datetime.now(UTC) - BOOTSTRAP_WINDOW
    return last.started_at - OVERLAP
