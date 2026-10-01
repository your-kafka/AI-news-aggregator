"""Database connections and transactions.

Two things live here:

  get_engine()    - the connection POOL. One per process, created lazily.
  session_scope() - one TRANSACTION. Created per unit of work.

The distinction matters. An engine is expensive and long-lived; it holds
open TCP connections and is shared. A session is cheap and short-lived; it
tracks the objects you are changing and the transaction they belong to.
Sharing one session across a whole process is a classic bug - unrelated
work ends up in the same transaction, and one failure rolls back the lot.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from lodestar.core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """The connection pool for this process. Created on first use.

    Lazy on purpose: importing this module must not open a connection, or
    `make test` and CI would need a live database just to collect tests.
    """
    settings = get_settings()
    return create_engine(
        settings.database_url,
        # Check a pooled connection is still alive before handing it out.
        # Without this, a connection idle through a database restart or a
        # network blip comes back as "server closed the connection
        # unexpectedly" on a random query, hours later.
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        # Recycle connections every 30 minutes. Managed Postgres providers
        # often drop idle connections silently.
        pool_recycle=1800,
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,  # keep attributes readable after commit
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    """One transaction: commit on success, roll back on any exception.

        with session_scope() as session:
            ArticleRepository(session).upsert_many(articles)
            RunRepository(session).finish(run_id, stats)

    Both of those commit together or not at all. That is why repositories
    take a session rather than making their own - the CALLER decides where
    the transaction boundary is.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop the cached engine and factory. For tests that re-point the DB."""
    get_engine.cache_clear()
    get_session_factory.cache_clear()
