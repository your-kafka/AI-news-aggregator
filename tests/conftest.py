"""Shared pytest fixtures.

The database fixtures here follow one rule: every test runs inside a
transaction that is ROLLED BACK afterwards. No cleanup code, no test
leaking rows into the next one, and it is much faster than recreating
tables per test.

If no database is reachable, the database tests skip with a clear message
rather than failing - so `make test` still works for someone who has not
run `make up`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from lodestar.storage.models import Base
from lodestar.storage.session import get_engine


@pytest.fixture(scope="session")
def engine() -> Engine:
    """One engine for the whole test session, or skip if there is no database."""
    eng = get_engine()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no database reachable (try `make up`): {exc.__class__.__name__}")

    # checkfirst is the default, so this is a no-op when migrations have
    # already run. It means a fresh database still works for tests.
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A Session whose work is always rolled back.

    The outer transaction belongs to the connection. The Session joins it
    using savepoints, so repository code can call commit() normally - that
    commit releases a savepoint rather than writing anything permanently.
    Rolling the outer transaction back at the end discards everything.
    """
    connection = engine.connect()
    transaction = connection.begin()
    db_session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield db_session
    finally:
        db_session.close()
        transaction.rollback()
        connection.close()
