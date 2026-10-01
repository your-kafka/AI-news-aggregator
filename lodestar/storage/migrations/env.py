"""Alembic migration environment.

Alembic runs this file before every migration command. Its job is to tell
Alembic two things: WHERE the database is, and WHAT the schema should look
like.

The URL comes from lodestar.core.config, not from alembic.ini. alembic.ini
is committed to git, so a URL containing the password must never live there.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from lodestar.core.config import get_settings
from lodestar.storage.models import Base

config = context.config

# Set up Python logging using the [logger_*] sections of alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# WHERE: injected from our validated settings, so there is one source of
# truth for the connection string and the password stays in .env.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

# WHAT: every table registered on Base. This is what `alembic revision
# --autogenerate` compares against the live database to produce a diff.
target_metadata = Base.metadata

# Both modes below pass the same three options explicitly rather than
# unpacking a shared dict: mypy cannot check **dict[str, object] against
# configure()'s precise keyword types, and reports ~30 errors if you try.
#
#   compare_type           - also detect a column whose TYPE changed,
#                            not just added/removed ones. Off by default.
#   compare_server_default - likewise for changed server defaults.
# Without these, autogenerate silently misses real schema drift.


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it.

    `alembic upgrade head --sql` uses this. Useful when a DBA has to review
    or apply the SQL by hand, and handy for seeing exactly what will run.
    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations. The normal path."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # one short-lived connection; no pool needed
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        # Postgres DDL is transactional, so a failed migration rolls back
        # completely rather than leaving the schema half-changed.
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
