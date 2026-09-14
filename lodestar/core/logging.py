"""Logging setup.

Two modes, chosen by LODESTAR_ENV:

    local  ->  coloured, aligned, human-readable lines
    prod   ->  one JSON object per line, for Grafana/Loki to index

Your application code is identical in both. Only the final renderer swaps.

Usage:

    from lodestar.core.logging import configure_logging, get_logger

    configure_logging()          # once, at startup
    log = get_logger(__name__)   # in every module

    log.info("ingest_complete", source="openai", count=14)
                ^ event name      ^ everything else is structured data

Never use print(). Never format numbers into the message string - pass
them as keyword arguments so they stay queryable.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from lodestar.core.config import get_settings

# Third-party libraries that are far too chatty at DEBUG level.
# We keep our own logs verbose while silencing theirs.
_NOISY_LOGGERS = (
    "urllib3",
    "httpx",
    "httpcore",
    "asyncio",
    "sqlalchemy.engine",
)


def configure_logging(
    level: str | None = None,
    json_logs: bool | None = None,
) -> None:
    """Set up logging for the whole process. Call this once, at startup.

    Both arguments default to whatever the settings say, so normally you
    just call configure_logging(). They exist so tests can force a mode.
    """
    settings = get_settings()
    resolved_level = level or settings.log_level
    use_json = settings.is_production if json_logs is None else json_logs

    # The standard library still handles the actual writing to stdout.
    # structlog renders the line, stdlib prints it. Using stdout (not
    # stderr) matters in containers: `docker logs` reads both, but log
    # shippers conventionally treat stdout as the application stream.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=resolved_level,
        force=True,  # replace any handler a library installed behind our back
    )

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # The processor pipeline. Every log call flows through these in order,
    # each one adding a field, and the LAST one turns the dict into text.
    shared: list[Any] = [
        # Pull in anything bound via bind_context() - e.g. run_id.
        structlog.contextvars.merge_contextvars,
        # Which module logged this.
        structlog.stdlib.add_logger_name,
        # level="info" as a field, not just a prefix.
        structlog.stdlib.add_log_level,
        # UTC always. Local timezones in logs cause real confusion when
        # your laptop, the server and CI are in three different ones.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Adds stack info when you pass stack_info=True.
        structlog.processors.StackInfoRenderer(),
    ]

    renderers: list[Any]
    if use_json:
        renderers = [
            # Exceptions become a structured object rather than a blob of
            # text, so a traceback stays searchable field by field.
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        renderers = [
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=[*shared, *renderers],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Small speed win: after a logger is first used, stop rebuilding it.
        # Safe because we configure once at startup and never reconfigure.
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a logger. Pass __name__ so lines say which module they came from."""
    return structlog.get_logger(name)


def bind_context(**values: Any) -> None:
    """Attach values to EVERY later log line in this task or request.

    Bind run_id once when a pipeline run starts and all 50 lines that
    follow - across every module - carry it, without being passed around
    as a function argument. Filtering one run out of interleaved logs
    then becomes: run_id = "a3f9".

    Uses contextvars, so concurrent tasks each get their own copy.
    """
    structlog.contextvars.bind_contextvars(**values)


def clear_context() -> None:
    """Drop everything bound by bind_context. Call when a run finishes."""
    structlog.contextvars.clear_contextvars()
