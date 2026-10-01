"""Importing this package registers every source.

The @register decorator only runs when the module is imported, so these
imports are what populate the registry. Without them the registry would be
empty and the runner would silently fetch nothing.
"""

from lodestar.ingestion.sources import (
    anthropic,
    arxiv,
    hackernews,
    openai,
    youtube,
)

__all__ = ["anthropic", "arxiv", "hackernews", "openai", "youtube"]
