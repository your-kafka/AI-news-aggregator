"""The shape of an article as it ARRIVES, before it is stored.

This is the boundary between the outside world and our database. Scrapers
produce ArticleIn objects; repositories consume them. Validation happens
here, so malformed feed data is rejected with a clear message naming the
field instead of becoming a confusing IntegrityError, or worse, a bad row.

It is deliberately separate from storage.models.Article:
  ArticleIn  - what a source gives us. No id, no timestamps.
  Article    - what the database holds. Has an id, created_at, updated_at.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lodestar.storage.models import Source


class ArticleIn(BaseModel):
    """One item fetched from a source, validated and ready to store."""

    model_config = ConfigDict(
        # Reject unknown fields instead of silently dropping them. A typo in
        # a scraper ("tittle=") then fails loudly rather than storing NULL.
        extra="forbid",
        # Trim whitespace on every string: RSS titles very often arrive
        # wrapped in newlines and indentation.
        str_strip_whitespace=True,
    )

    source: Source
    external_id: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=1024)
    url: str = Field(min_length=1, max_length=2048)
    published_at: datetime

    author: str | None = Field(default=None, max_length=256)
    summary: str | None = None
    content: str | None = None

    # Source-specific extras that don't deserve a column: channel_id,
    # arXiv categories, HN score.
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at")
    @classmethod
    def must_be_timezone_aware(cls, value: datetime) -> datetime:
        """Reject naive datetimes.

        The column is TIMESTAMPTZ. Handing it a naive datetime makes Postgres
        assume a timezone, and which one depends on server configuration - so
        the same feed can land at different instants on different machines.
        Feeds give us a timezone; refuse to guess if one is missing.
        """
        if value.tzinfo is None:
            raise ValueError(
                "published_at must be timezone-aware "
                "(parse the feed's timezone, do not assume UTC)"
            )
        return value

    @field_validator("url")
    @classmethod
    def must_be_http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"url must start with http:// or https://, got {value!r}")
        return value

    def to_row(self) -> dict[str, Any]:
        """As a dict for a bulk INSERT statement.

        Returned instead of an Article instance because bulk upserts use
        Core-level INSERT ... ON CONFLICT, which takes plain dicts.
        """
        return self.model_dump()
