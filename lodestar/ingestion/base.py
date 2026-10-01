"""The interface every source implements.

Two layers:

    BaseSource  - the contract: fetch(since) -> list[ArticleIn]. Nothing
                  about RSS, so a JSON API fits just as well.
    RssSource   - shared machinery for the four feed-based sources: HTTP,
                  parsing, date filtering, deduplication. A subclass only
                  maps one feed entry to one ArticleIn.

Why an interface at all: the reference project's three scrapers have three
different method names and signatures, so its runner has to know each one
by name. Here the runner asks the registry for every source and calls the
same method. Adding a source is adding a file.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, ClassVar

import feedparser
import httpx

from lodestar.core.logging import get_logger
from lodestar.domain.article import ArticleIn
from lodestar.storage.models import Source

log = get_logger(__name__)

# A real User-Agent. Several feed hosts reject the default python-httpx one.
USER_AGENT = "Lodestar/0.1 (+https://github.com/your-kafka/AI-news-aggregator)"

# Total seconds before giving up on one request. feedparser.parse(url) - which
# the reference project uses - has NO timeout, so a hung feed server stalls
# the whole pipeline indefinitely.
DEFAULT_TIMEOUT = 20.0


class SourceFetchError(RuntimeError):
    """A source could not be fetched. Raised so the runner can skip it."""


class BaseSource(ABC):
    """One place articles come from."""

    #: Which Source enum member this class supplies. Used as its registry key.
    name: ClassVar[Source]

    def __init__(self, client: httpx.Client | None = None) -> None:
        # The client is injectable so tests can supply a MockTransport and
        # never touch the network.
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=DEFAULT_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
                # Retry connection-level failures. Does not retry on HTTP
                # error statuses - those are handled per source.
                transport=httpx.HTTPTransport(retries=2),
            )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    @abstractmethod
    def fetch(self, since: datetime) -> list[ArticleIn]:
        """Return articles published at or after `since`.

        Must not raise for a single bad entry - skip it and log. Raise
        SourceFetchError only when the whole source is unreachable.
        """

    def _get(self, url: str) -> bytes:
        """One HTTP GET, with timeouts, raising SourceFetchError on failure."""
        try:
            response = self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceFetchError(f"{self.name}: GET {url} failed: {exc}") from exc
        return response.content


class RssSource(BaseSource):
    """A source backed by one or more RSS/Atom feeds."""

    #: Feeds to read. Several, because Anthropic publishes three.
    #: Static sources just set this. Sources whose URLs depend on settings
    #: (arXiv categories, YouTube channel ids) override resolve_feed_urls().
    feed_urls: ClassVar[tuple[str, ...]] = ()

    def resolve_feed_urls(self) -> tuple[str, ...]:
        return self.feed_urls

    @abstractmethod
    def _to_article(self, entry: Any, published_at: datetime) -> ArticleIn | None:
        """Map one parsed feed entry to an ArticleIn, or None to skip it."""

    def fetch(self, since: datetime) -> list[ArticleIn]:
        articles: list[ArticleIn] = []
        seen: set[str] = set()

        for url in self.resolve_feed_urls():
            # Fetch the bytes ourselves, then parse. Parsing from bytes keeps
            # the network out of feedparser, which makes this testable.
            feed = feedparser.parse(self._get(url))

            if feed.bozo and not feed.entries:
                raise SourceFetchError(f"{self.name}: unparseable feed {url}")

            for entry in feed.entries:
                published_at = _entry_datetime(entry)
                if published_at is None or published_at < since:
                    continue
                try:
                    article = self._to_article(entry, published_at)
                except Exception as exc:
                    # One malformed entry must never lose the other 49.
                    log.warning(
                        "feed_entry_skipped",
                        source=str(self.name),
                        feed=url,
                        error=str(exc),
                    )
                    continue
                if article is None or article.external_id in seen:
                    continue
                seen.add(article.external_id)
                articles.append(article)

        log.info("source_fetched", source=str(self.name), count=len(articles))
        return articles


def _entry_datetime(entry: Any) -> datetime | None:
    """Read a timezone-aware datetime out of a feedparser entry.

    feedparser hands back a time.struct_time in UTC, already normalised from
    whatever the feed used. It drops the original offset, so UTC is the
    correct and only choice here - unlike inventing a timezone for a naive
    value, which ArticleIn rejects.
    """
    parsed = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if parsed is None:
        return None
    return datetime.fromtimestamp(time.mktime(parsed) - time.timezone, tz=UTC)
