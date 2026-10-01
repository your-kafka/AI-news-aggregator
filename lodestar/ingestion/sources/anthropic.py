"""Anthropic news, research and engineering posts.

Anthropic publishes no RSS of its own, so these are community-maintained
mirrors - the same ones the reference project uses. Three feeds, one source:
RssSource loops over feed_urls and deduplicates by external_id.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from lodestar.domain.article import ArticleIn
from lodestar.ingestion.base import RssSource
from lodestar.ingestion.registry import register
from lodestar.storage.models import Source

_MIRROR = "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds"


@register
class AnthropicSource(RssSource):
    name: ClassVar[Source] = Source.ANTHROPIC
    feed_urls: ClassVar[tuple[str, ...]] = (
        f"{_MIRROR}/feed_anthropic_news.xml",
        f"{_MIRROR}/feed_anthropic_research.xml",
        f"{_MIRROR}/feed_anthropic_engineering.xml",
    )

    def _to_article(self, entry: Any, published_at: datetime) -> ArticleIn | None:
        link = entry.get("link")
        if not link:
            return None
        return ArticleIn(
            source=self.name,
            external_id=entry.get("id") or link,
            title=entry.get("title", ""),
            url=link,
            published_at=published_at,
            summary=entry.get("summary") or entry.get("description"),
        )
