"""OpenAI news, from their official RSS feed."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from lodestar.domain.article import ArticleIn
from lodestar.ingestion.base import RssSource
from lodestar.ingestion.registry import register
from lodestar.storage.models import Source


@register
class OpenAISource(RssSource):
    name: ClassVar[Source] = Source.OPENAI
    feed_urls: ClassVar[tuple[str, ...]] = ("https://openai.com/news/rss.xml",)

    def _to_article(self, entry: Any, published_at: datetime) -> ArticleIn | None:
        link = entry.get("link")
        if not link:
            return None
        return ArticleIn(
            source=self.name,
            # Prefer the feed's own guid; fall back to the URL, which is
            # stable enough to deduplicate on.
            external_id=entry.get("id") or link,
            title=entry.get("title", ""),
            url=link,
            published_at=published_at,
            summary=entry.get("summary") or entry.get("description"),
            meta={"category": _first_tag(entry)},
        )


def _first_tag(entry: Any) -> str | None:
    tags = entry.get("tags") or []
    return tags[0].get("term") if tags else None
