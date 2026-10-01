"""AI-related Hacker News stories.

This source exists partly to prove the abstraction is not secretly "an RSS
parser". Hacker News has no usable feed, so this subclasses BaseSource
directly, speaks JSON, and the runner cannot tell the difference.

It uses the Algolia search API rather than the Firebase one. Firebase would
need /topstories.json plus one request per story - around 100 requests for
one fetch. Algolia returns matching stories, already filtered and dated, in
one request per query.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import urlencode

from lodestar.core.logging import get_logger
from lodestar.domain.article import ArticleIn
from lodestar.ingestion.base import BaseSource
from lodestar.ingestion.registry import register
from lodestar.storage.models import Source

log = get_logger(__name__)

_API = "https://hn.algolia.com/api/v1/search_by_date"


@register
class HackerNewsSource(BaseSource):
    name: ClassVar[Source] = Source.HACKERNEWS

    #: Hacker News is a general-interest site, so unlike the other four
    #: sources it needs a topic filter. Crude on purpose: the retrieval
    #: pipeline in P4 is what decides relevance, not this.
    queries: ClassVar[tuple[str, ...]] = (
        "artificial intelligence",
        "LLM",
        "machine learning",
    )

    #: Skip stories nobody engaged with - they are usually spam or noise.
    min_points: ClassVar[int] = 10

    hits_per_query: ClassVar[int] = 50

    def fetch(self, since: datetime) -> list[ArticleIn]:
        articles: list[ArticleIn] = []
        seen: set[str] = set()

        for query in self.queries:
            params = urlencode({
                "tags": "story",
                "query": query,
                "hitsPerPage": self.hits_per_query,
            })
            payload = json.loads(self._get(f"{_API}?{params}"))

            for hit in payload.get("hits", []):
                try:
                    article = self._to_article(hit, since)
                except Exception as exc:
                    log.warning(
                        "hn_hit_skipped", query=query, error=str(exc)
                    )
                    continue
                if article is None or article.external_id in seen:
                    continue
                seen.add(article.external_id)
                articles.append(article)

        log.info("source_fetched", source=str(self.name), count=len(articles))
        return articles

    def _to_article(self, hit: dict[str, Any], since: datetime) -> ArticleIn | None:
        story_id = hit.get("objectID")
        title = hit.get("title")
        created = hit.get("created_at_i")
        if not story_id or not title or created is None:
            return None

        published_at = datetime.fromtimestamp(int(created), tz=UTC)
        if published_at < since:
            return None

        points = int(hit.get("points") or 0)
        if points < self.min_points:
            return None

        # Ask HN and text posts have no external url; link to the discussion.
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={story_id}"

        return ArticleIn(
            source=self.name,
            external_id=str(story_id),
            title=title,
            url=url,
            published_at=published_at,
            author=hit.get("author"),
            summary=hit.get("story_text") or None,
            meta={
                "points": points,
                "num_comments": int(hit.get("num_comments") or 0),
                "discussion_url": f"https://news.ycombinator.com/item?id={story_id}",
            },
        )
