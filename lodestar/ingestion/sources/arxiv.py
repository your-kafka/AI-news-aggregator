"""Recent arXiv preprints in the configured categories.

arXiv's API returns Atom, so feedparser handles it and this stays an
RssSource - only the URL is built rather than fixed.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, ClassVar
from urllib.parse import urlencode

from lodestar.core.config import get_settings
from lodestar.domain.article import ArticleIn
from lodestar.ingestion.base import RssSource
from lodestar.ingestion.registry import register
from lodestar.storage.models import Source

_API = "https://export.arxiv.org/api/query"

#: arXiv caps a single response, so a large request is split into pages.
#: RssSource already loops over feed_urls, so pagination is just several URLs.
_PAGE_SIZE = 200

# "http://arxiv.org/abs/2601.01234v2" -> "2601.01234"
_ABS_ID = re.compile(r"/abs/(?P<id>[^v\s]+)(?:v\d+)?$")


@register
class ArxivSource(RssSource):
    name: ClassVar[Source] = Source.ARXIV

    #: arXiv's API terms ask for 3 seconds between requests.
    request_delay: ClassVar[float] = 3.0

    def resolve_feed_urls(self) -> tuple[str, ...]:
        settings = get_settings()
        query = " OR ".join(f"cat:{c}" for c in settings.arxiv_category_list)
        wanted = settings.arxiv_max_results

        urls = []
        for start in range(0, wanted, _PAGE_SIZE):
            params = urlencode({
                "search_query": query,
                "start": start,
                "max_results": min(_PAGE_SIZE, wanted - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            })
            urls.append(f"{_API}?{params}")
        return tuple(urls)

    def _to_article(self, entry: Any, published_at: datetime) -> ArticleIn | None:
        raw_id = entry.get("id", "")
        match = _ABS_ID.search(raw_id)
        if not match:
            return None

        return ArticleIn(
            source=self.name,
            # Version-stripped, so a v2 revision is treated as the same paper
            # rather than appearing again as a new article.
            external_id=match.group("id"),
            title=_squash(entry.get("title", "")),
            url=entry.get("link") or raw_id,
            published_at=published_at,
            author=_authors(entry),
            summary=_squash(entry.get("summary", "")),
            meta={"categories": [t.get("term") for t in entry.get("tags") or []]},
        )


def _squash(text: str) -> str:
    """Collapse internal whitespace. arXiv hard-wraps titles and abstracts."""
    return " ".join(text.split())


def _authors(entry: Any) -> str | None:
    names = [a.get("name") for a in entry.get("authors") or [] if a.get("name")]
    if not names:
        return None
    return names[0] if len(names) == 1 else f"{names[0]} et al."
