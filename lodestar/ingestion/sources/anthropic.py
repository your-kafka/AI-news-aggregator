"""Anthropic news, research and engineering posts.

Anthropic publishes no RSS of its own, so these are community-maintained
mirrors - the same ones the reference project uses. Three feeds, one source:
RssSource loops over feed_urls and deduplicates by external_id.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, ClassVar

from lodestar.domain.article import ArticleIn
from lodestar.ingestion.base import RssSource
from lodestar.ingestion.registry import register
from lodestar.storage.models import Source

_MIRROR = "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds"


# The mirror feed concatenates date, category and title into one string:
#   "Oct 1, 2026ScienceClaude-shaped science"
# Left alone that text goes straight into the BM25 index and the embeddings,
# so every Anthropic document carries a date and a category as if they were
# part of its subject. Cleaned here, at the boundary, rather than downstream.
_LEADING_DATE = re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}")

# Anthropic's own section names, longest first so "Societal Impacts" is
# matched before any shorter prefix of it.
_CATEGORIES = sorted(
    (
        "Announcements", "Alignment", "Company", "Customers", "Economics",
        "Education", "Engineering", "Frontier Red Team", "Interpretability",
        "News", "Policy", "Product", "Research", "Science",
        "Societal Impacts",
    ),
    key=len,
    reverse=True,
)


_CATEGORY_RE = re.compile("^(" + "|".join(re.escape(c) for c in _CATEGORIES) + ")")


def split_title(raw: str) -> tuple[str, str | None]:
    """Return (title, category) from a mirror-feed title or summary string.

    The feed glues a date and a category onto the front, and it uses BOTH
    orders - "Oct 1, 2026Science..." in the title field and
    "AlignmentSep 9, 2026..." in the summary field - so this strips whichever
    comes next until neither matches.

    A clean string is returned untouched, so the posts that arrive tidy are
    not mangled by a cleaner aimed at the ones that do not.
    """
    # .strip() FIRST. RSS wraps titles in newlines and indentation, and the
    # ^ anchors below then never match - which silently did nothing for the
    # entries that had whitespace, while the tidy ones worked fine.
    text = raw.strip()
    category: str | None = None

    for _ in range(4):  # at most a date and a category, in either order
        match = _CATEGORY_RE.match(text)
        if match:
            category = match.group(1)
            text = text[match.end():].lstrip()
            continue
        match = _LEADING_DATE.match(text)
        if match:
            text = text[match.end():].lstrip()
            continue
        break

    return text, category


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
        title, category = split_title(entry.get("title", ""))
        raw_summary = entry.get("summary") or entry.get("description") or ""
        # The summary carries the same date/category noise, in the other order.
        summary, summary_category = split_title(raw_summary)
        category = category or summary_category

        return ArticleIn(
            source=self.name,
            external_id=entry.get("id") or link,
            title=title,
            url=link,
            published_at=published_at,
            summary=summary or None,
            meta={"category": category} if category else {},
        )
