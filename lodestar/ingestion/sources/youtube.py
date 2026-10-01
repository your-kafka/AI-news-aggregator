"""Videos from the configured YouTube channels.

Metadata only. Transcripts are NOT fetched here, deliberately: that would be
one network call per video, it is the part YouTube blocks from datacentre
IPs, and a failure would lose the whole batch.

Leaving content=None drops each video into the "content IS NULL" work queue
that ArticleRepository.list_needing_content() already exposes, so transcript
fetching becomes a separate, retryable step in P3.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from lodestar.core.config import get_settings
from lodestar.domain.article import ArticleIn
from lodestar.ingestion.base import RssSource
from lodestar.ingestion.registry import register
from lodestar.storage.models import Source

_FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


@register
class YouTubeSource(RssSource):
    name: ClassVar[Source] = Source.YOUTUBE

    def resolve_feed_urls(self) -> tuple[str, ...]:
        return tuple(
            _FEED.format(channel_id=cid) for cid in get_settings().youtube_channels
        )

    def _to_article(self, entry: Any, published_at: datetime) -> ArticleIn | None:
        link = entry.get("link", "")
        # Shorts are rarely substantive and have no useful transcript.
        if "/shorts/" in link:
            return None

        video_id = entry.get("yt_videoid") or _video_id_from(link)
        if not video_id:
            return None

        return ArticleIn(
            source=self.name,
            external_id=video_id,
            title=entry.get("title", ""),
            url=link,
            published_at=published_at,
            author=(entry.get("author") or None),
            summary=entry.get("summary"),
            content=None,  # filled by the enrichment step in P3
            meta={"channel_id": entry.get("yt_channelid")},
        )


def _video_id_from(url: str) -> str | None:
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None
