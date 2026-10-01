"""Tests for the ingestion layer. No network access.

Every source takes an injectable httpx.Client, so these tests hand them a
MockTransport returning recorded payloads. That matters for CI: a test that
fetches a live RSS feed fails when the feed is down, changes shape, or rate
limits you - which makes the build flaky for reasons that have nothing to do
with your code.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from lodestar.ingestion import registry
from lodestar.ingestion.base import BaseSource, SourceFetchError
from lodestar.ingestion.sources.anthropic import AnthropicSource
from lodestar.ingestion.sources.arxiv import ArxivSource
from lodestar.ingestion.sources.hackernews import HackerNewsSource
from lodestar.ingestion.sources.openai import OpenAISource
from lodestar.ingestion.sources.youtube import YouTubeSource
from lodestar.storage.models import Source

LONG_AGO = datetime(2020, 1, 1, tzinfo=UTC)


def mock_client(body: str | bytes, status: int = 200) -> httpx.Client:
    """A Client that answers every request with the same payload."""

    def handler(_: httpx.Request) -> httpx.Response:
        content = body.encode() if isinstance(body, str) else body
        return httpx.Response(status, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def rss(items: str) -> str:
    return f"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test</title>{items}</channel></rss>"""


def item(
    title: str = "A post",
    link: str = "https://openai.com/index/a-post",
    guid: str = "guid-1",
    date: str = "Wed, 01 Oct 2026 10:00:00 GMT",
    description: str = "Some summary",
) -> str:
    return f"""<item>
      <title>{title}</title><link>{link}</link><guid>{guid}</guid>
      <pubDate>{date}</pubDate><description>{description}</description>
    </item>"""


# ---------------------------------------------------------------------------
#  RSS parsing
# ---------------------------------------------------------------------------


def test_openai_source_parses_a_feed() -> None:
    source = OpenAISource(client=mock_client(rss(item())))

    articles = source.fetch(LONG_AGO)

    assert len(articles) == 1
    article = articles[0]
    assert article.source is Source.OPENAI
    assert article.external_id == "guid-1"
    assert article.title == "A post"
    assert article.url == "https://openai.com/index/a-post"
    assert article.published_at.tzinfo is not None
    assert article.summary == "Some summary"


def test_entries_older_than_since_are_dropped() -> None:
    """Incremental fetching: a normal run must not re-read the whole feed."""
    feed = rss(
        item(guid="old", date="Mon, 01 Jan 2024 10:00:00 GMT")
        + item(guid="new", date="Wed, 01 Oct 2026 10:00:00 GMT")
    )
    source = OpenAISource(client=mock_client(feed))

    articles = source.fetch(datetime(2026, 1, 1, tzinfo=UTC))

    assert [a.external_id for a in articles] == ["new"]


def test_entry_without_a_date_is_skipped_not_guessed() -> None:
    feed = rss("<item><title>No date</title><link>https://x.com/a</link></item>")

    assert OpenAISource(client=mock_client(feed)).fetch(LONG_AGO) == []


def test_one_bad_entry_does_not_lose_the_others() -> None:
    """A malformed entry must cost one article, not the whole batch."""
    feed = rss(
        item(guid="ok-1")
        # No link, so _to_article returns None.
        + "<item><title>Broken</title><pubDate>Wed, 01 Oct 2026 10:00:00 GMT</pubDate></item>"
        + item(guid="ok-2", link="https://openai.com/index/b")
    )
    source = OpenAISource(client=mock_client(feed))

    articles = source.fetch(LONG_AGO)

    assert {a.external_id for a in articles} == {"ok-1", "ok-2"}


def test_duplicate_guids_across_feeds_are_deduplicated() -> None:
    """Anthropic reads three feeds; the same post appears in more than one."""
    source = AnthropicSource(client=mock_client(rss(item(guid="same"))))

    articles = source.fetch(LONG_AGO)

    assert len(source.resolve_feed_urls()) == 3  # three feeds fetched
    assert len(articles) == 1  # one article kept


def test_http_error_raises_so_the_runner_can_skip_the_source() -> None:
    source = OpenAISource(client=mock_client("nope", status=500))

    with pytest.raises(SourceFetchError) as exc:
        source.fetch(LONG_AGO)

    assert "openai" in str(exc.value)


# ---------------------------------------------------------------------------
#  arXiv
# ---------------------------------------------------------------------------

ARXIV_ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.01234v3</id>
    <title>A Very
      Long   Wrapped
      Title</title>
    <published>2026-09-30T10:00:00Z</published>
    <summary>An abstract
      wrapped over lines.</summary>
    <link href="http://arxiv.org/abs/2601.01234v3"/>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
  </entry>
</feed>"""


def test_arxiv_strips_the_version_from_the_id() -> None:
    """v3 is a revision of the same paper, not a new one. Keeping the version
    would let every revision reappear as a fresh article."""
    articles = ArxivSource(client=mock_client(ARXIV_ATOM)).fetch(LONG_AGO)

    assert articles[0].external_id == "2601.01234"


def test_arxiv_collapses_wrapped_whitespace() -> None:
    """arXiv hard-wraps titles and abstracts. Left alone, the newlines end up
    in the search index and in the UI."""
    article = ArxivSource(client=mock_client(ARXIV_ATOM)).fetch(LONG_AGO)[0]

    assert article.title == "A Very Long Wrapped Title"
    assert "\n" not in (article.summary or "")


def test_arxiv_summarises_multiple_authors() -> None:
    article = ArxivSource(client=mock_client(ARXIV_ATOM)).fetch(LONG_AGO)[0]

    assert article.author == "Ada Lovelace et al."


def test_arxiv_builds_its_query_from_settings() -> None:
    url = ArxivSource().resolve_feed_urls()[0]

    assert "cat%3Acs.AI" in url or "cat:cs.AI" in url
    assert "sortBy=submittedDate" in url


# ---------------------------------------------------------------------------
#  YouTube
# ---------------------------------------------------------------------------

YT_ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015">
  <entry>
    <yt:videoId>abc123</yt:videoId>
    <yt:channelId>UCchan</yt:channelId>
    <title>A real video</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=abc123"/>
    <published>2026-10-01T10:00:00+00:00</published>
    <author><name>Some Channel</name></author>
  </entry>
  <entry>
    <yt:videoId>short1</yt:videoId>
    <title>A short</title>
    <link rel="alternate" href="https://www.youtube.com/shorts/short1"/>
    <published>2026-10-01T11:00:00+00:00</published>
  </entry>
</feed>"""


def test_youtube_skips_shorts() -> None:
    articles = YouTubeSource(client=mock_client(YT_ATOM)).fetch(LONG_AGO)

    assert [a.external_id for a in articles] == ["abc123"]


def test_youtube_leaves_content_empty_for_the_enrichment_queue() -> None:
    """Transcripts are fetched later, not during scraping: one call per video
    is slow, YouTube blocks datacentre IPs, and a failure would lose the
    whole batch. content=None puts the video in the work queue instead."""
    article = YouTubeSource(client=mock_client(YT_ATOM)).fetch(LONG_AGO)[0]

    assert article.content is None
    assert article.meta["channel_id"] == "UCchan"


# ---------------------------------------------------------------------------
#  Hacker News - JSON, not RSS
# ---------------------------------------------------------------------------


def hn_payload(*hits: dict[str, Any]) -> str:
    return json.dumps({"hits": list(hits)})


def hn_hit(**overrides: Any) -> dict[str, Any]:
    hit = {
        "objectID": "4242",
        "title": "An AI story",
        "url": "https://example.com/story",
        "created_at_i": int(datetime(2026, 10, 1, tzinfo=UTC).timestamp()),
        "points": 150,
        "author": "someone",
        "num_comments": 42,
    }
    hit.update(overrides)
    return hit


def test_hackernews_parses_json_not_rss() -> None:
    """Proves the abstraction is not secretly an RSS parser: this source
    subclasses BaseSource directly and the runner cannot tell."""
    source = HackerNewsSource(client=mock_client(hn_payload(hn_hit())))

    articles = source.fetch(LONG_AGO)

    assert isinstance(source, BaseSource)
    assert len(articles) == 1
    assert articles[0].external_id == "4242"
    assert articles[0].meta["points"] == 150


def test_hackernews_drops_low_scoring_noise() -> None:
    source = HackerNewsSource(client=mock_client(hn_payload(hn_hit(points=2))))

    assert source.fetch(LONG_AGO) == []


def test_hackernews_links_to_the_discussion_when_there_is_no_url() -> None:
    """Ask HN and text posts have no external link."""
    source = HackerNewsSource(client=mock_client(hn_payload(hn_hit(url=None))))

    article = source.fetch(LONG_AGO)[0]

    assert article.url == "https://news.ycombinator.com/item?id=4242"


def test_hackernews_respects_since() -> None:
    old = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())
    source = HackerNewsSource(client=mock_client(hn_payload(hn_hit(created_at_i=old))))

    assert source.fetch(datetime(2026, 1, 1, tzinfo=UTC)) == []


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry() -> Iterator[None]:
    """Run with an empty registry, then restore the real one."""
    saved = dict(registry._REGISTRY)
    registry.clear_registry()
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)


def test_all_five_sources_are_registered() -> None:
    import lodestar.ingestion.sources  # noqa: F401  (import triggers @register)

    assert {s.value for s in registry.registered_names()} == {
        "openai", "anthropic", "youtube", "arxiv", "hackernews",
    }


def test_registering_the_same_source_twice_is_an_error(
    isolated_registry: None,
) -> None:
    """Catches a copy-paste where a new source keeps the old name, which
    would otherwise silently replace it and drop a whole feed."""

    class First(BaseSource):
        name = Source.OPENAI

        def fetch(self, since: datetime) -> list[Any]:
            return []

    class Second(BaseSource):
        name = Source.OPENAI

        def fetch(self, since: datetime) -> list[Any]:
            return []

    registry.register(First)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Second)


def test_unknown_source_name_gives_a_helpful_error(isolated_registry: None) -> None:
    with pytest.raises(KeyError, match="no source registered"):
        registry.get_source_class(Source.OPENAI)


def test_source_without_a_name_is_rejected(isolated_registry: None) -> None:
    class Nameless(BaseSource):
        def fetch(self, since: datetime) -> list[Any]:
            return []

    with pytest.raises(TypeError, match="class-level"):
        registry.register(Nameless)


# ---------------------------------------------------------------------------
#  Timestamps
# ---------------------------------------------------------------------------


def test_feed_dates_become_timezone_aware_utc() -> None:
    """ArticleIn rejects naive datetimes, so a parsing slip here would fail
    loudly rather than silently storing the wrong instant."""
    feed = rss(item(date="Wed, 01 Oct 2026 10:00:00 +0530"))

    article = OpenAISource(client=mock_client(feed)).fetch(LONG_AGO)[0]

    assert article.published_at.tzinfo is not None
    assert article.published_at == datetime(2026, 10, 1, 4, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
#  Incremental fetching - which window does a run actually ask for?
#  No database needed: _default_since only calls last_successful().
# ---------------------------------------------------------------------------


class FakeRunRepo:
    """Stands in for RunRepository with one canned answer."""

    def __init__(self, last: object | None) -> None:
        self._last = last

    def last_successful(self) -> object | None:
        return self._last


class FakeRun:
    def __init__(self, started_at: datetime) -> None:
        self.started_at = started_at


def test_first_ever_run_looks_back_the_bootstrap_window() -> None:
    """With no previous run there is nothing to resume from, so it reads a
    fixed window rather than the entire history of every feed."""
    from lodestar.ingestion.runner import BOOTSTRAP_WINDOW, _default_since

    since = _default_since(FakeRunRepo(None))  # type: ignore[arg-type]

    expected = datetime.now(UTC) - BOOTSTRAP_WINDOW
    assert abs((since - expected).total_seconds()) < 5


def test_later_runs_resume_from_the_last_success_minus_an_overlap() -> None:
    """The overlap costs nothing - ON CONFLICT DO NOTHING makes re-reading an
    item free - and it catches feed items that arrive slightly out of order."""
    from lodestar.ingestion.runner import _default_since

    last_started = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

    since = _default_since(FakeRunRepo(FakeRun(last_started)))  # type: ignore[arg-type]

    assert since == datetime(2026, 10, 1, 10, 0, tzinfo=UTC)


def test_a_failed_run_does_not_become_the_resume_point() -> None:
    """last_successful() filters failures, so a crashed run cannot cause the
    next one to skip the articles it never managed to store."""
    from lodestar.ingestion.runner import BOOTSTRAP_WINDOW, _default_since

    since = _default_since(FakeRunRepo(None))  # type: ignore[arg-type]

    assert since < datetime.now(UTC) - BOOTSTRAP_WINDOW + timedelta(minutes=1)
