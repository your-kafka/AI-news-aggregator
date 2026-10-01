"""Tests for content enrichment. No network, no YouTube.

The httpx client and the transcript API are both injectable, so these tests
exercise the real classification logic against fake transports.

The behaviour that matters most is the permanent/transient split: get it
wrong and either the queue never drains (retrying a 404 forever) or real
articles are dropped after one flaky timeout.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from lodestar.enrichment.content import (
    MIN_CONTENT_CHARS,
    ContentFetcher,
    ContentTemporarilyUnavailable,
    ContentUnavailable,
    extract_article_text,
)
from lodestar.storage.models import Article, Source

PROSE = (
    "Retrieval augmented generation combines a retriever with a generator. "
    "The retriever narrows a large corpus to a handful of passages, and the "
    "generator conditions on those passages when producing an answer. "
) * 4


def make_article(source: Source = Source.OPENAI, **kw: Any) -> Article:
    data: dict[str, Any] = {
        "id": uuid.uuid4(),
        "source": source,
        "external_id": "x1",
        "title": "A title",
        "url": "https://example.com/post",
        "published_at": datetime.now(UTC),
    }
    data.update(kw)
    return Article(**data)


def client_returning(status: int = 200, body: str = "") -> httpx.Client:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def client_raising(exc: Exception) -> httpx.Client:
    def handler(_: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.Client(transport=httpx.MockTransport(handler))


def page(body_html: str) -> str:
    return f"""<html><head><title>T</title></head><body>
      <nav><a href="/x">Home</a><a href="/y">Products</a></nav>
      <article>{body_html}</article>
      <footer>Copyright 2026. Cookie notice. Subscribe to our newsletter.</footer>
    </body></html>"""


# ---------------------------------------------------------------------------
#  HTML extraction
# ---------------------------------------------------------------------------


def test_extraction_keeps_the_article_and_drops_the_chrome() -> None:
    """Boilerplate is not harmless. Left in, every chunked document carries
    the same nav and footer text, which makes unrelated documents look
    similar to each other and to every query."""
    text = extract_article_text(page(f"<h1>Real heading</h1><p>{PROSE}</p>"))

    assert "Real heading" in text
    assert "retriever" in text
    assert "Cookie notice" not in text
    assert "Subscribe to our newsletter" not in text
    assert "Products" not in text


def test_extraction_produces_markdown() -> None:
    text = extract_article_text(page(f"<h1>Heading</h1><p>{PROSE}</p>"))

    assert "# Heading" in text


def test_too_little_text_is_a_permanent_failure() -> None:
    """A paywall, a JS-only page or a redirect stub extracts to almost
    nothing. Storing that is worse than storing nothing, because it looks
    like content to everything downstream."""
    with pytest.raises(ContentUnavailable, match="extraction produced"):
        extract_article_text(page("<p>Too short.</p>"))


# ---------------------------------------------------------------------------
#  Failure classification - the part that makes the queue drain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 410, 451])
def test_client_errors_are_permanent(status: int) -> None:
    fetcher = ContentFetcher(client=client_returning(status))

    with pytest.raises(ContentUnavailable, match=f"HTTP {status}"):
        fetcher.fetch(make_article())


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 419])
def test_server_errors_and_rate_limits_are_transient(status: int) -> None:
    """A 429 or a 503 means try later. Treating these as permanent would
    silently discard good articles after one bad minute."""
    fetcher = ContentFetcher(client=client_returning(status))

    with pytest.raises(ContentTemporarilyUnavailable, match=f"HTTP {status}"):
        fetcher.fetch(make_article())


def test_network_errors_are_transient() -> None:
    fetcher = ContentFetcher(client=client_raising(httpx.ConnectTimeout("slow")))

    with pytest.raises(ContentTemporarilyUnavailable, match="ConnectTimeout"):
        fetcher.fetch(make_article())


# ---------------------------------------------------------------------------
#  arXiv - the abstract is the content
# ---------------------------------------------------------------------------


def test_arxiv_uses_the_abstract_without_any_network_call() -> None:
    """Parsing the PDF would add a heavy dependency for little gain: an
    abstract is already dense and self-contained, so it chunks well."""
    fetcher = ContentFetcher(client=client_raising(AssertionError("must not fetch")))

    text = fetcher.fetch(make_article(Source.ARXIV, summary=PROSE))

    assert text.startswith("Retrieval augmented generation")


def test_arxiv_without_a_usable_abstract_is_permanent() -> None:
    fetcher = ContentFetcher()

    with pytest.raises(ContentUnavailable, match="abstract too short"):
        fetcher.fetch(make_article(Source.ARXIV, summary="Short."))


# ---------------------------------------------------------------------------
#  YouTube transcripts
# ---------------------------------------------------------------------------


class FakeSnippet:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeTranscript:
    def __init__(self, texts: list[str]) -> None:
        self.snippets = [FakeSnippet(t) for t in texts]


class FakeTranscriptApi:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def fetch(self, video_id: str) -> Any:
        if self._error is not None:
            raise self._error
        return self._result


def test_transcript_snippets_are_joined_into_one_body() -> None:
    api = FakeTranscriptApi(FakeTranscript([PROSE[:200], PROSE[200:400], PROSE[400:]]))
    fetcher = ContentFetcher(transcript_api=api)

    text = fetcher.fetch(make_article(Source.YOUTUBE))

    assert len(text) >= MIN_CONTENT_CHARS
    assert "retriever" in text


def test_transcripts_disabled_is_permanent() -> None:
    """The video exists but has no captions and never will. Retrying wastes
    a request on every run forever."""
    from youtube_transcript_api import _errors as yt

    error = yt.TranscriptsDisabled("vid")
    fetcher = ContentFetcher(transcript_api=FakeTranscriptApi(error=error))

    with pytest.raises(ContentUnavailable, match="TranscriptsDisabled"):
        fetcher.fetch(make_article(Source.YOUTUBE))


def test_blocked_ip_is_transient_not_permanent() -> None:
    """YouTube blocks datacentre IPs. That is about where we are calling
    from, not about the video, so it must not mark the video unusable."""
    from youtube_transcript_api import _errors as yt

    error = yt.IpBlocked("vid")
    fetcher = ContentFetcher(transcript_api=FakeTranscriptApi(error=error))

    with pytest.raises(ContentTemporarilyUnavailable, match="IpBlocked"):
        fetcher.fetch(make_article(Source.YOUTUBE))


def test_empty_transcript_is_permanent() -> None:
    api = FakeTranscriptApi(FakeTranscript(["um", "yeah"]))
    fetcher = ContentFetcher(transcript_api=api)

    with pytest.raises(ContentUnavailable, match="transcript too short"):
        fetcher.fetch(make_article(Source.YOUTUBE))


# ---------------------------------------------------------------------------
#  Anthropic title cleanup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "title", "category"),
    [
        ("Oct 1, 2026ScienceClaude-shaped science", "Claude-shaped science", "Science"),
        ("Sep 30, 2026EconomicsWhat work can robots do?", "What work can robots do?", "Economics"),
        (
            "Sep 29, 2026Frontier Red TeamGLM-5.3 and cyber capabilities",
            "GLM-5.3 and cyber capabilities",
            "Frontier Red Team",
        ),
        (
            "Sep 29, 2026Societal ImpactsWhat do you want from AI?",
            "What do you want from AI?",
            "Societal Impacts",
        ),
        ("Barclays scales Claude to upgrade operations", "Barclays scales Claude to upgrade operations", None),
    ],
)
def test_anthropic_titles_are_split_from_date_and_category(
    raw: str, title: str, category: str | None
) -> None:
    """The mirror feed glues date + category + title together. Left alone, a
    date and a section name end up in the index as if they were the subject
    of every Anthropic article."""
    from lodestar.ingestion.sources.anthropic import split_title

    assert split_title(raw) == (title, category)


@pytest.mark.parametrize(
    "raw",
    [
        "Oct 1, 2026ScienceClaude-shaped science",
        "  Oct 1, 2026ScienceClaude-shaped science",
        "\n      Oct 1, 2026ScienceClaude-shaped science",
        "\n\t Oct 1, 2026ScienceClaude-shaped science  \n",
    ],
)
def test_title_cleanup_survives_the_whitespace_rss_adds(raw: str) -> None:
    """Regression test for a bug the original tests could not catch.

    split_title anchors its date pattern with ^, and RSS wraps titles in
    newlines and indentation - so entries with whitespace were silently left
    uncleaned while the tidy ones worked. The first test only ever passed
    pre-stripped strings, so it reported success on broken code.
    """
    from lodestar.ingestion.sources.anthropic import split_title

    assert split_title(raw) == ("Claude-shaped science", "Science")


# ---------------------------------------------------------------------------
#  Hacker News text posts
# ---------------------------------------------------------------------------


def test_hn_text_posts_use_the_summary_instead_of_fetching() -> None:
    """An Ask HN post's url is our fallback link to the discussion page.
    Fetching that returned HTTP 419 five times for text Algolia had already
    given us."""
    fetcher = ContentFetcher(client=client_raising(AssertionError("must not fetch")))
    article = make_article(
        Source.HACKERNEWS,
        url="https://news.ycombinator.com/item?id=4242",
        summary=PROSE,
    )

    assert fetcher.fetch(article).startswith("Retrieval augmented")


def test_hn_link_posts_still_fetch_the_linked_article() -> None:
    body = page(f"<h1>Linked</h1><p>{PROSE}</p>")
    fetcher = ContentFetcher(client=client_returning(200, body))
    article = make_article(Source.HACKERNEWS, url="https://example.com/article")

    assert "Linked" in fetcher.fetch(article)


def test_browser_headers_are_sent() -> None:
    """A bare User-Agent got 403 from openai.com for 25 of 71 articles. Real
    browsers also send Accept, Accept-Language and the Sec-Fetch-* set."""
    from lodestar.enrichment.content import BROWSER_HEADERS

    assert "User-Agent" in BROWSER_HEADERS
    assert "Accept-Language" in BROWSER_HEADERS
    assert BROWSER_HEADERS["Sec-Fetch-Mode"] == "navigate"
