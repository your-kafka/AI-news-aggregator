"""Fetch the real body text for articles that only have a title and summary.

Why this exists: at ingestion time a feed gives us a title and a one- or
two-line description. An OpenAI post averages 160 characters of summary and
an Anthropic post 57 - far too little to chunk, embed, or retrieve against.
RAG needs bodies.

It drains the work queue ArticleRepository exposes: articles where
`content IS NULL` and the attempt count is still under the limit. Dispatch
is by source:

    openai / anthropic / hackernews -> GET the URL, extract the article
    youtube                         -> fetch the transcript
    arxiv                           -> the abstract we already have

TWO KINDS OF FAILURE, which is what makes the queue drain:

    permanent  - a 404, a deleted video, transcripts switched off. Asking
                 again will never help, so the attempt count jumps straight
                 to the limit.
    transient  - a timeout, a 429, a 5xx, a blocked IP. Worth retrying, so
                 the attempt count goes up by one.

The reference project instead writes the literal string "__UNAVAILABLE__"
into the transcript column. That both pollutes the text it later embeds and
cannot distinguish "never going to work" from "try again in an hour".
"""

from __future__ import annotations

from typing import Any

import httpx
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import _errors as yt_errors

from lodestar.core.logging import get_logger
from lodestar.storage.models import Article, Source
from lodestar.storage.repositories import ArticleRepository
from lodestar.storage.session import session_scope

log = get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TIMEOUT = 25.0

#: Headers a real browser sends. Sites fingerprint the absence of these.
BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

#: Extraction below this length means trafilatura found navigation chrome
#: rather than an article - a paywall, a JS-only page, a redirect stub.
#: Embedding that is worse than having nothing, because it looks like content.
MIN_CONTENT_CHARS = 250

#: HTTP statuses where retrying cannot help.
PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 405, 410, 451})


class ContentUnavailable(Exception):
    """This body will never be fetchable. Stop asking."""


class ContentTemporarilyUnavailable(Exception):
    """Fetch failed for a reason that may pass. Try again later."""


# Transcript errors that will never resolve on their own.
_PERMANENT_TRANSCRIPT_ERRORS: tuple[type[Exception], ...] = tuple(
    cls
    for name in (
        "TranscriptsDisabled",
        "NoTranscriptFound",
        "NotTranslatable",
        "AgeRestricted",
        "InvalidVideoId",
        "VideoUnavailable",
        "VideoUnplayable",
    )
    if (cls := getattr(yt_errors, name, None)) is not None
)


class ContentFetcher:
    """Turns one Article into body text, or explains why it cannot."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        transcript_api: Any | None = None,
    ) -> None:
        # Both injectable so tests never touch the network.
        self._client = client
        self._owns_client = client is None
        self._transcript_api = transcript_api

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=TIMEOUT,
                follow_redirects=True,
                # A User-Agent alone is not enough: openai.com answered 403
                # for 25 of 71 articles. Real browsers also send Accept,
                # Accept-Language and the Sec-Fetch-* set, and their absence
                # is an easy bot signal.
                headers=BROWSER_HEADERS,
                transport=httpx.HTTPTransport(retries=1),
            )
        return self._client

    @property
    def transcript_api(self) -> Any:
        if self._transcript_api is None:
            self._transcript_api = YouTubeTranscriptApi()
        return self._transcript_api

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def fetch(self, article: Article) -> str:
        """Return body text for `article`.

        Raises ContentUnavailable (never retry) or
        ContentTemporarilyUnavailable (retry later).
        """
        if article.source is Source.ARXIV:
            return self._from_abstract(article)
        if article.source is Source.YOUTUBE:
            return self._from_transcript(article)
        if article.source is Source.HACKERNEWS and "/item?id=" in article.url:
            # An Ask HN or text post. Its url is our fallback link to the
            # discussion page, and fetching that just rate-limits us (five
            # HTTP 419s) for text Algolia already gave us in `summary`.
            return self._from_summary(article)
        return self._from_html(article)

    # -- per-source strategies --------------------------------------------

    def _from_abstract(self, article: Article) -> str:
        """arXiv: the abstract IS the content.

        Parsing the PDF would add a heavy dependency for little gain - an
        abstract is already a dense, self-contained summary, which chunks
        well. Honest tradeoff rather than an oversight.
        """
        abstract = (article.summary or "").strip()
        if len(abstract) < MIN_CONTENT_CHARS:
            raise ContentUnavailable(
                f"abstract too short ({len(abstract)} chars)"
            )
        return abstract

    def _from_summary(self, article: Article) -> str:
        """Use text the feed already supplied, with no network call at all."""
        text = (article.summary or "").strip()
        if len(text) < MIN_CONTENT_CHARS:
            raise ContentUnavailable(f"summary too short ({len(text)} chars)")
        return text

    def _from_transcript(self, article: Article) -> str:
        """YouTube: the spoken transcript.

        Deliberately not done during ingestion - it is one network call per
        video and the part YouTube blocks from datacentre IPs, so a failure
        there would have cost the whole batch.
        """
        try:
            fetched = self.transcript_api.fetch(article.external_id)
        except _PERMANENT_TRANSCRIPT_ERRORS as exc:
            raise ContentUnavailable(f"{type(exc).__name__}") from exc
        except Exception as exc:
            # IpBlocked, RequestBlocked, PoTokenRequired, network trouble.
            raise ContentTemporarilyUnavailable(
                f"{type(exc).__name__}: {exc}"
            ) from exc

        text = " ".join(snippet.text for snippet in fetched.snippets).strip()
        if len(text) < MIN_CONTENT_CHARS:
            raise ContentUnavailable(f"transcript too short ({len(text)} chars)")
        return text

    def _from_html(self, article: Article) -> str:
        """Everything else: download the page and extract the article."""
        try:
            response = self.client.get(article.url)
        except httpx.HTTPError as exc:
            raise ContentTemporarilyUnavailable(
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code in PERMANENT_STATUSES:
            raise ContentUnavailable(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise ContentTemporarilyUnavailable(f"HTTP {response.status_code}")

        return extract_article_text(response.text, article.url)


def extract_article_text(html: str, url: str = "") -> str:
    """Pull the article out of a page, as markdown.

    trafilatura strips navigation, adverts, cookie banners, related-links
    rails and comment threads - the boilerplate that otherwise dominates a
    chunked page and makes every document look similar to every other one.

    favor_precision=True prefers losing a paragraph over keeping chrome,
    which is the right trade for retrieval: a clean 600-word article beats a
    2,000-word mixture of article and sidebar.
    """
    text = trafilatura.extract(
        html,
        url=url or None,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not text or len(text.strip()) < MIN_CONTENT_CHARS:
        length = len(text.strip()) if text else 0
        raise ContentUnavailable(f"extraction produced {length} chars")
    return text.strip()


def enrich_content(limit: int = 50, max_attempts: int = 3) -> dict[str, Any]:
    """Drain the content work queue. Returns per-source counts."""
    fetcher = ContentFetcher()
    stats: dict[str, Any] = {
        "attempted": 0,
        "fetched": 0,
        "permanent_failures": 0,
        "transient_failures": 0,
        "chars": 0,
        "by_source": {},
    }

    try:
        with session_scope() as session:
            repo = ArticleRepository(session)
            pending = repo.list_needing_content(limit=limit, max_attempts=max_attempts)
            log.info("enrich_started", pending=len(pending))

            for article in pending:
                key = article.source.value
                bucket = stats["by_source"].setdefault(
                    key, {"fetched": 0, "permanent": 0, "transient": 0}
                )
                stats["attempted"] += 1

                try:
                    text = fetcher.fetch(article)
                except ContentUnavailable as exc:
                    repo.mark_content_failed(
                        article.id, str(exc), permanent=True, max_attempts=max_attempts
                    )
                    stats["permanent_failures"] += 1
                    bucket["permanent"] += 1
                    log.info(
                        "content_unavailable",
                        source=key,
                        article_id=str(article.id),
                        reason=str(exc),
                    )
                except ContentTemporarilyUnavailable as exc:
                    repo.mark_content_failed(
                        article.id, str(exc), max_attempts=max_attempts
                    )
                    stats["transient_failures"] += 1
                    bucket["transient"] += 1
                    log.warning(
                        "content_fetch_retryable",
                        source=key,
                        article_id=str(article.id),
                        reason=str(exc),
                        attempts=article.content_attempts,
                    )
                else:
                    repo.set_content(article.id, text)
                    stats["fetched"] += 1
                    stats["chars"] += len(text)
                    bucket["fetched"] += 1
                    log.info(
                        "content_fetched",
                        source=key,
                        article_id=str(article.id),
                        chars=len(text),
                    )

                # Commit each article as we go. A crash or a rate limit
                # halfway through then keeps the bodies already fetched
                # instead of discarding the whole batch.
                session.commit()

            log.info("enrich_complete", **{k: v for k, v in stats.items() if k != "by_source"})
    finally:
        fetcher.close()

    return stats
