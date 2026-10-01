"""Tests for ArticleIn - the validation boundary. No database needed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from lodestar.domain.article import ArticleIn
from lodestar.storage.models import Source


def valid(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "source": Source.OPENAI,
        "external_id": "post-1",
        "title": "A real title",
        "url": "https://openai.com/index/something",
        "published_at": datetime.now(UTC),
    }
    data.update(overrides)
    return data


def test_minimal_valid_article() -> None:
    article = ArticleIn(**valid())  # type: ignore[arg-type]

    assert article.source is Source.OPENAI
    assert article.author is None
    assert article.content is None
    assert article.meta == {}


def test_whitespace_is_stripped() -> None:
    """RSS titles very often arrive wrapped in newlines and indentation."""
    article = ArticleIn(**valid(title="\n  Padded title  \n"))  # type: ignore[arg-type]

    assert article.title == "Padded title"


def test_naive_datetime_is_rejected() -> None:
    """The column is TIMESTAMPTZ; a naive value would make Postgres guess."""
    with pytest.raises(ValidationError) as exc:
        ArticleIn(**valid(published_at=datetime(2026, 1, 1, 12, 0)))  # type: ignore[arg-type]

    assert "timezone-aware" in str(exc.value)


def test_non_utc_timezone_is_accepted() -> None:
    """Feeds publish in local time. We keep the instant, not the offset."""
    ist = timezone(timedelta(hours=5, minutes=30))
    article = ArticleIn(**valid(published_at=datetime(2026, 1, 1, 12, 0, tzinfo=ist)))  # type: ignore[arg-type]

    assert article.published_at.utcoffset() == timedelta(hours=5, minutes=30)


@pytest.mark.parametrize("bad_url", ["example.com", "ftp://x.com/a", "", "javascript:x"])
def test_non_http_urls_are_rejected(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        ArticleIn(**valid(url=bad_url))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_title", ["", "   ", "\n"])
def test_empty_titles_are_rejected(bad_title: str) -> None:
    """Whitespace is stripped first, so '   ' becomes '' and fails min_length."""
    with pytest.raises(ValidationError):
        ArticleIn(**valid(title=bad_title))  # type: ignore[arg-type]


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ArticleIn(**valid(source="medium"))  # type: ignore[arg-type]


def test_misspelled_field_is_rejected_not_ignored() -> None:
    """extra='forbid' turns a scraper typo into an error instead of a NULL."""
    with pytest.raises(ValidationError) as exc:
        ArticleIn(**valid(tittle="oops"))  # type: ignore[arg-type]

    assert "tittle" in str(exc.value)


def test_to_row_is_insertable() -> None:
    row = ArticleIn(**valid()).to_row()  # type: ignore[arg-type]

    assert set(row) == {
        "source", "external_id", "title", "url", "published_at",
        "author", "summary", "content", "meta",
    }
    assert "id" not in row  # the repository generates ids
