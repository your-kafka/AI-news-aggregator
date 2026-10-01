"""Tests for lodestar.core.config.

Two things these tests must get right:

1. ISOLATION. They never read the developer's real .env file. CI has no
   .env at all, so a test that quietly depends on yours passes on your
   laptop and fails in GitHub Actions. Every test builds its own
   environment from nothing.

2. THEY MUST BE ABLE TO FAIL. Half of this file asserts that BAD config
   is rejected. A validation test that cannot fail is worse than no test,
   because it reports success either way.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from lodestar.core.config import Settings, get_settings

# Any variable starting with these is wiped before each test, so a value
# left in the developer's shell can never influence the result.
_OWNED_PREFIXES = ("POSTGRES_", "LODESTAR_")

SettingsBuilder = Callable[..., Settings]


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch) -> SettingsBuilder:
    """Return a function that builds Settings from an explicit environment.

    _env_file=None is the important part: it tells pydantic-settings to
    ignore .env entirely, so these tests behave identically on a laptop
    and on a CI runner.
    """

    def _build(**values: str) -> Settings:
        for key in list(os.environ):
            if key.startswith(_OWNED_PREFIXES):
                monkeypatch.delenv(key, raising=False)
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        return Settings(_env_file=None)

    return _build


# ---------------------------------------------------------------------------
#  Happy path
# ---------------------------------------------------------------------------


def test_defaults_apply_when_only_password_is_given(build: SettingsBuilder) -> None:
    settings = build(POSTGRES_PASSWORD="pw")

    assert settings.postgres_host == "localhost"
    assert settings.postgres_port == 5432
    assert settings.postgres_user == "lodestar"
    assert settings.postgres_db == "lodestar"
    assert settings.env == "local"
    assert settings.log_level == "INFO"


def test_environment_values_override_defaults(build: SettingsBuilder) -> None:
    settings = build(
        POSTGRES_PASSWORD="pw",
        POSTGRES_HOST="db.example.com",
        POSTGRES_USER="someone",
        POSTGRES_DB="otherdb",
        LODESTAR_ENV="prod",
        LODESTAR_LOG_LEVEL="WARNING",
    )

    assert settings.postgres_host == "db.example.com"
    assert settings.postgres_user == "someone"
    assert settings.postgres_db == "otherdb"
    assert settings.env == "prod"
    assert settings.log_level == "WARNING"


def test_lowercase_env_var_names_are_accepted(build: SettingsBuilder) -> None:
    """case_sensitive=False means postgres_host matches POSTGRES_HOST."""
    settings = build(POSTGRES_PASSWORD="pw", postgres_host="lower.example.com")

    assert settings.postgres_host == "lower.example.com"


# ---------------------------------------------------------------------------
#  Type conversion - environment variables are ALWAYS strings
# ---------------------------------------------------------------------------


def test_port_arrives_as_an_int_not_a_string(build: SettingsBuilder) -> None:
    settings = build(POSTGRES_PASSWORD="pw", POSTGRES_PORT="6543")

    assert settings.postgres_port == 6543
    assert isinstance(settings.postgres_port, int)


# ---------------------------------------------------------------------------
#  Rejection - the half of the file that must be able to fail
# ---------------------------------------------------------------------------


def test_missing_password_refuses_to_start(build: SettingsBuilder) -> None:
    """No default for postgres_password, so a missing one is fatal.

    This mirrors ${POSTGRES_PASSWORD:?...} in docker-compose.yml: never
    boot with a guessable password, fail at startup instead.
    """
    with pytest.raises(ValidationError) as exc:
        build(POSTGRES_HOST="localhost")

    assert "postgres_password" in str(exc.value)


@pytest.mark.parametrize("bad_port", ["abc", "", "5432.5", "0", "-1", "70000"])
def test_invalid_ports_are_rejected(build: SettingsBuilder, bad_port: str) -> None:
    """Not a number, or outside the valid 1-65535 TCP range."""
    with pytest.raises(ValidationError):
        build(POSTGRES_PASSWORD="pw", POSTGRES_PORT=bad_port)


@pytest.mark.parametrize("bad_env", ["production", "PROD", "dev", "staging", ""])
def test_invalid_env_values_are_rejected(build: SettingsBuilder, bad_env: str) -> None:
    """Literal["local", "prod"] accepts nothing else.

    'production' is the realistic typo: without this, the app would
    silently take the not-local branch forever.
    """
    with pytest.raises(ValidationError):
        build(POSTGRES_PASSWORD="pw", LODESTAR_ENV=bad_env)


@pytest.mark.parametrize("bad_level", ["VERBOSE", "TRACE", "info", "CRITICAL"])
def test_invalid_log_levels_are_rejected(build: SettingsBuilder, bad_level: str) -> None:
    with pytest.raises(ValidationError):
        build(POSTGRES_PASSWORD="pw", LODESTAR_LOG_LEVEL=bad_level)


# ---------------------------------------------------------------------------
#  Secret handling
# ---------------------------------------------------------------------------


def test_password_is_masked_in_str_and_repr(build: SettingsBuilder) -> None:
    """The commonest way credentials leak is a log line or a traceback
    that printed a config object. SecretStr makes that impossible."""
    secret = "super-secret-value"
    settings = build(POSTGRES_PASSWORD=secret)

    assert secret not in str(settings)
    assert secret not in repr(settings)
    assert secret not in str(settings.postgres_password)
    # ...but it is still readable when explicitly asked for.
    assert settings.postgres_password.get_secret_value() == secret


def test_safe_database_url_hides_the_password(build: SettingsBuilder) -> None:
    secret = "super-secret-value"
    settings = build(POSTGRES_PASSWORD=secret)

    assert secret not in settings.safe_database_url
    assert "***" in settings.safe_database_url


# ---------------------------------------------------------------------------
#  The connection string
# ---------------------------------------------------------------------------


def test_database_url_is_assembled_from_the_parts(build: SettingsBuilder) -> None:
    settings = build(
        POSTGRES_PASSWORD="pw",
        POSTGRES_USER="someone",
        POSTGRES_HOST="db.example.com",
        POSTGRES_PORT="6543",
        POSTGRES_DB="otherdb",
    )

    assert settings.database_url == (
        "postgresql+psycopg://someone:pw@db.example.com:6543/otherdb"
    )


def test_password_with_url_special_characters_still_parses(build: SettingsBuilder) -> None:
    """Regression guard for a real and painful bug.

    '@', '/' and ':' are structural characters in a URL. An unescaped
    password containing them silently produces a URL pointing at the
    wrong host. quote_plus escapes them; this proves the value survives
    the round trip.
    """
    nasty = "p@ss/w:rd#1"
    settings = build(POSTGRES_PASSWORD=nasty, POSTGRES_HOST="realhost")

    url = make_url(settings.database_url)
    assert url.password == nasty
    assert url.host == "realhost"
    assert url.database == "lodestar"


# ---------------------------------------------------------------------------
#  Derived values and caching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("local", False), ("prod", True)],
)
def test_is_production_reflects_env(
    build: SettingsBuilder, env_value: str, expected: bool
) -> None:
    settings = build(POSTGRES_PASSWORD="pw", LODESTAR_ENV=env_value)

    assert settings.is_production is expected


def test_get_settings_returns_the_same_object_every_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """@lru_cache makes get_settings a singleton, so .env is read once and
    configuration cannot drift between modules."""
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        # Never leave a cached Settings behind for the next test.
        get_settings.cache_clear()
