"""Application settings.

Every configurable value in Lodestar lives here, in one typed object.
Nothing else in the codebase should call os.getenv().

Values are read once, at startup, from (in order of priority):
    1. real environment variables   (how production supplies them)
    2. the .env file                (how your laptop supplies them)
    3. the defaults written below

If anything required is missing or the wrong type, the app refuses to
start and says exactly what is wrong. That is deliberate: a crash on
boot is a two-minute fix, a wrong value discovered at 3am is not.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the app, validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",          # read this file if it exists
        env_file_encoding="utf-8",
        case_sensitive=False,     # POSTGRES_HOST matches postgres_host
        extra="ignore",           # ignore env vars we don't declare
    )

    # -----------------------------------------------------------------
    #  App
    #
    #  Literal[...] means "only these exact strings are allowed".
    #  LODESTAR_ENV=production would be REJECTED at startup, naming the
    #  field and listing the valid options - rather than silently being
    #  treated as "not local" somewhere deep in the code.
    #
    #  validation_alias lets the Python name differ from the env name:
    #  we write settings.env, but it reads LODESTAR_ENV.
    # -----------------------------------------------------------------
    env: Literal["local", "prod"] = Field(
        default="local",
        validation_alias="LODESTAR_ENV",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        validation_alias="LODESTAR_LOG_LEVEL",
    )

    # -----------------------------------------------------------------
    #  Postgres
    #
    #  postgres_password has NO default. That makes it required: if it
    #  is missing the app will not start. Same idea as ${VAR:?msg} in
    #  docker-compose.yml - never boot with a guessable password.
    #
    #  SecretStr means the value is masked as '**********' anywhere it
    #  is printed, logged, or shown in a traceback. Reading it requires
    #  an explicit .get_secret_value() call.
    # -----------------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = "lodestar"
    postgres_password: SecretStr
    postgres_db: str = "lodestar"

    # -----------------------------------------------------------------
    #  Derived values - computed from the fields above, never stored.
    # -----------------------------------------------------------------

    @property
    def is_production(self) -> bool:
        """True in prod. Use this to switch behaviour, never a raw string compare."""
        return self.env == "prod"

    @property
    def database_url(self) -> str:
        """SQLAlchemy connection string. Contains the real password.

        quote_plus escapes the password because a password containing
        '@', '/' or ':' would otherwise break URL parsing - a real bug
        that is genuinely painful to diagnose.
        """
        password = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql+psycopg://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def safe_database_url(self) -> str:
        """Same string with the password masked. Use this in logs."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:***"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return the one shared Settings object.

    @lru_cache makes this a singleton: .env is read and validated on the
    first call only, and every caller afterwards gets the same object,
    so configuration cannot drift between modules.

    In tests, call get_settings.cache_clear() to force a fresh read.
    """
    # No "type: ignore" needed here. The pydantic.mypy plugin we enabled
    # in pyproject.toml teaches mypy that BaseSettings fills its required
    # fields from the environment, so Settings() with no arguments is
    # correctly understood. Without that plugin this line would error.
    return Settings()
