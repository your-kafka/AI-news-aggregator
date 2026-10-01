"""The source registry.

A source registers itself with a decorator:

    @register
    class OpenAISource(RssSource):
        name = Source.OPENAI

The runner then asks for every registered source and calls fetch() on each.
It never imports a source class or knows a source's name, so adding arXiv is
adding a file - no edit to the runner, which is the thing the reference
project cannot do.
"""

from __future__ import annotations

from lodestar.ingestion.base import BaseSource
from lodestar.storage.models import Source

_REGISTRY: dict[Source, type[BaseSource]] = {}


def register(cls: type[BaseSource]) -> type[BaseSource]:
    """Class decorator that adds a source to the registry."""
    if not hasattr(cls, "name"):
        raise TypeError(f"{cls.__name__} must set a class-level `name`")
    if cls.name in _REGISTRY:
        raise ValueError(
            f"{cls.name} already registered by {_REGISTRY[cls.name].__name__}"
        )
    _REGISTRY[cls.name] = cls
    return cls


def registered_names() -> list[Source]:
    return sorted(_REGISTRY, key=lambda s: s.value)


def get_source_class(name: Source) -> type[BaseSource]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no source registered for {name!r}; have {registered_names()}"
        ) from None


def build_all() -> list[BaseSource]:
    """One instance of every registered source."""
    return [cls() for cls in (_REGISTRY[n] for n in registered_names())]


def clear_registry() -> None:
    """Empty the registry. For tests only."""
    _REGISTRY.clear()
