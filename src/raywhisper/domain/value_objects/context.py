"""Context value object for search results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """Immutable search result value object."""

    content: str
    score: float
    metadata: dict[str, str]

