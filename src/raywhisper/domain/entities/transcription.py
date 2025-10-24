"""Transcription entity."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Transcription:
    """Transcription entity representing a transcribed audio segment."""

    text: str
    language: str
    confidence: float
    timestamp: datetime
    context_used: str | None = None

    def __post_init__(self) -> None:
        """Validate transcription data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

