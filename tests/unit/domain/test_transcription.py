"""Tests for Transcription entity."""

from datetime import datetime

import pytest

from raywhisper.domain.entities.transcription import Transcription


def test_transcription_creation() -> None:
    """Test creating a valid transcription."""
    trans = Transcription(
        text="Hello world",
        language="en",
        confidence=0.95,
        timestamp=datetime.now(),
    )
    assert trans.text == "Hello world"
    assert trans.language == "en"
    assert trans.confidence == 0.95
    assert trans.context_used is None


def test_transcription_with_context() -> None:
    """Test creating a transcription with context."""
    trans = Transcription(
        text="Hello world",
        language="en",
        confidence=0.95,
        timestamp=datetime.now(),
        context_used="Some context",
    )
    assert trans.context_used == "Some context"


def test_transcription_invalid_confidence_too_high() -> None:
    """Test that confidence > 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        Transcription(
            text="Hello",
            language="en",
            confidence=1.5,
            timestamp=datetime.now(),
        )


def test_transcription_invalid_confidence_negative() -> None:
    """Test that negative confidence raises ValueError."""
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        Transcription(
            text="Hello",
            language="en",
            confidence=-0.1,
            timestamp=datetime.now(),
        )


def test_transcription_boundary_confidence_zero() -> None:
    """Test that confidence = 0.0 is valid."""
    trans = Transcription(
        text="Hello",
        language="en",
        confidence=0.0,
        timestamp=datetime.now(),
    )
    assert trans.confidence == 0.0


def test_transcription_boundary_confidence_one() -> None:
    """Test that confidence = 1.0 is valid."""
    trans = Transcription(
        text="Hello",
        language="en",
        confidence=1.0,
        timestamp=datetime.now(),
    )
    assert trans.confidence == 1.0

