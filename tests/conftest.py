"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_audio_data():
    """Fixture providing sample audio data for testing."""
    import numpy as np
    from raywhisper.domain.value_objects.audio_data import AudioData

    # Create 1 second of silence at 16kHz
    samples = np.zeros(16000, dtype=np.float32)
    return AudioData(samples=samples, sample_rate=16000, channels=1)


@pytest.fixture
def sample_transcription():
    """Fixture providing sample transcription for testing."""
    from datetime import datetime
    from raywhisper.domain.entities.transcription import Transcription

    return Transcription(
        text="Hello world",
        language="en",
        confidence=0.95,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_document():
    """Fixture providing sample document for testing."""
    from pathlib import Path
    from raywhisper.domain.entities.document import Document

    return Document(
        content="This is a test document.",
        source_path=Path("test.md"),
        metadata={"type": "markdown"},
    )

