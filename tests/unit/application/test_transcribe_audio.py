"""Tests for TranscribeAudio use case."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from raywhisper.application.use_cases.transcribe_audio import TranscribeAudioUseCase
from raywhisper.domain.entities.transcription import Transcription
from raywhisper.domain.value_objects.audio_data import AudioData
from raywhisper.domain.value_objects.context import SearchResult


@pytest.fixture
def mock_transcriber() -> MagicMock:
    """Create a mock transcriber."""
    transcriber = MagicMock()
    transcriber.transcribe.return_value = Transcription(
        text="Test transcription",
        language="en",
        confidence=0.95,
        timestamp=datetime.now(),
    )
    return transcriber


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock vector store."""
    vector_store = MagicMock()
    vector_store.search.return_value = [
        SearchResult(
            content="Context 1",
            score=0.9,
            metadata={"source": "test1.md"},
        ),
        SearchResult(
            content="Context 2",
            score=0.8,
            metadata={"source": "test2.md"},
        ),
    ]
    return vector_store


@pytest.fixture
def sample_audio() -> AudioData:
    """Create sample audio data."""
    samples = np.zeros(16000, dtype=np.float32)
    return AudioData(samples=samples, sample_rate=16000, channels=1)


def test_transcribe_without_rag(mock_transcriber: MagicMock, sample_audio: AudioData) -> None:
    """Test transcription without RAG."""
    use_case = TranscribeAudioUseCase(transcriber=mock_transcriber)

    result = use_case.execute(sample_audio, use_rag=False)

    assert result.text == "Test transcription"
    mock_transcriber.transcribe.assert_called_once_with(sample_audio)


def test_transcribe_with_rag_no_vector_store(
    mock_transcriber: MagicMock, sample_audio: AudioData
) -> None:
    """Test transcription with RAG enabled but no vector store."""
    use_case = TranscribeAudioUseCase(transcriber=mock_transcriber, vector_store=None)

    result = use_case.execute(sample_audio, use_rag=True)

    assert result.text == "Test transcription"
    mock_transcriber.transcribe.assert_called_once()


def test_transcribe_with_rag_and_vector_store(
    mock_transcriber: MagicMock, mock_vector_store: MagicMock, sample_audio: AudioData
) -> None:
    """Test transcription with RAG and vector store."""
    use_case = TranscribeAudioUseCase(
        transcriber=mock_transcriber, vector_store=mock_vector_store
    )

    # Set up mock to return different results for initial and final transcription
    initial_trans = Transcription(
        text="Initial transcription",
        language="en",
        confidence=0.9,
        timestamp=datetime.now(),
    )
    final_trans = Transcription(
        text="Final transcription with context",
        language="en",
        confidence=0.95,
        timestamp=datetime.now(),
        context_used="Context 1\nContext 2",
    )
    mock_transcriber.transcribe.side_effect = [initial_trans, final_trans]

    result = use_case.execute(sample_audio, use_rag=True)

    # Should call transcribe twice: once for initial, once with context
    assert mock_transcriber.transcribe.call_count == 2
    # Should search vector store
    mock_vector_store.search.assert_called_once_with(query="Initial transcription", top_k=5)
    # Should return final transcription
    assert result.text == "Final transcription with context"


def test_transcribe_with_rag_empty_initial_transcription(
    mock_transcriber: MagicMock, mock_vector_store: MagicMock, sample_audio: AudioData
) -> None:
    """Test transcription with RAG when initial transcription is empty."""
    use_case = TranscribeAudioUseCase(
        transcriber=mock_transcriber, vector_store=mock_vector_store
    )

    # Set up mock to return empty initial transcription
    empty_trans = Transcription(
        text="",
        language="en",
        confidence=0.0,
        timestamp=datetime.now(),
    )
    mock_transcriber.transcribe.return_value = empty_trans

    result = use_case.execute(sample_audio, use_rag=True)

    # Should not search vector store if initial transcription is empty
    mock_vector_store.search.assert_not_called()
    assert result.text == ""


def test_transcribe_with_rag_no_search_results(
    mock_transcriber: MagicMock, mock_vector_store: MagicMock, sample_audio: AudioData
) -> None:
    """Test transcription with RAG when vector store returns no results."""
    use_case = TranscribeAudioUseCase(
        transcriber=mock_transcriber, vector_store=mock_vector_store
    )

    # Set up mock to return no search results
    mock_vector_store.search.return_value = []

    initial_trans = Transcription(
        text="Initial transcription",
        language="en",
        confidence=0.9,
        timestamp=datetime.now(),
    )
    mock_transcriber.transcribe.return_value = initial_trans

    result = use_case.execute(sample_audio, use_rag=True)

    # Should search but not find anything
    mock_vector_store.search.assert_called_once()
    # Should still return a transcription
    assert result.text == "Initial transcription"

