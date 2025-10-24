"""Tests for AudioData value object."""

import numpy as np

from raywhisper.domain.value_objects.audio_data import AudioData


def test_audio_data_creation() -> None:
    """Test creating audio data."""
    samples = np.zeros(16000, dtype=np.float32)
    audio = AudioData(samples=samples, sample_rate=16000, channels=1)

    assert len(audio.samples) == 16000
    assert audio.sample_rate == 16000
    assert audio.channels == 1


def test_audio_data_duration() -> None:
    """Test audio duration calculation."""
    # 1 second of audio at 16kHz
    samples = np.zeros(16000, dtype=np.float32)
    audio = AudioData(samples=samples, sample_rate=16000, channels=1)

    assert audio.duration == 1.0


def test_audio_data_duration_half_second() -> None:
    """Test audio duration calculation for 0.5 seconds."""
    # 0.5 seconds of audio at 16kHz
    samples = np.zeros(8000, dtype=np.float32)
    audio = AudioData(samples=samples, sample_rate=16000, channels=1)

    assert audio.duration == 0.5


def test_audio_data_to_wav_bytes() -> None:
    """Test converting audio data to WAV bytes."""
    samples = np.zeros(16000, dtype=np.float32)
    audio = AudioData(samples=samples, sample_rate=16000, channels=1)

    wav_bytes = audio.to_wav_bytes()

    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0
    # WAV header should start with 'RIFF'
    assert wav_bytes[:4] == b"RIFF"


def test_audio_data_immutability() -> None:
    """Test that AudioData is immutable (frozen dataclass)."""
    samples = np.zeros(16000, dtype=np.float32)
    audio = AudioData(samples=samples, sample_rate=16000, channels=1)

    # Attempting to modify should raise an error
    try:
        audio.sample_rate = 48000  # type: ignore
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass  # Expected

