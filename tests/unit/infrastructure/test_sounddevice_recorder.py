"""Tests for SoundDeviceRecorder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from raywhisper.infrastructure.audio.sounddevice_recorder import SoundDeviceRecorder


@pytest.fixture
def recorder() -> SoundDeviceRecorder:
    """Create a SoundDeviceRecorder instance for testing."""
    return SoundDeviceRecorder(sample_rate=16000, channels=1)


def test_recorder_initialization(recorder: SoundDeviceRecorder) -> None:
    """Test recorder initialization."""
    assert recorder._sample_rate == 16000
    assert recorder._channels == 1
    assert not recorder.is_recording()


@patch("raywhisper.infrastructure.audio.sounddevice_recorder.sd.InputStream")
def test_start_recording(mock_input_stream: MagicMock, recorder: SoundDeviceRecorder) -> None:
    """Test starting recording."""
    mock_stream = MagicMock()
    mock_input_stream.return_value = mock_stream

    recorder.start_recording()

    assert recorder.is_recording()
    mock_input_stream.assert_called_once()
    mock_stream.start.assert_called_once()


@patch("raywhisper.infrastructure.audio.sounddevice_recorder.sd.InputStream")
def test_start_recording_twice_raises_error(
    mock_input_stream: MagicMock, recorder: SoundDeviceRecorder
) -> None:
    """Test that starting recording twice raises an error."""
    mock_stream = MagicMock()
    mock_input_stream.return_value = mock_stream

    recorder.start_recording()

    with pytest.raises(RuntimeError, match="Recording is already in progress"):
        recorder.start_recording()


@patch("raywhisper.infrastructure.audio.sounddevice_recorder.sd.InputStream")
def test_stop_recording(mock_input_stream: MagicMock, recorder: SoundDeviceRecorder) -> None:
    """Test stopping recording."""
    mock_stream = MagicMock()
    mock_input_stream.return_value = mock_stream

    recorder.start_recording()

    # Simulate some audio data
    test_data = np.zeros((100, 1), dtype=np.float32)
    recorder._audio_queue.put(test_data)

    audio_data = recorder.stop_recording()

    assert not recorder.is_recording()
    assert audio_data.sample_rate == 16000
    assert audio_data.channels == 1
    assert len(audio_data.samples) == 100
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()


def test_stop_recording_without_start_raises_error(recorder: SoundDeviceRecorder) -> None:
    """Test that stopping recording without starting raises an error."""
    with pytest.raises(RuntimeError, match="No recording in progress"):
        recorder.stop_recording()


@patch("raywhisper.infrastructure.audio.sounddevice_recorder.sd.InputStream")
def test_stop_recording_empty_audio(
    mock_input_stream: MagicMock, recorder: SoundDeviceRecorder
) -> None:
    """Test stopping recording with no audio data."""
    mock_stream = MagicMock()
    mock_input_stream.return_value = mock_stream

    recorder.start_recording()
    audio_data = recorder.stop_recording()

    assert len(audio_data.samples) == 0
    assert audio_data.sample_rate == 16000
    assert audio_data.channels == 1


@patch("raywhisper.infrastructure.audio.sounddevice_recorder.sd.InputStream")
def test_recording_with_callback(
    mock_input_stream: MagicMock, recorder: SoundDeviceRecorder
) -> None:
    """Test recording with callback function."""
    mock_stream = MagicMock()
    mock_input_stream.return_value = mock_stream

    callback_called = []

    def test_callback(audio_data):  # type: ignore
        callback_called.append(audio_data)

    recorder.start_recording(callback=test_callback)

    # Get the audio callback that was registered
    call_args = mock_input_stream.call_args
    audio_callback = call_args.kwargs["callback"]

    # Simulate audio data coming in
    test_data = np.zeros((100, 1), dtype=np.float32)
    audio_callback(test_data, 100, None, None)

    # Verify callback was called
    assert len(callback_called) == 1
    assert callback_called[0].sample_rate == 16000


@patch("raywhisper.infrastructure.audio.sounddevice_recorder.sd.InputStream")
def test_multiple_chunks_concatenation(
    mock_input_stream: MagicMock, recorder: SoundDeviceRecorder
) -> None:
    """Test that multiple audio chunks are concatenated correctly."""
    mock_stream = MagicMock()
    mock_input_stream.return_value = mock_stream

    recorder.start_recording()

    # Simulate multiple chunks
    chunk1 = np.ones((100, 1), dtype=np.float32)
    chunk2 = np.ones((100, 1), dtype=np.float32) * 2
    chunk3 = np.ones((100, 1), dtype=np.float32) * 3

    recorder._audio_queue.put(chunk1)
    recorder._audio_queue.put(chunk2)
    recorder._audio_queue.put(chunk3)

    audio_data = recorder.stop_recording()

    assert len(audio_data.samples) == 300
    # Verify the chunks were concatenated in order
    assert np.allclose(audio_data.samples[:100], 1.0)
    assert np.allclose(audio_data.samples[100:200], 2.0)
    assert np.allclose(audio_data.samples[200:300], 3.0)

