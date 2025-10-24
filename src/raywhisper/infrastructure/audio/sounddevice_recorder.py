"""SoundDevice-based audio recorder implementation."""

from queue import Queue
from threading import Event
from typing import Callable

import numpy as np
import sounddevice as sd

from ...domain.interfaces.audio_recorder import IAudioRecorder
from ...domain.value_objects.audio_data import AudioData


class SoundDeviceRecorder(IAudioRecorder):
    """Audio recorder implementation using sounddevice library."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        """Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_queue: Queue[np.ndarray] = Queue()
        self._recording_event = Event()
        self._stream: sd.InputStream | None = None

    def start_recording(self, callback: Callable[[AudioData], None] | None = None) -> None:
        """Start recording audio.

        Args:
            callback: Optional callback function to receive audio chunks during recording.
        """
        if self._recording_event.is_set():
            raise RuntimeError("Recording is already in progress")

        self._recording_event.set()
        self._audio_queue = Queue()

        def audio_callback(
            indata: np.ndarray,
            frames: int,
            time_info: sd.CallbackFlags,
            status: sd.CallbackFlags,
        ) -> None:
            """Callback for audio stream."""
            if status:
                print(f"Audio callback status: {status}")

            if self._recording_event.is_set():
                self._audio_queue.put(indata.copy())

                if callback:
                    audio_chunk = AudioData(
                        samples=indata.copy().flatten(),
                        sample_rate=self._sample_rate,
                        channels=self._channels,
                    )
                    callback(audio_chunk)

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            callback=audio_callback,
            dtype=np.float32,
        )
        self._stream.start()

    def stop_recording(self) -> AudioData:
        """Stop recording and return complete audio data.

        Returns:
            AudioData: The recorded audio data.
        """
        if not self._recording_event.is_set():
            raise RuntimeError("No recording in progress")

        self._recording_event.clear()

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Collect all audio chunks
        chunks: list[np.ndarray] = []
        while not self._audio_queue.empty():
            chunks.append(self._audio_queue.get())

        if chunks:
            samples = np.concatenate(chunks, axis=0)
            return AudioData(
                samples=samples.flatten(),
                sample_rate=self._sample_rate,
                channels=self._channels,
            )

        # Return empty audio if no chunks were recorded
        return AudioData(
            samples=np.array([], dtype=np.float32),
            sample_rate=self._sample_rate,
            channels=self._channels,
        )

    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        return self._recording_event.is_set()

