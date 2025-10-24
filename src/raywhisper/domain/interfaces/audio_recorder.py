"""Audio recorder interface."""

from abc import ABC, abstractmethod
from typing import Callable

from ..value_objects.audio_data import AudioData


class IAudioRecorder(ABC):
    """Interface for audio recording."""

    @abstractmethod
    def start_recording(self, callback: Callable[[AudioData], None] | None = None) -> None:
        """Start recording audio and optionally call callback with audio chunks.

        Args:
            callback: Optional callback function to receive audio chunks during recording.
        """
        pass

    @abstractmethod
    def stop_recording(self) -> AudioData:
        """Stop recording and return complete audio data.

        Returns:
            AudioData: The recorded audio data.
        """
        pass

    @abstractmethod
    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        pass

