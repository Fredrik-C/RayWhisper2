"""Transcriber interface."""

from abc import ABC, abstractmethod

from ..entities.transcription import Transcription
from ..value_objects.audio_data import AudioData


class ITranscriber(ABC):
    """Interface for audio transcription."""

    @abstractmethod
    def transcribe(self, audio: AudioData, context: str | None = None) -> Transcription:
        """Transcribe audio to text, optionally using context for guidance.

        Args:
            audio: The audio data to transcribe.
            context: Optional context to guide transcription.

        Returns:
            Transcription: The transcription result.
        """
        pass

