"""Audio data value object."""

import io
import wave
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioData:
    """Immutable audio data value object."""

    samples: np.ndarray
    sample_rate: int
    channels: int

    @property
    def duration(self) -> float:
        """Duration in seconds.

        Returns:
            float: Duration of the audio in seconds.
        """
        return len(self.samples) / self.sample_rate

    def to_wav_bytes(self) -> bytes:
        """Convert to WAV format bytes.

        Returns:
            bytes: Audio data in WAV format.
        """
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            # Convert float32 to int16 if needed
            if self.samples.dtype == np.float32:
                audio_int16 = (self.samples * 32767).astype(np.int16)
            else:
                audio_int16 = self.samples.astype(np.int16)

            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

