"""Small audio feedback helpers used to play start/stop tones.

The functions are intentionally simple and blocking (short tones) so they
can give immediate audible feedback before/after recording.
"""

from __future__ import annotations

from loguru import logger
import numpy as np
import sounddevice as sd


def _play_tone(frequency: float, duration: float, volume: float = 0.12, sample_rate: int = 44100) -> None:
    """Play a short sine-wave tone. Silently fails if audio stack isn't available.

    Args:
        frequency: Frequency in Hz.
        duration: Duration in seconds.
        volume: Linear volume (0.0-1.0).
        sample_rate: Sampling rate in Hz.
    """
    try:
        samples = int(sample_rate * duration)
        if samples < 1:
            return
        t = np.linspace(0, duration, samples, False)
        wave = np.sin(2 * np.pi * frequency * t)

        # Apply short fade-in/fade-out to avoid clicks
        fade_len = min(int(0.01 * sample_rate), samples // 2)
        if fade_len > 0:
            fade_in = np.linspace(0, 1.0, fade_len)
            fade_out = np.linspace(1.0, 0, fade_len)
            wave[:fade_len] *= fade_in
            wave[-fade_len:] *= fade_out

        audio = (wave * volume).astype(np.float32)
        sd.play(audio, samplerate=sample_rate, blocking=True)
    except Exception as e:
        # Audio is optional; log debug and continue
        logger.debug(f"Audio feedback not available: {e}")


def play_start_sound() -> None:
    """Play a short start tone (higher pitch)."""
    _play_tone(frequency=880.0, duration=0.08, volume=0.10)


def play_stop_sound() -> None:
    """Play a short stop tone (lower pitch)."""
    _play_tone(frequency=600.0, duration=0.12, volume=0.10)
