"""Whisper-based transcriber implementation."""

from datetime import datetime
from typing import Literal

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from ...domain.entities.transcription import Transcription
from ...domain.interfaces.transcriber import ITranscriber
from ...domain.value_objects.audio_data import AudioData


class WhisperTranscriber(ITranscriber):
    """Transcriber implementation using faster-whisper."""

    def __init__(
        self,
        model_size: str,
        device: Literal["cpu", "cuda", "auto"],
        compute_type: Literal["int8", "int8_float16", "float16", "float32"],
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        condition_on_previous_text: bool = False,
        compression_ratio_threshold: float | None = 2.4,
        log_prob_threshold: float | None = -1.0,
        no_speech_threshold: float | None = 0.6,
        repetition_penalty: float = 1.0,
    ) -> None:
        """Initialize the Whisper transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3, distil-large-v3)
                       or HuggingFace model ID (e.g., Systran/faster-whisper-medium.en).
            device: Device to run model on (cpu, cuda, auto).
            compute_type: CTranslate2 compute type for quantization.
            beam_size: Beam size for beam search (higher = more accurate but slower).
            best_of: Number of candidates to generate (higher = better quality).
            temperature: Sampling temperature (0.0 = deterministic).
            condition_on_previous_text: Use previous text as context.
            compression_ratio_threshold: Detect repetitions (lower = more strict).
            log_prob_threshold: Filter low-confidence segments (higher = more strict).
            no_speech_threshold: Detect silence (higher = more strict).
            repetition_penalty: Penalize repetitions (1.0 = no penalty).
        """
        logger.info(
            f"Loading Whisper model: {model_size} on {device} with compute_type={compute_type}"
        )

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        # Store transcription parameters
        self._beam_size = beam_size
        self._best_of = best_of
        self._temperature = temperature
        self._condition_on_previous_text = condition_on_previous_text
        self._compression_ratio_threshold = compression_ratio_threshold
        self._log_prob_threshold = log_prob_threshold
        self._no_speech_threshold = no_speech_threshold
        self._repetition_penalty = repetition_penalty

        logger.info(
            f"Whisper model loaded successfully "
            f"(beam_size={beam_size}, temperature={temperature}, "
            f"condition_on_previous_text={condition_on_previous_text})"
        )

    def transcribe(self, audio: AudioData, context: str | None = None) -> Transcription:
        """Transcribe audio to text.

        Args:
            audio: The audio data to transcribe.
            context: Optional context to guide transcription.

        Returns:
            Transcription: The transcription result.
        """
        if audio.duration < 0.1:
            logger.warning(f"Audio too short ({audio.duration:.2f}s), returning empty transcription")
            return Transcription(
                text="",
                language="en",
                confidence=0.0,
                timestamp=datetime.now(),
                context_used=context,
            )

        # Prepare audio (faster-whisper expects float32 normalized to [-1, 1])
        audio_float = audio.samples.astype(np.float32)

        # Normalize if needed
        if audio_float.max() > 1.0:
            audio_float = audio_float / 32768.0  # Normalize int16 to float32

        logger.debug(
            f"Transcribing {audio.duration:.2f}s of audio"
            + (f" with context ({len(context)} chars)" if context else "")
        )

        # Transcribe with optional context
        # Note: consider BatchedInferencePipeline for throughput or distil-large-v3 for speed
        segments, info = self._model.transcribe(
            audio_float,
            initial_prompt=context,
            beam_size=self._beam_size,
            best_of=self._best_of,
            temperature=self._temperature,
            condition_on_previous_text=self._condition_on_previous_text,
            compression_ratio_threshold=self._compression_ratio_threshold,
            log_prob_threshold=self._log_prob_threshold,
            no_speech_threshold=self._no_speech_threshold,
            repetition_penalty=self._repetition_penalty,
            vad_filter=True,  # Voice activity detection
        )

        # Combine all segments
        segment_list = list(segments)
        text = " ".join([segment.text for segment in segment_list])

        # Calculate average confidence from log probabilities
        if segment_list:
            # avg_logprob is typically negative; convert to a 0-1 scale
            # Note: This is a rough approximation
            avg_logprob = sum(s.avg_logprob for s in segment_list) / len(segment_list)
            # Convert log probability to approximate confidence
            # avg_logprob typically ranges from -1 to 0 for good transcriptions
            confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
        else:
            confidence = 0.0

        logger.debug(
            f"Transcription complete: '{text[:50]}...' "
            f"(language={info.language}, confidence={confidence:.2f})"
        )

        return Transcription(
            text=text.strip(),
            language=info.language,
            confidence=confidence,
            timestamp=datetime.now(),
            context_used=context,
        )

