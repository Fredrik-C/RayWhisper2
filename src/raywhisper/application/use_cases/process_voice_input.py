"""Process voice input use case."""

from typing import Callable

from loguru import logger

from ...domain.interfaces.audio_recorder import IAudioRecorder
from ...domain.interfaces.output_simulator import IOutputSimulator
from ...domain.interfaces.transcriber import ITranscriber
from ..services.rag_service import RAGService


class ProcessVoiceInputUseCase:
    """Use case for processing voice input end-to-end."""

    def __init__(
        self,
        audio_recorder: IAudioRecorder,
        transcriber: ITranscriber,
        rag_service: RAGService | None,
        output_simulator: IOutputSimulator,
        on_output_start: Callable[[], None] | None = None,
        on_output_end: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            audio_recorder: The audio recorder to use.
            transcriber: The transcriber to use.
            rag_service: Optional RAG service for context retrieval.
            output_simulator: The output simulator to use.
            on_output_start: Optional callback to call before outputting text.
            on_output_end: Optional callback to call after outputting text.
        """
        self._audio_recorder = audio_recorder
        self._transcriber = transcriber
        self._rag_service = rag_service
        self._output_simulator = output_simulator
        self._on_output_start = on_output_start
        self._on_output_end = on_output_end

    def start_recording(self) -> None:
        """Start recording audio."""
        logger.info("Starting audio recording...")
        self._audio_recorder.start_recording()

    def stop_recording_and_transcribe(self) -> None:
        """Stop recording, transcribe, and output text."""
        logger.info("Stopping audio recording...")
        audio = self._audio_recorder.stop_recording()

        if audio.duration < 0.5:
            logger.warning(f"Audio too short ({audio.duration:.2f}s), skipping transcription")
            return

        logger.info(f"Transcribing {audio.duration:.2f}s of audio...")

        # Adaptive strategy based on audio length:
        # - Short audio (<30s): Two-pass with full context (initial + RAG-enhanced)
        # - Long audio (≥30s): Single-pass with condensed context to avoid token limits
        
        if audio.duration < 30.0:
            # SHORT AUDIO: Use two-pass approach for maximum accuracy
            logger.info("Using two-pass transcription (short audio)")
            
            # Get initial transcription for context retrieval
            initial_transcription = self._transcriber.transcribe(audio)
            logger.info(f"Initial transcription: {initial_transcription.text}")

            # Retrieve context if RAG service is available
            context = None
            if self._rag_service and initial_transcription.text:
                logger.info("Retrieving RAG context...")
                context = self._rag_service.retrieve_context(
                    initial_transcription.text,
                    max_tokens=100  # Full context for short audio
                )

            # Final transcription with context
            if context:
                logger.info("Re-transcribing with RAG context...")
                final_transcription = self._transcriber.transcribe(audio, context=context)
            else:
                final_transcription = initial_transcription
        else:
            # LONG AUDIO: Single-pass with condensed context
            # Whisper internally chunks at 30s, so we use a minimal context
            # that fits within the token budget for each chunk
            logger.info("Using single-pass transcription with condensed context (long audio)")
            
            # For long audio, do a quick initial pass to get query terms
            initial_transcription = self._transcriber.transcribe(audio)
            logger.info(f"Initial transcription preview: {initial_transcription.text[:100]}...")
            
            # Get highly condensed context (just key terms)
            context = None
            if self._rag_service and initial_transcription.text:
                logger.info("Retrieving condensed RAG context for long audio...")
                context, hotwords = self._rag_service.retrieve_context(
                    initial_transcription.text,
                    max_tokens=50,  # Minimal context to leave room for long transcription
                    condensed=True,  # Only phonetic matches + top technical terms
                    return_hotwords=True,
                )

            # Re-transcribe with minimal context and hotwords
            if context:
                logger.info("Re-transcribing with condensed context...")
                # For long audio, enable carrying the prompt and conditioning across chunks
                final_transcription = self._transcriber.transcribe(
                    audio,
                    context=context,
                    hotwords=hotwords,
                    condition_on_previous_text=True,
                    carry_initial_prompt=True,
                )
            else:
                final_transcription = initial_transcription

        logger.info(f"Final transcription ({len(final_transcription.text)} chars): {final_transcription.text[:100]}...")

        # Output text
        if final_transcription.text:
            logger.info("Outputting transcribed text...")

            # Call pre-output callback (e.g., to disable keyboard listener)
            if self._on_output_start:
                try:
                    self._on_output_start()
                except Exception as e:
                    logger.error(f"Error in on_output_start callback: {e}", exc_info=True)

            try:
                self._output_simulator.type_text(final_transcription.text)
                logger.info("Text output complete")
            finally:
                # Call post-output callback (e.g., to re-enable keyboard listener)
                if self._on_output_end:
                    try:
                        self._on_output_end()
                    except Exception as e:
                        logger.error(f"Error in on_output_end callback: {e}", exc_info=True)
        else:
            logger.warning("No text to output")

