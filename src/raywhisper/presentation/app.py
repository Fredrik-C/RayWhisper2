"""Main application orchestrator."""

import sys
import threading

from loguru import logger

from ..application.services.rag_service import RAGService
from ..application.use_cases.process_voice_input import ProcessVoiceInputUseCase
from ..config.settings import Settings
from ..infrastructure.audio.sounddevice_recorder import SoundDeviceRecorder
from ..infrastructure.embeddings.reranker import CrossEncoderReranker
from ..infrastructure.keyboard.factory import create_keyboard_controller
from ..infrastructure.output.keystroke_simulator import KeystrokeSimulator, SmartOutputSimulator
from ..infrastructure.transcription.whisper_transcriber import WhisperTranscriber
from ..infrastructure.vector_db.chroma_store import ChromaVectorStore


class RayWhisperApp:
    """Main application class for RayWhisper."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the application.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._setup_logging()
        self._initialize_components()

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("logs/raywhisper.log", rotation="10 MB", level="DEBUG")
        logger.info("Logging initialized")

    def _initialize_components(self) -> None:
        """Initialize all application components."""
        logger.info("Initializing RayWhisper components...")

        # Infrastructure - Audio
        self._audio_recorder = SoundDeviceRecorder(
            sample_rate=self._settings.audio.sample_rate,
            channels=self._settings.audio.channels,
        )

        # Infrastructure - Transcription
        self._transcriber = WhisperTranscriber(
            model_size=self._settings.whisper.model_size,
            device=self._settings.whisper.device,
            compute_type=self._settings.whisper.compute_type,
            beam_size=self._settings.whisper.beam_size,
            best_of=self._settings.whisper.best_of,
            temperature=self._settings.whisper.temperature,
            condition_on_previous_text=self._settings.whisper.condition_on_previous_text,
            compression_ratio_threshold=self._settings.whisper.compression_ratio_threshold,
            log_prob_threshold=self._settings.whisper.log_prob_threshold,
            no_speech_threshold=self._settings.whisper.no_speech_threshold,
            repetition_penalty=self._settings.whisper.repetition_penalty,
        )

        # Infrastructure - Vector Database
        try:
            self._vector_store = ChromaVectorStore(
                collection_name=self._settings.vector_db.collection_name,
                persist_directory=self._settings.vector_db.persist_directory,
                embedding_model_name=self._settings.vector_db.embedding_model,
                chunk_size=self._settings.vector_db.chunk_size,
                chunk_overlap=self._settings.vector_db.chunk_overlap,
                use_query_instruction=self._settings.vector_db.use_query_instruction,
            )

            # Infrastructure - Reranker
            self._reranker = CrossEncoderReranker(
                model_name=self._settings.reranker.model_name,
            )

            # Application - RAG Service
            self._rag_service: RAGService | None = RAGService(
                vector_store=self._vector_store,
                reranker=self._reranker,
                top_k=self._settings.vector_db.top_k,
                top_n=self._settings.reranker.top_n,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize RAG components: {e}")
            logger.warning("RAG features will be disabled")
            self._rag_service = None

        # Infrastructure - Output
        if self._settings.output.use_clipboard_fallback:
            self._output_simulator = SmartOutputSimulator(
                typing_speed=self._settings.output.typing_speed,
                use_clipboard_fallback=True,
            )
        else:
            self._output_simulator = KeystrokeSimulator(
                typing_speed=self._settings.output.typing_speed,
            )

        # Infrastructure - Keyboard
        self._keyboard_controller = create_keyboard_controller()

        # Application - Use Cases
        self._voice_input_use_case = ProcessVoiceInputUseCase(
            audio_recorder=self._audio_recorder,
            transcriber=self._transcriber,
            rag_service=self._rag_service,
            output_simulator=self._output_simulator,
            on_output_start=self._pause_keyboard_listener,
            on_output_end=self._resume_keyboard_listener,
        )

        logger.info("Components initialized successfully")

    def run(self) -> None:
        """Start the application."""
        logger.info("Starting RayWhisper...")

        # Register key hold (press to start, release to stop)
        self._keyboard_controller.register_key_hold(
            self._settings.keyboard.start_stop_hotkey,
            self._start_recording,
            self._stop_recording,
        )

        self._keyboard_controller.start_listening()

        logger.info(f"Listening for key hold: {self._settings.keyboard.start_stop_hotkey}")
        logger.info("Hold keys to record, release to transcribe")
        logger.info("Press Ctrl+C to exit")

        try:
            # Keep running
            event = threading.Event()
            event.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.shutdown()

    def _start_recording(self) -> None:
        """Start recording when keys are pressed."""
        try:
            if not self._audio_recorder.is_recording():
                logger.info("Keys pressed: Starting recording")
                self._voice_input_use_case.start_recording()
        except Exception as e:
            logger.error(f"Error starting recording: {e}", exc_info=True)

    def _stop_recording(self) -> None:
        """Stop recording and transcribe when keys are released."""
        try:
            if self._audio_recorder.is_recording():
                logger.info("Keys released: Stopping recording and transcribing")
                self._voice_input_use_case.stop_recording_and_transcribe()
        except Exception as e:
            logger.error(f"Error stopping recording: {e}", exc_info=True)

    def _pause_keyboard_listener(self) -> None:
        """Pause keyboard listener to prevent feedback from simulated keystrokes."""
        logger.debug("Pausing keyboard listener during text output")
        self._keyboard_controller.stop_listening()

    def _resume_keyboard_listener(self) -> None:
        """Resume keyboard listener after text output is complete."""
        logger.debug("Resuming keyboard listener after text output")
        # Re-register the key hold
        self._keyboard_controller.register_key_hold(
            self._settings.keyboard.start_stop_hotkey,
            self._start_recording,
            self._stop_recording,
        )
        self._keyboard_controller.start_listening()

    def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down RayWhisper...")

        # Stop keyboard listener
        self._keyboard_controller.stop_listening()

        # Stop recording if in progress
        if self._audio_recorder.is_recording():
            try:
                self._audio_recorder.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")

        logger.info("RayWhisper stopped")

