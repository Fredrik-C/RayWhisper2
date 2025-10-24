"""Transcribe audio use case."""

from loguru import logger

from ...domain.entities.transcription import Transcription
from ...domain.interfaces.transcriber import ITranscriber
from ...domain.interfaces.vector_store import IVectorStore
from ...domain.value_objects.audio_data import AudioData


class TranscribeAudioUseCase:
    """Use case for transcribing audio with optional RAG enhancement."""

    def __init__(
        self,
        transcriber: ITranscriber,
        vector_store: IVectorStore | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            transcriber: The transcriber to use.
            vector_store: Optional vector store for RAG enhancement.
        """
        self._transcriber = transcriber
        self._vector_store = vector_store

    def execute(self, audio: AudioData, use_rag: bool = True) -> Transcription:
        """Execute the transcription use case.

        Args:
            audio: The audio data to transcribe.
            use_rag: Whether to use RAG enhancement.

        Returns:
            Transcription: The transcription result.
        """
        context = None

        if use_rag and self._vector_store:
            logger.info("Using RAG enhancement for transcription")

            # First pass: quick transcription for context retrieval
            logger.debug("Performing initial transcription for context retrieval")
            initial_transcription = self._transcriber.transcribe(audio)

            if initial_transcription.text:
                # Retrieve relevant context
                logger.debug(f"Searching for context: '{initial_transcription.text[:50]}...'")
                results = self._vector_store.search(
                    query=initial_transcription.text,
                    top_k=5,
                )

                if results:
                    context = "\n".join([r.content for r in results])
                    logger.debug(f"Retrieved {len(results)} context chunks ({len(context)} chars)")
                else:
                    logger.debug("No context found in vector store")

        # Final transcription with context (or single transcription if no RAG)
        if context:
            logger.info("Performing final transcription with RAG context")
            return self._transcriber.transcribe(audio, context=context)
        else:
            if use_rag and self._vector_store:
                logger.info("No context found, using initial transcription")
                # Return the initial transcription if we already did it
                return self._transcriber.transcribe(audio)
            else:
                logger.info("Performing transcription without RAG")
                return self._transcriber.transcribe(audio)

