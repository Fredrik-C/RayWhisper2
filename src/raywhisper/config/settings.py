"""Configuration management using Pydantic settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    """Audio recording configuration."""

    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_duration: float = Field(default=0.1, description="Audio chunk duration in seconds")


class WhisperSettings(BaseSettings):
    """Whisper model configuration."""

    model_size: str = Field(
        default="base",
        description="Whisper model size (tiny, base, small, medium, large-v2, large-v3, distil-large-v3) or HuggingFace model ID (e.g., Systran/faster-whisper-medium.en)",
    )
    device: Literal["cpu", "cuda", "auto"] = Field(
        default="auto", description="Device to run model on"
    )
    compute_type: Literal["int8", "int8_float16", "float16", "float32"] = Field(
        default="float16",
        description="CTranslate2 compute type; int8_float16 for INT8 weights with FP16 accumulators",
    )
    language: str | None = Field(default=None, description="Language code or None for auto-detect")

    # Transcription parameters for accuracy optimization
    beam_size: int = Field(
        default=5,
        ge=1,
        description="Beam size for beam search. Higher = more accurate but slower. Recommended: 5-10"
    )
    best_of: int = Field(
        default=5,
        ge=1,
        description="Number of candidates to generate. Higher = better quality. Recommended: 5"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature. 0.0 = deterministic (recommended for accuracy), higher = more creative"
    )
    condition_on_previous_text: bool = Field(
        default=False,
        description="Use previous text as context. False recommended for better accuracy with RAG context"
    )
    compression_ratio_threshold: float | None = Field(
        default=2.4,
        description="Detect repetitions. Lower = more strict. None = disable. Recommended: 2.4"
    )
    log_prob_threshold: float | None = Field(
        default=-1.0,
        description="Filter low-confidence segments. Higher = more strict. None = disable. Recommended: -1.0"
    )
    no_speech_threshold: float | None = Field(
        default=0.6,
        description="Detect silence. Higher = more strict. None = disable. Recommended: 0.6"
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        description="Penalize repetitions. 1.0 = no penalty, higher = more penalty. Recommended: 1.0-1.2"
    )


class VectorDBSettings(BaseSettings):
    """Vector database configuration."""

    collection_name: str = Field(default="raywhisper_docs", description="ChromaDB collection name")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name; consider BAAI/bge-m3 for multilingual/long-docs",
    )
    persist_directory: str = Field(
        default="./data/chroma", description="Directory to persist vector database"
    )
    top_k: int = Field(default=5, description="Number of results to retrieve")
    chunk_size: int = Field(
        default=3, description="Number of words per chunk for embedding (use small values like 2-5 for keyword/phrase matching)"
    )
    chunk_overlap: int = Field(
        default=2, description="Number of overlapping words between chunks (should be less than chunk_size)"
    )
    use_query_instruction: bool = Field(
        default=False, description="Whether to use instruction prefix for queries (True for sentence-based, False for keyword-based search)"
    )


class RerankerSettings(BaseSettings):
    """Reranker configuration."""

    model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranker model name; alternative: jinaai/jina-reranker-v2-base-multilingual (non-commercial)",
    )
    top_n: int = Field(default=3, description="Number of results after reranking")


class KeyboardSettings(BaseSettings):
    """Keyboard control configuration."""

    start_stop_hotkey: str = Field(
        default="ctrl+shift+space",
        description="Key combination to hold for recording (press to start, release to stop)",
    )
    cancel_hotkey: str = Field(default="ctrl+shift+esc", description="Hotkey to cancel recording")


class OutputSettings(BaseSettings):
    """Output simulation configuration."""

    typing_speed: float = Field(
        default=0.01, description="Delay between keystrokes in seconds"
    )
    use_clipboard_fallback: bool = Field(
        default=False, description="Use clipboard fallback if keystroke simulation fails"
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        env_prefix="RAYWHISPER_",
        case_sensitive=False,
    )

    audio: AudioSettings = Field(default_factory=AudioSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    keyboard: KeyboardSettings = Field(default_factory=KeyboardSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

