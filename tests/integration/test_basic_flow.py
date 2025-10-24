"""Basic integration tests for RayWhisper."""

import pytest

from raywhisper.config.settings import Settings


def test_settings_load() -> None:
    """Test that settings can be loaded."""
    settings = Settings()

    assert settings.audio.sample_rate == 16000
    assert settings.audio.channels == 1
    # model_size can be a standard size or a HuggingFace model ID
    assert isinstance(settings.whisper.model_size, str)
    assert len(settings.whisper.model_size) > 0
    assert settings.whisper.device in ["cpu", "cuda", "auto"]
    assert settings.whisper.compute_type in ["int8", "int8_float16", "float16", "float32"]


def test_settings_nested_access() -> None:
    """Test nested settings access."""
    settings = Settings()

    # Test audio settings
    assert hasattr(settings.audio, "sample_rate")
    assert hasattr(settings.audio, "channels")

    # Test whisper settings
    assert hasattr(settings.whisper, "model_size")
    assert hasattr(settings.whisper, "device")

    # Test vector_db settings
    assert hasattr(settings.vector_db, "collection_name")
    assert hasattr(settings.vector_db, "embedding_model")

    # Test keyboard settings
    assert hasattr(settings.keyboard, "start_stop_hotkey")


def test_default_values() -> None:
    """Test that default values are set correctly (or loaded from .env if present)."""
    settings = Settings()

    assert settings.audio.sample_rate == 16000
    # model_size can be default "base" or loaded from .env
    assert isinstance(settings.whisper.model_size, str)
    assert len(settings.whisper.model_size) > 0
    # Note: reranker model may vary based on config (default is BAAI/bge-reranker-v2-m3)
    assert isinstance(settings.reranker.model_name, str)
    assert len(settings.reranker.model_name) > 0
    # keyboard.start_stop_hotkey can be default "ctrl+shift+space" or loaded from .env
    assert isinstance(settings.keyboard.start_stop_hotkey, str)
    assert len(settings.keyboard.start_stop_hotkey) > 0
    # typing_speed may vary based on config
    assert isinstance(settings.output.typing_speed, float)
    assert settings.output.typing_speed > 0
    assert settings.output.use_clipboard_fallback is False


def test_custom_whisper_model() -> None:
    """Test that custom Whisper model IDs can be configured."""
    import os

    # Save original value
    original = os.environ.get("RAYWHISPER_WHISPER__MODEL_SIZE")

    try:
        # Set custom model
        os.environ["RAYWHISPER_WHISPER__MODEL_SIZE"] = "Systran/faster-whisper-medium.en"

        # Create new settings instance
        settings = Settings()

        # Verify custom model is loaded
        assert settings.whisper.model_size == "Systran/faster-whisper-medium.en"

    finally:
        # Restore original value
        if original is not None:
            os.environ["RAYWHISPER_WHISPER__MODEL_SIZE"] = original
        elif "RAYWHISPER_WHISPER__MODEL_SIZE" in os.environ:
            del os.environ["RAYWHISPER_WHISPER__MODEL_SIZE"]

