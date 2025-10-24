"""Tests for configuration loader."""

import tempfile
from pathlib import Path

import pytest

from raywhisper.config.loader import load_settings, load_yaml_config


def test_load_yaml_config_from_file():
    """Test loading YAML config from a file."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
audio:
  sample_rate: 48000
  channels: 2

whisper:
  model_size: "large-v3"
  device: "cuda"

vector_db:
  embedding_model: "BAAI/bge-base-en-v1.5"
  chunk_size: 5

keyboard:
  start_stop_hotkey: "ctrl+alt+space"
""")
        config_path = Path(f.name)
    
    try:
        # Load the config
        config = load_yaml_config(config_path)
        
        # Verify the values
        assert config['audio']['sample_rate'] == 48000
        assert config['audio']['channels'] == 2
        assert config['whisper']['model_size'] == "large-v3"
        assert config['whisper']['device'] == "cuda"
        assert config['vector_db']['embedding_model'] == "BAAI/bge-base-en-v1.5"
        assert config['vector_db']['chunk_size'] == 5
        assert config['keyboard']['start_stop_hotkey'] == "ctrl+alt+space"
        
    finally:
        # Clean up
        config_path.unlink()


def test_load_yaml_config_nonexistent_file():
    """Test loading config from a nonexistent file returns empty dict."""
    config = load_yaml_config("/nonexistent/path/config.yaml")
    assert config == {}


def test_load_settings_from_yaml():
    """Test loading Settings from YAML config."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
audio:
  sample_rate: 48000
  channels: 2

whisper:
  model_size: "large-v3"
  device: "cuda"

vector_db:
  embedding_model: "BAAI/bge-base-en-v1.5"
  chunk_size: 5

keyboard:
  start_stop_hotkey: "ctrl+alt+space"
""")
        config_path = Path(f.name)
    
    try:
        # Load settings
        settings = load_settings(config_path)
        
        # Verify the values
        assert settings.audio.sample_rate == 48000
        assert settings.audio.channels == 2
        assert settings.whisper.model_size == "large-v3"
        assert settings.whisper.device == "cuda"
        assert settings.vector_db.embedding_model == "BAAI/bge-base-en-v1.5"
        assert settings.vector_db.chunk_size == 5
        assert settings.keyboard.start_stop_hotkey == "ctrl+alt+space"
        
    finally:
        # Clean up
        config_path.unlink()


def test_load_settings_with_defaults():
    """Test loading Settings without config file uses defaults."""
    settings = load_settings("/nonexistent/path/config.yaml")
    
    # Should use defaults from Settings class
    assert settings.audio.sample_rate == 16000
    assert settings.audio.channels == 1
    assert settings.whisper.model_size == "base"
    assert settings.whisper.device == "auto"

