"""Configuration loader for YAML files."""

import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .settings import Settings


def get_default_config_path() -> Path:
    """Get the default config file path based on execution context.
    
    Returns:
        Path to the default config.yaml file.
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable (PyInstaller)
        application_path = Path(sys.executable).parent
        config_path = application_path / "config" / "config.yaml"
    else:
        # Running as script - look in project root
        # Go up from src/raywhisper/config to project root
        application_path = Path(__file__).parent.parent.parent.parent
        config_path = application_path / "config" / "config.yaml"
    
    return config_path


def load_yaml_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the config file. If None, uses default location.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid YAML.
    """
    if config_path is None:
        config_path = get_default_config_path()
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        if config_data is None:
            logger.warning(f"Config file is empty: {config_path}")
            return {}
            
        logger.debug(f"Loaded config: {config_data}")
        return config_data
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load Settings from YAML config file with environment variable overrides.
    
    This function:
    1. Loads configuration from YAML file (if it exists)
    2. Creates Settings object with YAML values
    3. Allows environment variables to override YAML values
    
    Args:
        config_path: Path to the config file. If None, uses default location.
        
    Returns:
        Settings object with configuration loaded.
    """
    # Load YAML config
    yaml_config = load_yaml_config(config_path)
    
    if yaml_config:
        logger.info("Creating settings from YAML config (environment variables can override)")
        # Create Settings with YAML values, allowing env vars to override
        # Pydantic will use the provided values as defaults and check env vars
        settings = Settings(**yaml_config)
    else:
        logger.info("No YAML config found, using defaults and environment variables")
        settings = Settings()
    
    return settings

