"""Manual test script for Caps Lock toggle functionality."""

import time
from loguru import logger

from src.raywhisper.infrastructure.keyboard.factory import create_keyboard_controller


def on_caps_lock_enabled():
    """Callback when Caps Lock is enabled."""
    logger.info("✓ Caps Lock ENABLED - Recording would start here")


def on_caps_lock_disabled():
    """Callback when Caps Lock is disabled."""
    logger.info("✗ Caps Lock DISABLED - Recording would stop here")


def main():
    """Test the Caps Lock toggle functionality."""
    logger.info("Starting Caps Lock toggle test...")
    logger.info("Press Caps Lock to toggle between enabled/disabled states")
    logger.info("Press Ctrl+C to exit")
    
    # Create keyboard controller
    controller = create_keyboard_controller()
    
    # Register Caps Lock toggle
    controller.register_caps_lock_toggle(
        on_enabled=on_caps_lock_enabled,
        on_disabled=on_caps_lock_disabled,
    )
    
    # Start listening
    controller.start_listening()
    
    try:
        # Keep running
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        controller.stop_listening()


if __name__ == "__main__":
    main()

