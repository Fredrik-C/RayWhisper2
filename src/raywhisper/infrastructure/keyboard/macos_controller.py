"""macOS-specific keyboard controller."""

import subprocess
from typing import Callable

from loguru import logger

from .base_controller import PynputKeyboardController


class MacOSKeyboardController(PynputKeyboardController):
    """macOS-specific keyboard controller.

    Inherits from PynputKeyboardController and adds macOS-specific features if needed.
    Note: Requires Accessibility permissions on macOS.
    """

    def __init__(self) -> None:
        """Initialize the macOS keyboard controller."""
        super().__init__()
        logger.debug("Initialized macOS keyboard controller")
        logger.info(
            "Note: macOS requires Accessibility permissions for global hotkeys. "
            "Please grant permissions in System Preferences > Security & Privacy > Privacy > Accessibility"
        )

    def _is_caps_lock_on(self) -> bool:
        """Check if Caps Lock is currently enabled on macOS.

        Returns:
            bool: True if Caps Lock is on, False otherwise.
        """
        try:
            # Use hidutil to check Caps Lock state on macOS
            result = subprocess.run(
                ["hidutil", "property", "--get", "CapsLockLEDState"],
                capture_output=True,
                text=True,
                check=True
            )
            # Output is "1" if on, "0" if off
            return result.stdout.strip() == "1"
        except Exception as e:
            logger.warning(f"Failed to check Caps Lock state on macOS: {e}")
            return False

    def _check_caps_lock_state(
        self, on_enabled: Callable[[], None], on_disabled: Callable[[], None]
    ) -> None:
        """Check Caps Lock state and call appropriate callback.

        Args:
            on_enabled: Function to call when Caps Lock is enabled.
            on_disabled: Function to call when Caps Lock is disabled.
        """
        is_on = self._is_caps_lock_on()

        # Check if state has changed
        if is_on and not self._caps_lock_active:
            self._caps_lock_active = True
            logger.debug("Caps Lock enabled")
            try:
                on_enabled()
            except Exception as e:
                logger.error(f"Error in Caps Lock enabled callback: {e}", exc_info=True)
        elif not is_on and self._caps_lock_active:
            self._caps_lock_active = False
            logger.debug("Caps Lock disabled")
            try:
                on_disabled()
            except Exception as e:
                logger.error(f"Error in Caps Lock disabled callback: {e}", exc_info=True)

