"""Windows-specific keyboard controller."""

import ctypes
from typing import Callable

from loguru import logger

from .base_controller import PynputKeyboardController


class WindowsKeyboardController(PynputKeyboardController):
    """Windows-specific keyboard controller.

    Inherits from PynputKeyboardController and adds Windows-specific features if needed.
    """

    VK_CAPITAL = 0x14  # Virtual key code for Caps Lock

    def __init__(self) -> None:
        """Initialize the Windows keyboard controller."""
        super().__init__()
        logger.debug("Initialized Windows keyboard controller")

    def _is_caps_lock_on(self) -> bool:
        """Check if Caps Lock is currently enabled on Windows.

        Returns:
            bool: True if Caps Lock is on, False otherwise.
        """
        # GetKeyState returns the state of the specified virtual key
        # The low-order bit indicates whether the key is toggled (on/off)
        return bool(ctypes.windll.user32.GetKeyState(self.VK_CAPITAL) & 1)

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

