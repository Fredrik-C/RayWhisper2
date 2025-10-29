"""Windows-specific keyboard controller."""

import ctypes

from loguru import logger

from .base_controller import PynputKeyboardController


class WindowsKeyboardController(PynputKeyboardController):
    """Windows-specific keyboard controller.

    Inherits from PynputKeyboardController and adds Windows-specific features.
    """

    VK_CAPITAL = 0x14  # Virtual key code for Caps Lock

    def __init__(self) -> None:
        """Initialize the Windows keyboard controller."""
        super().__init__()
        logger.debug("Initialized Windows keyboard controller")

    def _is_caps_lock_on(self) -> bool:
        """Check if Caps Lock is currently enabled on Windows.

        Uses Windows API GetKeyState() to check the toggle state.

        Returns:
            bool: True if Caps Lock is on, False otherwise.
        """
        # GetKeyState returns the state of the specified virtual key
        # The low-order bit indicates whether the key is toggled (on/off)
        return bool(ctypes.windll.user32.GetKeyState(self.VK_CAPITAL) & 1)

