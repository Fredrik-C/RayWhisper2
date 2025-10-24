"""Factory for creating platform-specific keyboard controllers."""

import platform

from loguru import logger

from ...domain.interfaces.keyboard_controller import IKeyboardController
from .macos_controller import MacOSKeyboardController
from .windows_controller import WindowsKeyboardController


def create_keyboard_controller() -> IKeyboardController:
    """Create a keyboard controller for the current platform.

    Returns:
        IKeyboardController: Platform-specific keyboard controller.

    Raises:
        NotImplementedError: If the platform is not supported.
    """
    system = platform.system()
    logger.info(f"Creating keyboard controller for platform: {system}")

    if system == "Windows":
        return WindowsKeyboardController()
    elif system == "Darwin":  # macOS
        return MacOSKeyboardController()
    else:
        raise NotImplementedError(f"Platform {system} is not supported")

