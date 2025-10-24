"""macOS-specific keyboard controller."""

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

    # Add macOS-specific methods here if needed in the future

