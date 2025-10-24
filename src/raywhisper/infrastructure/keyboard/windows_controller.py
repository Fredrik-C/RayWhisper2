"""Windows-specific keyboard controller."""

from loguru import logger

from .base_controller import PynputKeyboardController


class WindowsKeyboardController(PynputKeyboardController):
    """Windows-specific keyboard controller.

    Inherits from PynputKeyboardController and adds Windows-specific features if needed.
    """

    def __init__(self) -> None:
        """Initialize the Windows keyboard controller."""
        super().__init__()
        logger.debug("Initialized Windows keyboard controller")

    # Add Windows-specific methods here if needed in the future

