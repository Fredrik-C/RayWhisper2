"""macOS-specific keyboard controller."""

import subprocess

from loguru import logger

from .base_controller import PynputKeyboardController


class MacOSKeyboardController(PynputKeyboardController):
    """macOS-specific keyboard controller.

    Inherits from PynputKeyboardController and adds macOS-specific features.
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

        Uses hidutil command to check the Caps Lock LED state.
        Note: This spawns a subprocess which may have performance implications.
        Consider caching or using native APIs for production use.

        Returns:
            bool: True if Caps Lock is on, False otherwise.
        """
        try:
            # Use hidutil to check Caps Lock state on macOS
            # TODO: Consider using native macOS API via ctypes for better performance
            result = subprocess.run(
                ["hidutil", "property", "--get", "CapsLockLEDState"],
                capture_output=True,
                text=True,
                check=True,
                timeout=0.1  # Short timeout to prevent UI lag on key press
            )
            # Output is "1" if on, "0" if off
            return result.stdout.strip() == "1"
        except subprocess.TimeoutExpired:
            logger.warning("Timeout checking Caps Lock state on macOS")
            return False
        except subprocess.CalledProcessError as e:
            logger.warning(f"hidutil command failed (exit code {e.returncode}): {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("hidutil command not found - Caps Lock detection unavailable on this macOS version")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking Caps Lock state on macOS: {e}")
            return False

