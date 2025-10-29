"""Keyboard controller interface."""

from abc import ABC, abstractmethod
from typing import Callable


class IKeyboardController(ABC):
    """Interface for keyboard control and hotkey management."""

    @abstractmethod
    def register_hotkey(self, hotkey: str, callback: Callable[[], None]) -> None:
        """Register a global hotkey.

        Args:
            hotkey: Hotkey string (e.g., 'ctrl+shift+space').
            callback: Function to call when hotkey is pressed.
        """
        pass

    @abstractmethod
    def register_key_hold(
        self, hotkey: str, on_press: Callable[[], None], on_release: Callable[[], None]
    ) -> None:
        """Register a key combination that triggers on press and release.

        Args:
            hotkey: Hotkey string (e.g., 'ctrl+shift+space').
            on_press: Function to call when keys are pressed down.
            on_release: Function to call when keys are released.
        """
        pass

    @abstractmethod
    def register_caps_lock_toggle(
        self, on_enabled: Callable[[], None], on_disabled: Callable[[], None]
    ) -> None:
        """Register callbacks for Caps Lock state changes.

        Args:
            on_enabled: Function to call when Caps Lock is enabled.
            on_disabled: Function to call when Caps Lock is disabled.
        """
        pass

    @abstractmethod
    def unregister_all(self) -> None:
        """Unregister all hotkeys."""
        pass

    @abstractmethod
    def start_listening(self) -> None:
        """Start listening for hotkeys."""
        pass

    @abstractmethod
    def stop_listening(self) -> None:
        """Stop listening for hotkeys."""
        pass

