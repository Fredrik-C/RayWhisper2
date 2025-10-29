"""Base keyboard controller implementation using pynput."""

from abc import abstractmethod
from typing import Callable

from loguru import logger
from pynput import keyboard

from ...domain.interfaces.keyboard_controller import IKeyboardController


class PynputKeyboardController(IKeyboardController):
    """Base keyboard controller implementation using pynput."""

    def __init__(self) -> None:
        """Initialize the keyboard controller."""
        self._hotkeys: dict[str, Callable[[], None]] = {}
        self._key_hold_config: dict[str, tuple[Callable[[], None], Callable[[], None]]] | None = None
        self._caps_lock_config: tuple[Callable[[], None], Callable[[], None]] | None = None
        self._listener: keyboard.Listener | None = None
        self._pressed_keys: set[keyboard.Key | keyboard.KeyCode] = set()
        self._key_hold_active = False
        self._caps_lock_active = False

    def register_hotkey(self, hotkey: str, callback: Callable[[], None]) -> None:
        """Register a global hotkey.

        Args:
            hotkey: Hotkey string (e.g., 'ctrl+shift+space').
            callback: Function to call when hotkey is pressed.
        """
        # Convert from our format to pynput format
        pynput_hotkey = self._convert_hotkey_format(hotkey)
        self._hotkeys[pynput_hotkey] = callback
        logger.debug(f"Registered hotkey: {hotkey} -> {pynput_hotkey}")

    def register_key_hold(
        self, hotkey: str, on_press: Callable[[], None], on_release: Callable[[], None]
    ) -> None:
        """Register a key combination that triggers on press and release.

        Args:
            hotkey: Hotkey string (e.g., 'ctrl+shift+space').
            on_press: Function to call when keys are pressed down.
            on_release: Function to call when keys are released.
        """
        self._key_hold_config = (hotkey, (on_press, on_release))
        logger.debug(f"Registered key hold: {hotkey}")

    def register_caps_lock_toggle(
        self, on_enabled: Callable[[], None], on_disabled: Callable[[], None]
    ) -> None:
        """Register callbacks for Caps Lock state changes.

        Args:
            on_enabled: Function to call when Caps Lock is enabled.
            on_disabled: Function to call when Caps Lock is disabled.
        """
        self._caps_lock_config = (on_enabled, on_disabled)

        # Initialize state to match actual Caps Lock state at registration time
        # Since _is_caps_lock_on() is an abstract method, it will always be implemented
        # by subclasses (or Python will raise TypeError at instantiation)
        self._caps_lock_active = self._is_caps_lock_on()
        logger.debug(f"Registered Caps Lock toggle (initial state: {'ON' if self._caps_lock_active else 'OFF'})")

    @abstractmethod
    def _is_caps_lock_on(self) -> bool:
        """Check if Caps Lock is currently enabled.

        Platform-specific implementation required.

        Returns:
            bool: True if Caps Lock is on, False otherwise.
        """
        pass

    def _check_caps_lock_state(
        self, on_enabled: Callable[[], None], on_disabled: Callable[[], None]
    ) -> None:
        """Check Caps Lock state and call appropriate callback.

        This method is shared across all platforms. Only _is_caps_lock_on()
        needs to be implemented by platform-specific controllers.

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

    def _convert_hotkey_format(self, hotkey: str) -> str:
        """Convert 'ctrl+shift+space' to '<ctrl>+<shift>+<space>'.

        Args:
            hotkey: Hotkey string in our format.

        Returns:
            str: Hotkey string in pynput format.
        """
        parts = hotkey.lower().split("+")
        return "+".join(f"<{part}>" for part in parts)

    def _parse_hotkey_to_keys(self, hotkey: str) -> list[set[keyboard.Key | keyboard.KeyCode]]:
        """Parse hotkey string to list of key sets (for handling modifier variants).

        Args:
            hotkey: Hotkey string (e.g., 'ctrl+shift+space').

        Returns:
            List of sets, where each set contains the possible keys for that position.
            For modifiers like 'ctrl', the set contains both left and right variants.
            For regular keys, the set contains just that key.
        """
        parts = hotkey.lower().split("+")
        key_groups = []

        for part in parts:
            part = part.strip()
            # Map common key names to pynput keys
            # For modifiers, create a set with both left and right variants
            if part == "ctrl":
                key_groups.append({keyboard.Key.ctrl_l, keyboard.Key.ctrl_r})
            elif part == "shift":
                key_groups.append({keyboard.Key.shift_l, keyboard.Key.shift_r})
            elif part == "alt":
                key_groups.append({keyboard.Key.alt_l, keyboard.Key.alt_r})
            elif part == "cmd" or part == "super":
                key_groups.append({keyboard.Key.cmd})
            elif part == "space":
                key_groups.append({keyboard.Key.space})
            elif part == "esc":
                key_groups.append({keyboard.Key.esc})
            elif len(part) == 1:
                # Single character key
                key_groups.append({keyboard.KeyCode.from_char(part)})
            else:
                logger.warning(f"Unknown key: {part}")

        return key_groups

    def _check_key_combination(
        self, required_key_groups: list[set[keyboard.Key | keyboard.KeyCode]]
    ) -> bool:
        """Check if all required key groups are currently pressed.

        Args:
            required_key_groups: List of key sets. For each set, at least one key must be pressed.

        Returns:
            True if all required key groups have at least one key pressed.
        """
        # For each group of keys (e.g., {ctrl_l, ctrl_r}), check if ANY key in that group is pressed
        for key_group in required_key_groups:
            group_satisfied = False
            for req_key in key_group:
                if req_key in self._pressed_keys:
                    group_satisfied = True
                    break
                # Handle canonical form matching for character keys
                for pressed_key in self._pressed_keys:
                    try:
                        if hasattr(pressed_key, "char") and hasattr(req_key, "char"):
                            if pressed_key.char == req_key.char:
                                group_satisfied = True
                                break
                    except AttributeError:
                        pass
                if group_satisfied:
                    break

            if not group_satisfied:
                return False

        return True

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        """Handle key press events.

        Args:
            key: The key that was pressed.
        """
        self._pressed_keys.add(key)

        # Check if Caps Lock was pressed and toggle monitoring is configured
        if self._caps_lock_config and key == keyboard.Key.caps_lock:
            on_enabled, on_disabled = self._caps_lock_config
            # Call the shared method to check Caps Lock state
            self._check_caps_lock_state(on_enabled, on_disabled)

        # Check if key hold is configured
        if self._key_hold_config:
            hotkey, (on_press_callback, _) = self._key_hold_config
            required_keys = self._parse_hotkey_to_keys(hotkey)

            if self._check_key_combination(required_keys) and not self._key_hold_active:
                self._key_hold_active = True
                logger.debug(f"Key combination pressed: {hotkey}")
                try:
                    on_press_callback()
                except Exception as e:
                    logger.error(f"Error in key press callback: {e}", exc_info=True)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        """Handle key release events.

        Args:
            key: The key that was released.
        """
        # Remove the key from pressed keys
        if key in self._pressed_keys:
            self._pressed_keys.remove(key)

        # Check if key hold was active and should be deactivated
        if self._key_hold_config and self._key_hold_active:
            hotkey, (_, on_release_callback) = self._key_hold_config
            required_keys = self._parse_hotkey_to_keys(hotkey)

            # If any required key is released, trigger the release callback
            if not self._check_key_combination(required_keys):
                self._key_hold_active = False
                logger.debug(f"Key combination released: {hotkey}")
                try:
                    on_release_callback()
                except Exception as e:
                    logger.error(f"Error in key release callback: {e}", exc_info=True)

    def start_listening(self) -> None:
        """Start listening for hotkeys."""
        if self._listener is not None:
            logger.warning("Listener already started")
            return

        if not self._hotkeys and not self._key_hold_config and not self._caps_lock_config:
            logger.warning("No hotkeys, key holds, or Caps Lock toggle registered")
            return

        logger.info("Starting keyboard listener")
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        logger.info("Keyboard listener started")

    def stop_listening(self) -> None:
        """Stop listening for hotkeys."""
        if self._listener is None:
            logger.debug("Listener not running")
            return

        logger.info("Stopping keyboard listener")
        self._listener.stop()
        self._listener = None
        logger.info("Keyboard listener stopped")

    def unregister_all(self) -> None:
        """Unregister all hotkeys."""
        logger.info("Unregistering all hotkeys")
        self.stop_listening()
        self._hotkeys.clear()
        self._key_hold_config = None
        self._caps_lock_config = None
        self._pressed_keys.clear()
        self._key_hold_active = False
        self._caps_lock_active = False

