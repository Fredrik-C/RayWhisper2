"""Windows-specific keyboard controller."""

import ctypes
import threading
import time

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
        # Background polling for Caps Lock to avoid missed events
        self._caps_monitor_thread: threading.Thread | None = None
        self._caps_monitor_stop: threading.Event | None = None
        self._caps_monitor_interval: float = 0.05  # 50 ms for responsiveness
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

    def set_caps_lock_monitor_interval(self, seconds: float) -> None:
        """Configure the monitoring poll interval in seconds."""
        self._caps_monitor_interval = max(0.01, float(seconds))

    # --- Caps Lock monitoring (Windows) -------------------------------------
    def _start_caps_lock_monitor(self) -> None:
        """Start a lightweight polling loop to track Caps Lock state changes.

        This complements pynput events to prevent getting out of sync when
        a Caps Lock press is missed (common with global hooks on Windows).
        """
        if not self._caps_lock_config:
            return
        if self._caps_monitor_thread and self._caps_monitor_thread.is_alive():
            return

        on_enabled, on_disabled = self._caps_lock_config
        self._caps_monitor_stop = threading.Event()

        def _loop() -> None:
            logger.debug(
                f"Starting Caps Lock monitor thread (interval={self._caps_monitor_interval*1000:.0f}ms)"
            )
            try:
                while self._caps_monitor_stop and not self._caps_monitor_stop.is_set():
                    self._check_caps_lock_state(on_enabled, on_disabled)
                    time.sleep(self._caps_monitor_interval)
            except Exception as e:
                # Fail fast with context; do not silently swallow issues
                logger.error(f"Caps Lock monitor thread crashed: {e}", exc_info=True)
            finally:
                logger.debug("Caps Lock monitor thread stopped")

        self._caps_monitor_thread = threading.Thread(target=_loop, name="CapsLockMonitor", daemon=True)
        self._caps_monitor_thread.start()

    def _stop_caps_lock_monitor(self) -> None:
        """Stop the Caps Lock monitor thread if running."""
        if self._caps_monitor_stop:
            self._caps_monitor_stop.set()
        if self._caps_monitor_thread:
            self._caps_monitor_thread.join(timeout=0.5)
        self._caps_monitor_thread = None
        self._caps_monitor_stop = None

    # --- Lifecycle ----------------------------------------------------------
    def start_listening(self) -> None:
        """Start pynput listener and begin monitoring Caps Lock state."""
        super().start_listening()
        # Only start monitoring if Caps Lock callbacks are registered
        if self._caps_lock_config:
            self._start_caps_lock_monitor()

    def stop_listening(self) -> None:
        """Stop monitoring and stop the pynput listener."""
        self._stop_caps_lock_monitor()
        super().stop_listening()

