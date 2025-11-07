"""macOS-specific keyboard controller."""

import ctypes
import threading
import time
from ctypes.util import find_library

from loguru import logger

from .base_controller import PynputKeyboardController


class MacOSKeyboardController(PynputKeyboardController):
    """macOS-specific keyboard controller.

    Inherits from PynputKeyboardController and adds macOS-specific features.
    Note: Requires Accessibility permissions on macOS.
    """

    # Quartz constants
    _KCG_EVENT_SOURCE_STATE_HID_SYSTEM_STATE = 1
    _KCG_EVENT_FLAG_MASK_ALPHA_SHIFT = 1 << 16

    def __init__(self) -> None:
        """Initialize the macOS keyboard controller."""
        super().__init__()
        logger.debug("Initialized macOS keyboard controller")
        logger.info(
            "Note: macOS requires Accessibility permissions for global hotkeys. "
            "Please grant permissions in System Preferences > Security & Privacy > Privacy > Accessibility"
        )

        # Background polling for Caps Lock using native Quartz API (no subprocess)
        self._caps_monitor_thread: threading.Thread | None = None
        self._caps_monitor_stop: threading.Event | None = None
        self._caps_monitor_interval: float = 0.05  # 50 ms default

        # Initialize Quartz ApplicationServices
        self._quartz = None
        self._cg_event_source_flags_state = None
        try:
            lib = find_library("ApplicationServices")
            if lib is None:
                raise OSError("ApplicationServices not found")
            self._quartz = ctypes.cdll.LoadLibrary(lib)
            # Set up CGEventSourceFlagsState
            self._cg_event_source_flags_state = self._quartz.CGEventSourceFlagsState
            self._cg_event_source_flags_state.argtypes = [ctypes.c_uint32]
            # Return type is CGEventFlags (unsigned long long)
            self._cg_event_source_flags_state.restype = ctypes.c_uint64
            logger.debug("Loaded Quartz ApplicationServices for Caps Lock detection")
        except Exception as e:
            # We still function (pynput events will trigger checks), but monitoring won't be native
            logger.warning(f"Failed to initialize Quartz for Caps Lock detection: {e}")

    def set_caps_lock_monitor_interval(self, seconds: float) -> None:
        """Configure the monitoring poll interval in seconds."""
        self._caps_monitor_interval = max(0.01, float(seconds))

    def _is_caps_lock_on(self) -> bool:
        """Check if Caps Lock is currently enabled on macOS using Quartz flags."""
        try:
            if self._cg_event_source_flags_state is None:
                return False
            flags = self._cg_event_source_flags_state(self._KCG_EVENT_SOURCE_STATE_HID_SYSTEM_STATE)
            return bool(flags & self._KCG_EVENT_FLAG_MASK_ALPHA_SHIFT)
        except Exception as e:
            logger.warning(f"Quartz Caps Lock check failed: {e}")
            return False

    # --- Caps Lock monitoring (macOS) ---------------------------------------
    def _start_caps_lock_monitor(self) -> None:
        """Start a lightweight polling loop to track Caps Lock state using Quartz."""
        if not self._caps_lock_config:
            return
        if self._caps_monitor_thread and self._caps_monitor_thread.is_alive():
            return

        if self._cg_event_source_flags_state is None:
            logger.warning("Quartz not initialized; skipping Caps Lock monitor thread on macOS")
            return

        on_enabled, on_disabled = self._caps_lock_config
        self._caps_monitor_stop = threading.Event()

        def _loop() -> None:
            logger.debug(
                f"Starting Caps Lock monitor thread (macOS, interval={self._caps_monitor_interval*1000:.0f}ms)"
            )
            try:
                while self._caps_monitor_stop and not self._caps_monitor_stop.is_set():
                    self._check_caps_lock_state(on_enabled, on_disabled)
                    time.sleep(self._caps_monitor_interval)
            except Exception as e:
                logger.error(f"Caps Lock monitor thread (macOS) crashed: {e}", exc_info=True)
            finally:
                logger.debug("Caps Lock monitor thread (macOS) stopped")

        self._caps_monitor_thread = threading.Thread(target=_loop, name="CapsLockMonitorMac", daemon=True)
        self._caps_monitor_thread.start()

    def _stop_caps_lock_monitor(self) -> None:
        """Stop the Quartz Caps Lock monitor thread if running."""
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
        if self._caps_lock_config:
            self._start_caps_lock_monitor()

    def stop_listening(self) -> None:
        """Stop monitoring and stop the pynput listener."""
        self._stop_caps_lock_monitor()
        super().stop_listening()

