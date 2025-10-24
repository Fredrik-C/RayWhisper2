"""Keystroke simulator for typing text output."""

import time

from loguru import logger
from pynput.keyboard import Controller, Key

from ...domain.interfaces.output_simulator import IOutputSimulator


class KeystrokeSimulator(IOutputSimulator):
    """Output simulator that types text using keystroke simulation."""

    def __init__(self, typing_speed: float = 0.01) -> None:
        """Initialize the keystroke simulator.

        Args:
            typing_speed: Delay between keystrokes in seconds.
        """
        self._keyboard = Controller()
        self._typing_speed = typing_speed
        logger.debug(f"Initialized keystroke simulator with typing_speed={typing_speed}")

    def _type_chunk(self, chunk: str) -> None:
        """Type a chunk of text.

        Args:
            chunk: The text chunk to type.
        """
        for ch in chunk:
            self._keyboard.type(ch)
            time.sleep(self._typing_speed)

    def type_text(self, text: str) -> None:
        """Simulate typing the given text.

        Args:
            text: The text to type.
        """
        if not text:
            logger.debug("No text to type")
            return

        logger.info(f"Typing {len(text)} characters")

        # Split by lines to preserve newlines
        lines = text.splitlines()

        for i, line in enumerate(lines):
            # Type in chunks to prevent event queue overflow on very long strings
            start = 0
            max_chunk = 200

            while start < len(line):
                end = min(start + max_chunk, len(line))
                self._type_chunk(line[start:end])
                start = end

                # Brief yield between big chunks
                if start < len(line):
                    time.sleep(0.02)

            # Press Enter between lines (except after the last line)
            if i < len(lines) - 1:
                self._keyboard.press(Key.enter)
                self._keyboard.release(Key.enter)
                time.sleep(self._typing_speed)

        logger.info("Typing complete")


class SmartOutputSimulator(IOutputSimulator):
    """Output simulator with clipboard fallback option."""

    def __init__(
        self, typing_speed: float = 0.01, use_clipboard_fallback: bool = False
    ) -> None:
        """Initialize the smart output simulator.

        Args:
            typing_speed: Delay between keystrokes in seconds.
            use_clipboard_fallback: Whether to use clipboard fallback on error.
        """
        self._keystroke_sim = KeystrokeSimulator(typing_speed)
        self._use_clipboard_fallback = use_clipboard_fallback
        logger.debug(
            f"Initialized smart output simulator with clipboard_fallback={use_clipboard_fallback}"
        )

    def type_text(self, text: str) -> None:
        """Simulate typing the given text with optional clipboard fallback.

        Args:
            text: The text to type.
        """
        try:
            self._keystroke_sim.type_text(text)
        except Exception as e:
            logger.error(f"Keystroke simulation failed: {e}")

            if self._use_clipboard_fallback:
                logger.info("Falling back to clipboard paste")
                self._fallback_to_clipboard(text)
            else:
                raise

    def _fallback_to_clipboard(self, text: str) -> None:
        """Fallback: copy to clipboard and paste.

        Args:
            text: The text to copy and paste.
        """
        try:
            import pyperclip

            pyperclip.copy(text)
            logger.debug("Text copied to clipboard")

            # Simulate Ctrl+V (or Cmd+V on macOS)
            import platform

            kb = Controller()

            if platform.system() == "Darwin":  # macOS
                # Use Cmd+V on macOS
                from pynput.keyboard import Key as K

                with kb.pressed(K.cmd):
                    kb.press("v")
                    kb.release("v")
            else:
                # Use Ctrl+V on Windows/Linux
                with kb.pressed(Key.ctrl):
                    kb.press("v")
                    kb.release("v")

            logger.info("Text pasted from clipboard")

        except Exception as e:
            logger.error(f"Clipboard fallback failed: {e}")
            raise

