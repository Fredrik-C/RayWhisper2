"""Tests for Caps Lock toggle functionality in keyboard controllers."""

from unittest.mock import MagicMock, patch

import pytest

from raywhisper.infrastructure.keyboard.base_controller import PynputKeyboardController
from raywhisper.infrastructure.keyboard.windows_controller import WindowsKeyboardController


class MockKeyboardController(PynputKeyboardController):
    """Mock keyboard controller for testing base class functionality."""

    def __init__(self, initial_caps_lock_state: bool = False):
        """Initialize mock controller with configurable Caps Lock state."""
        super().__init__()
        self._mock_caps_lock_state = initial_caps_lock_state

    def _is_caps_lock_on(self) -> bool:
        """Mock implementation of Caps Lock state check."""
        return self._mock_caps_lock_state

    def set_caps_lock_state(self, state: bool) -> None:
        """Helper method to change Caps Lock state for testing."""
        self._mock_caps_lock_state = state


@pytest.fixture
def mock_controller() -> MockKeyboardController:
    """Create a mock keyboard controller for testing."""
    return MockKeyboardController(initial_caps_lock_state=False)


@pytest.fixture
def mock_controller_caps_on() -> MockKeyboardController:
    """Create a mock keyboard controller with Caps Lock initially on."""
    return MockKeyboardController(initial_caps_lock_state=True)


class TestCapsLockStateInitialization:
    """Tests for Caps Lock state initialization."""

    def test_state_initialized_to_false_when_caps_lock_off(
        self, mock_controller: MockKeyboardController
    ) -> None:
        """Test that state is initialized to False when Caps Lock is off."""
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        # Caps Lock is off initially
        assert mock_controller._is_caps_lock_on() is False

        # Register callbacks
        mock_controller.register_caps_lock_toggle(on_enabled, on_disabled)

        # State should be initialized to False
        assert mock_controller._caps_lock_active is False

        # No callbacks should be triggered during registration
        on_enabled.assert_not_called()
        on_disabled.assert_not_called()

    def test_state_initialized_to_true_when_caps_lock_on(
        self, mock_controller_caps_on: MockKeyboardController
    ) -> None:
        """Test that state is initialized to True when Caps Lock is on."""
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        # Caps Lock is on initially
        assert mock_controller_caps_on._is_caps_lock_on() is True

        # Register callbacks
        mock_controller_caps_on.register_caps_lock_toggle(on_enabled, on_disabled)

        # State should be initialized to True
        assert mock_controller_caps_on._caps_lock_active is True

        # No callbacks should be triggered during registration
        on_enabled.assert_not_called()
        on_disabled.assert_not_called()

    def test_state_matches_actual_caps_lock_state_off(
        self, mock_controller: MockKeyboardController
    ) -> None:
        """Test that initialized state matches actual Caps Lock state when off."""
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        # Test with Caps Lock off
        mock_controller.set_caps_lock_state(False)
        mock_controller.register_caps_lock_toggle(on_enabled, on_disabled)
        assert mock_controller._caps_lock_active == mock_controller._is_caps_lock_on()
        assert mock_controller._caps_lock_active is False

    def test_state_matches_actual_caps_lock_state_on(
        self, mock_controller_caps_on: MockKeyboardController
    ) -> None:
        """Test that initialized state matches actual Caps Lock state when on."""
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        # Test with Caps Lock on (using fresh fixture)
        mock_controller_caps_on.set_caps_lock_state(True)
        mock_controller_caps_on.register_caps_lock_toggle(on_enabled, on_disabled)
        assert mock_controller_caps_on._caps_lock_active == mock_controller_caps_on._is_caps_lock_on()
        assert mock_controller_caps_on._caps_lock_active is True


class TestCapsLockStateChanges:
    """Tests for Caps Lock state change detection."""

    def test_enabling_caps_lock_triggers_on_enabled_callback(
        self, mock_controller: MockKeyboardController
    ) -> None:
        """Test that enabling Caps Lock triggers the on_enabled callback."""
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        # Register with Caps Lock off
        mock_controller.register_caps_lock_toggle(on_enabled, on_disabled)
        assert mock_controller._caps_lock_active is False

        # Enable Caps Lock
        mock_controller.set_caps_lock_state(True)
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)

        # on_enabled should be called once
        on_enabled.assert_called_once()
        on_disabled.assert_not_called()

        # State should be updated
        assert mock_controller._caps_lock_active is True

    def test_disabling_caps_lock_triggers_on_disabled_callback(
        self, mock_controller_caps_on: MockKeyboardController
    ) -> None:
        """Test that disabling Caps Lock triggers the on_disabled callback."""
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        # Register with Caps Lock on
        mock_controller_caps_on.register_caps_lock_toggle(on_enabled, on_disabled)
        assert mock_controller_caps_on._caps_lock_active is True

        # Disable Caps Lock
        mock_controller_caps_on.set_caps_lock_state(False)
        mock_controller_caps_on._check_caps_lock_state(on_enabled, on_disabled)

        # on_disabled should be called once
        on_disabled.assert_called_once()
        on_enabled.assert_not_called()

        # State should be updated
        assert mock_controller_caps_on._caps_lock_active is False

    def test_no_callback_when_state_unchanged(
        self, mock_controller: MockKeyboardController
    ) -> None:
        """Test that no callback is triggered when state hasn't changed."""
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        # Register with Caps Lock off
        mock_controller.register_caps_lock_toggle(on_enabled, on_disabled)

        # Check state multiple times without changing it
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)

        # No callbacks should be triggered
        on_enabled.assert_not_called()
        on_disabled.assert_not_called()

    def test_multiple_state_changes(self, mock_controller: MockKeyboardController) -> None:
        """Test multiple Caps Lock state changes."""
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        # Register with Caps Lock off
        mock_controller.register_caps_lock_toggle(on_enabled, on_disabled)

        # Enable -> Disable -> Enable -> Disable
        mock_controller.set_caps_lock_state(True)
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)

        mock_controller.set_caps_lock_state(False)
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)

        mock_controller.set_caps_lock_state(True)
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)

        mock_controller.set_caps_lock_state(False)
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)

        # Each callback should be called twice
        assert on_enabled.call_count == 2
        assert on_disabled.call_count == 2


class TestCapsLockCallbackErrorHandling:
    """Tests for error handling in Caps Lock callbacks."""

    def test_exception_in_on_enabled_callback_is_caught(
        self, mock_controller: MockKeyboardController
    ) -> None:
        """Test that exceptions in on_enabled callback are caught and logged."""
        on_enabled = MagicMock(side_effect=Exception("Test error"))
        on_disabled = MagicMock()

        mock_controller.register_caps_lock_toggle(on_enabled, on_disabled)

        # Enable Caps Lock - should not raise exception
        mock_controller.set_caps_lock_state(True)
        mock_controller._check_caps_lock_state(on_enabled, on_disabled)

        # Callback was called despite error
        on_enabled.assert_called_once()

        # State should still be updated
        assert mock_controller._caps_lock_active is True

    def test_exception_in_on_disabled_callback_is_caught(
        self, mock_controller_caps_on: MockKeyboardController
    ) -> None:
        """Test that exceptions in on_disabled callback are caught and logged."""
        on_enabled = MagicMock()
        on_disabled = MagicMock(side_effect=Exception("Test error"))

        mock_controller_caps_on.register_caps_lock_toggle(on_enabled, on_disabled)

        # Disable Caps Lock - should not raise exception
        mock_controller_caps_on.set_caps_lock_state(False)
        mock_controller_caps_on._check_caps_lock_state(on_enabled, on_disabled)

        # Callback was called despite error
        on_disabled.assert_called_once()

        # State should still be updated
        assert mock_controller_caps_on._caps_lock_active is False


class TestWindowsKeyboardController:
    """Tests for Windows-specific keyboard controller."""

    @patch("ctypes.windll.user32.GetKeyState")
    def test_is_caps_lock_on_returns_true_when_on(
        self, mock_get_key_state: MagicMock
    ) -> None:
        """Test that _is_caps_lock_on returns True when Caps Lock is on."""
        # GetKeyState returns odd number when Caps Lock is on (low-order bit is 1)
        mock_get_key_state.return_value = 1

        controller = WindowsKeyboardController()
        assert controller._is_caps_lock_on() is True

        mock_get_key_state.assert_called_once_with(WindowsKeyboardController.VK_CAPITAL)

    @patch("ctypes.windll.user32.GetKeyState")
    def test_is_caps_lock_on_returns_false_when_off(
        self, mock_get_key_state: MagicMock
    ) -> None:
        """Test that _is_caps_lock_on returns False when Caps Lock is off."""
        # GetKeyState returns even number when Caps Lock is off (low-order bit is 0)
        mock_get_key_state.return_value = 0

        controller = WindowsKeyboardController()
        assert controller._is_caps_lock_on() is False

        mock_get_key_state.assert_called_once_with(WindowsKeyboardController.VK_CAPITAL)

    @patch("ctypes.windll.user32.GetKeyState")
    def test_windows_controller_has_check_caps_lock_state_from_base(
        self, mock_get_key_state: MagicMock
    ) -> None:
        """Test that Windows controller inherits _check_caps_lock_state from base."""
        mock_get_key_state.return_value = 0

        controller = WindowsKeyboardController()

        # Should have the method from base class
        assert hasattr(controller, "_check_caps_lock_state")
        assert callable(controller._check_caps_lock_state)

        # Should be able to use it
        on_enabled = MagicMock()
        on_disabled = MagicMock()

        controller.register_caps_lock_toggle(on_enabled, on_disabled)
        controller._check_caps_lock_state(on_enabled, on_disabled)

        # No callbacks should be triggered (state unchanged)
        on_enabled.assert_not_called()
        on_disabled.assert_not_called()

