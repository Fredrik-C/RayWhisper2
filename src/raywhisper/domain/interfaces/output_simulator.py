"""Output simulator interface."""

from abc import ABC, abstractmethod


class IOutputSimulator(ABC):
    """Interface for output simulation (typing text)."""

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Simulate typing the given text.

        Args:
            text: The text to type.
        """
        pass

