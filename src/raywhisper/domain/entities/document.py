"""Document entity."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Document entity representing a text document for embedding."""

    content: str
    source_path: Path
    metadata: dict[str, str]

    @property
    def file_type(self) -> str:
        """Get the file type from the source path.

        Returns:
            str: File extension (e.g., '.md', '.cs').
        """
        return self.source_path.suffix

