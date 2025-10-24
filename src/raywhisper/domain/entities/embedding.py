"""Embedding entity."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Embedding:
    """Embedding entity representing a vector embedding of text."""

    vector: np.ndarray
    text: str
    metadata: dict[str, str]

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vector.

        Returns:
            int: Dimension of the embedding.
        """
        return len(self.vector)

