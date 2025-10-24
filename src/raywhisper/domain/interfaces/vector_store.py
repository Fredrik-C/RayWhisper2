"""Vector store interface."""

from abc import ABC, abstractmethod

from ..entities.document import Document
from ..value_objects.context import SearchResult


class IVectorStore(ABC):
    """Interface for vector database operations."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of search results.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store."""
        pass

