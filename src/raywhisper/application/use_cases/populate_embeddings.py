"""Populate embeddings use case."""

from pathlib import Path

from loguru import logger

from ...domain.interfaces.vector_store import IVectorStore
from ...infrastructure.embeddings.document_parser import DocumentParser


class PopulateEmbeddingsUseCase:
    """Use case for populating the vector database with document embeddings."""

    def __init__(
        self,
        vector_store: IVectorStore,
        chunk_size: int = 3,
        chunk_overlap: int = 2,
    ) -> None:
        """Initialize the use case.

        Args:
            vector_store: The vector store to populate.
            chunk_size: Number of words per chunk for document parsing.
            chunk_overlap: Number of overlapping words between chunks.
        """
        self._vector_store = vector_store
        self._parser = DocumentParser()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def execute(self, source_directories: list[Path], clear_existing: bool = False) -> int:
        """Execute the populate embeddings use case.

        Args:
            source_directories: List of directories to scan for documents.
            clear_existing: Whether to clear existing embeddings before populating.

        Returns:
            int: Total number of documents added.
        """
        if clear_existing:
            logger.warning("Clearing existing embeddings")
            self._vector_store.clear()

        total_docs = 0

        for directory in source_directories:
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory}")
                continue

            if not directory.is_dir():
                logger.warning(f"Not a directory: {directory}")
                continue

            logger.info(f"Processing directory: {directory}")

            # Find all markdown files
            md_files = list(directory.rglob("*.md"))

            logger.info(f"Found {len(md_files)} markdown files")

            # Process markdown files
            for md_file in md_files:
                try:
                    docs = self._parser.parse_markdown(
                        md_file,
                        chunk_size=self._chunk_size,
                        chunk_overlap=self._chunk_overlap,
                    )
                    if docs:
                        self._vector_store.add_documents(docs)
                        total_docs += len(docs)
                        logger.debug(f"Added {len(docs)} chunks from {md_file}")
                except Exception as e:
                    logger.error(f"Failed to process {md_file}: {e}")

        logger.info(f"Successfully added {total_docs} documents to vector database")
        return total_docs

