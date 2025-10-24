"""ChromaDB vector store implementation."""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer

from ...domain.entities.document import Document
from ...domain.interfaces.vector_store import IVectorStore
from ...domain.value_objects.context import SearchResult


class ChromaVectorStore(IVectorStore):
    """Vector store implementation using ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model_name: str,
        chunk_size: int = 3,
        chunk_overlap: int = 2,
        use_query_instruction: bool = False,
    ) -> None:
        """Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
            embedding_model_name: Name of the embedding model to use.
            chunk_size: Number of words per chunk for document parsing.
            chunk_overlap: Number of overlapping words between chunks.
            use_query_instruction: Whether to use instruction prefix for queries.
        """
        logger.info(f"Initializing ChromaDB at {persist_directory}")

        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Create embedded, local client
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        self._embedding_model_name = embedding_model_name
        self._collection_name = collection_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._use_query_instruction = use_query_instruction

        # Load sentence transformer for embeddings
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self._embedder = SentenceTransformer(embedding_model_name)

        # Get or create collection
        # Note: ChromaDB will use the default embedding function if we don't provide one
        # We'll handle embeddings manually for more control
        try:
            self._collection = self._client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
            logger.info(f"Created new collection: {collection_name}")

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        # Generate embeddings
        contents = [doc.content for doc in documents]
        embeddings = self._embedder.encode(
            contents,
            normalize_embeddings=True,
            show_progress_bar=len(contents) > 10,
        )

        # Prepare data for ChromaDB
        ids = [f"{doc.source_path}_{i}" for i, doc in enumerate(documents)]
        metadatas = [doc.metadata for doc in documents]

        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=contents,
            metadatas=metadatas,
        )

        logger.info(f"Successfully added {len(documents)} documents")

    def _encode_query(self, query: str) -> list[float]:
        """Encode a query string to an embedding.

        Args:
            query: The query string.

        Returns:
            List of floats representing the embedding.
        """
        # Conditionally use instruction prefix based on configuration
        if self._use_query_instruction:
            # For sentence-based semantic search
            instruction = "Represent this sentence for searching relevant passages: "
            query_text = instruction + query
        else:
            # For keyword/phrase-based search, use query directly
            query_text = query

        embedding = self._embedder.encode(
            query_text,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of search results.
        """
        logger.debug(f"Searching for: '{query[:50]}...' (top_k={top_k})")

        # Encode query
        query_embedding = self._encode_query(query)

        # Query collection
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Convert to SearchResult objects
        search_results: list[SearchResult] = []

        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB returns distances; convert to similarity scores
                # For cosine distance, similarity = 1 - distance
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = 1.0 - distance

                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

                search_results.append(
                    SearchResult(
                        content=doc,
                        score=score,
                        metadata=metadata,
                    )
                )

        logger.debug(f"Found {len(search_results)} results")
        return search_results

    def clear(self) -> None:
        """Clear all documents from the store."""
        logger.warning(f"Clearing collection: {self._collection_name}")

        try:
            self._client.delete_collection(name=self._collection_name)
            logger.info(f"Deleted collection: {self._collection_name}")

            # Recreate empty collection
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Recreated empty collection: {self._collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

