"""Integration test for RAG context formatting with real transcription scenarios."""

import sys
from pathlib import Path

import pytest

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from raywhisper.application.services.rag_service import RAGService
from raywhisper.config.loader import load_settings
from raywhisper.domain.value_objects.context import SearchResult
from raywhisper.infrastructure.embeddings.reranker import CrossEncoderReranker
from raywhisper.infrastructure.vector_db.chroma_store import ChromaVectorStore


class TestRAGContextFormatting:
    """Test RAG context formatting for Whisper transcription improvement."""

    @pytest.fixture
    def settings(self):
        """Load settings from config."""
        return load_settings()

    @pytest.fixture
    def vector_store(self, settings):
        """Create vector store instance."""
        return ChromaVectorStore(
            collection_name=settings.vector_db.collection_name,
            persist_directory=settings.vector_db.persist_directory,
            embedding_model_name=settings.vector_db.embedding_model,
            chunk_size=settings.vector_db.chunk_size,
            chunk_overlap=settings.vector_db.chunk_overlap,
            use_query_instruction=settings.vector_db.use_query_instruction,
        )

    @pytest.fixture
    def reranker(self, settings):
        """Create reranker instance."""
        return CrossEncoderReranker(model_name=settings.reranker.model_name)

    @pytest.fixture
    def rag_service(self, vector_store, reranker, settings):
        """Create RAG service instance."""
        return RAGService(
            vector_store=vector_store,
            reranker=reranker,
            top_k=settings.vector_db.top_k,
            top_n=settings.reranker.top_n,
        )

    def test_rag_context_for_nuget_packages_transcription(self, rag_service):
        """Test RAG context retrieval and formatting for NuGet packages transcription.

        This test uses the actual transcription from the logs:
        "Let's test with a similar challenge again. Let's talk about top net mocking
        libraries like N substitute and test fixture setup that's called auto fixture
        and also Microsoft's HTTP client library called Richard Shalay mock HTTP."

        Expected corrections:
        - "N substitute" -> "NSubstitute"
        - "auto fixture" -> "AutoFixture"
        - "Richard Shalay mock HTTP" -> "RichardSzalay.MockHttp"
        """
        # The actual transcription from the logs (with errors)
        transcription = (
            "Let's test with a similar challenge again. Let's talk about top net mocking "
            "libraries like N substitute and test fixture setup that's called auto fixture "
            "and also Microsoft's HTTP client library called Richard Shalay mock HTTP."
        )

        # Retrieve context
        context = rag_service.retrieve_context(transcription)

        # Assertions about context format
        assert context, "Context should not be empty"
        assert len(context) > 0, "Context should have content"

        # The context should contain the exact technical terms
        # These are the correct spellings that should guide Whisper
        expected_terms = ["NSubstitute", "AutoFixture", "RichardSzalay.MockHttp"]

        # Check that at least some of the expected terms are in the context
        found_terms = [term for term in expected_terms if term in context]
        assert len(found_terms) > 0, (
            f"Context should contain at least one of the expected terms: {expected_terms}. "
            f"Found: {found_terms}. Context: {context}"
        )

        # The context should NOT contain markdown formatting that confuses Whisper
        assert "**" not in context, "Context should not contain markdown bold formatting"
        assert "[" not in context or "]" not in context, (
            "Context should not contain markdown link formatting"
        )

        # The context should be concise (Whisper's initial_prompt works best with short context)
        assert len(context) < 500, (
            f"Context should be concise (<500 chars) for Whisper. Got {len(context)} chars"
        )

        # Print context for manual inspection
        print(f"\n{'='*80}")
        print(f"Transcription (with errors):\n{transcription}")
        print(f"\n{'='*80}")
        print(f"RAG Context (for Whisper initial_prompt):\n{context}")
        print(f"\n{'='*80}")
        print(f"Expected terms: {expected_terms}")
        print(f"Found terms: {found_terms}")
        print(f"Context length: {len(context)} characters")

    def test_rag_context_contains_technical_terms_section(self, rag_service):
        """Test that RAG context starts with 'Technical terms:' section."""
        transcription = "NSubstitute and AutoFixture are great testing libraries"

        context = rag_service.retrieve_context(transcription)

        # The new format should start with "Technical terms:"
        assert context.startswith("Technical terms:"), (
            f"Context should start with 'Technical terms:'. Got: {context[:50]}"
        )

    def test_rag_context_extracts_unique_terms(self, rag_service):
        """Test that RAG context extracts unique technical terms from retrieved chunks."""
        transcription = "I use Moq and NSubstitute for mocking"

        context = rag_service.retrieve_context(transcription)

        # Should contain the technical terms
        assert "Moq" in context or "NSubstitute" in context, (
            f"Context should contain mocking library names. Got: {context}"
        )

    def test_rag_context_format_is_whisper_friendly(self, rag_service):
        """Test that RAG context format is optimized for Whisper's initial_prompt."""
        transcription = "AutoFixture helps with test data generation"

        context = rag_service.retrieve_context(transcription)

        # Should be plain text without complex formatting
        forbidden_chars = ["**", "###", "```", "![", "]("]
        for char_seq in forbidden_chars:
            assert char_seq not in context, (
                f"Context should not contain markdown formatting '{char_seq}'. Got: {context}"
            )

        # Should be readable and concise
        assert len(context.split()) < 100, (
            f"Context should be concise (<100 words). Got {len(context.split())} words"
        )

    @pytest.mark.parametrize(
        "query,expected_term",
        [
            ("N substitute mocking library", "NSubstitute"),
            ("auto fixture test data", "AutoFixture"),
            ("Richard Shalay mock HTTP", "RichardSzalay.MockHttp"),
            ("entity framework core", "Entity Framework Core"),
            ("dapper micro ORM", "Dapper"),
        ],
    )
    def test_rag_retrieves_correct_terms_for_common_misheard_phrases(
        self, rag_service, query, expected_term
    ):
        """Test that RAG retrieves correct technical terms for commonly misheard phrases."""
        context = rag_service.retrieve_context(query)

        assert expected_term in context, (
            f"Context should contain '{expected_term}' for query '{query}'. "
            f"Got: {context}"
        )


class TestRAGServiceWithMockData:
    """Test RAG service context formatting with mock search results."""

    def test_context_formatting_with_header_chunks(self):
        """Test context formatting when search results include header chunks."""
        from unittest.mock import Mock

        # Create mock search results
        mock_results = [
            SearchResult(
                content="NSubstitute Friendly substitute for .NET mocking libraries",
                score=0.95,
                metadata={
                    "header_raw": "NSubstitute",
                    "type": "markdown_header",
                    "header": "### NSubstitute",
                },
            ),
            SearchResult(
                content="AutoFixture Automates non-relevant Test Fixture Setup for TDD",
                score=0.92,
                metadata={
                    "header_raw": "AutoFixture",
                    "type": "markdown_header",
                    "header": "### AutoFixture",
                },
            ),
            SearchResult(
                content="RichardSzalay.MockHttp Testing layer for Microsoft's HttpClient library",
                score=0.88,
                metadata={
                    "header_raw": "RichardSzalay.MockHttp",
                    "type": "markdown_header",
                    "header": "### RichardSzalay.MockHttp",
                },
            ),
        ]

        # Create mock vector store and reranker
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_results

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = mock_results

        # Create RAG service
        rag_service = RAGService(
            vector_store=mock_vector_store,
            reranker=mock_reranker,
            top_k=5,
            top_n=3,
        )

        # Get context
        context = rag_service.retrieve_context("test query")

        # Verify format - should extract technical terms from content
        assert "Technical terms:" in context
        # Top terms should include NSubstitute and AutoFixture (from headers)
        assert "NSubstitute" in context
        assert "AutoFixture" in context
        # Note: RichardSzalay.MockHttp may not be in top 8 due to relevance ranking

        # Should not have markdown formatting
        assert "**" not in context
        assert "###" not in context

        # Should be concise
        assert len(context) < 500

        print(f"\nFormatted context:\n{context}")

    def test_context_formatting_extracts_technical_terms_from_content(self):
        """Test that technical terms are extracted from content using heuristics."""
        from unittest.mock import Mock

        # Create mock search results with technical terms in the content
        mock_results = [
            SearchResult(
                content="Vector Store Technologies FAISS Facebook AI Similarity Search In-memory CPU GPU",
                score=0.98,
                metadata={
                    "header_raw": "Vector Store Technologies",
                    "type": "markdown",
                    "header": "## Vector Store Technologies",
                },
            ),
            SearchResult(
                content="Pinecone Cloud-native vector database",
                score=0.96,
                metadata={
                    "header_raw": "Vector Store Technologies",
                    "type": "markdown",
                    "header": "## Vector Store Technologies",
                },
            ),
            SearchResult(
                content="Weaviate Open-source vector database Milvus Distributed",
                score=0.94,
                metadata={
                    "header_raw": "Vector Store Technologies",
                    "type": "markdown",
                    "header": "## Vector Store Technologies",
                },
            ),
        ]

        # Create mock vector store and reranker
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = mock_results

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = mock_results

        # Create RAG service
        rag_service = RAGService(
            vector_store=mock_vector_store,
            reranker=mock_reranker,
            top_k=5,
            top_n=3,
        )

        # Get context
        context = rag_service.retrieve_context("pinecote beviate mills")

        # Verify format - should use "Technical terms:" for extracted technical terms
        assert "Technical terms:" in context, f"Expected 'Technical terms:' in context. Got: {context}"

        # Should extract technical terms from content (FAISS, Pinecone, Weaviate, Milvus, etc.)
        # At least some of these should be present
        technical_terms_found = sum([
            "FAISS" in context,
            "Pinecone" in context,
            "Weaviate" in context,
            "Milvus" in context,
        ])
        assert technical_terms_found >= 2, f"Should extract at least 2 technical terms. Got: {context}"

        # Should not have markdown formatting
        assert "**" not in context
        assert "###" not in context

        # Should be concise
        assert len(context) < 500

        print(f"\nFormatted context with extracted technical terms:\n{context}")

