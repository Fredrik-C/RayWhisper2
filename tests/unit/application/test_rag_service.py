"""Tests for RAG service."""

import pytest

from raywhisper.application.services.rag_service import RAGService


class TestTechnicalTermExtraction:
    """Test technical term extraction from text."""

    def test_extract_capitalized_words(self):
        """Test extraction of capitalized words (proper nouns)."""
        text = "Pinecone is a vector database. Weaviate is another option."
        terms = RAGService._extract_technical_terms(text)
        
        assert "Pinecone" in terms
        assert "Weaviate" in terms

    def test_extract_camelcase_terms(self):
        """Test extraction of CamelCase identifiers."""
        text = "Use AutoFixture for test data and NSubstitute for mocking."
        terms = RAGService._extract_technical_terms(text)
        
        assert "AutoFixture" in terms
        assert "NSubstitute" in terms

    def test_extract_dot_separated_identifiers(self):
        """Test extraction of dot-separated identifiers."""
        text = "RichardSzalay.MockHttp is great for testing HTTP clients."
        terms = RAGService._extract_technical_terms(text)
        
        assert "RichardSzalay.MockHttp" in terms

    def test_extract_acronyms(self):
        """Test extraction of all-caps acronyms."""
        text = "FAISS is a library for efficient similarity search. Use API for HTTP requests."
        terms = RAGService._extract_technical_terms(text)
        
        assert "FAISS" in terms
        assert "API" in terms
        assert "HTTP" in terms

    def test_filter_common_words(self):
        """Test that common English words are filtered out."""
        text = "The quick brown fox jumps over the lazy dog."
        terms = RAGService._extract_technical_terms(text)
        
        # Common words should be filtered
        assert "The" not in terms
        assert "This" not in terms

    def test_extract_mixed_content(self):
        """Test extraction from text with mixed technical and common content."""
        text = """
        Vector Store Technologies:
        - FAISS (Facebook AI Similarity Search): In-memory CPU/GPU
        - Pinecone: Cloud-native vector database
        - Weaviate: Open-source vector database
        - Chroma: Lightweight embedded vector database
        - Milvus: Distributed vector database
        """
        terms = RAGService._extract_technical_terms(text)
        
        # Should extract technical terms
        assert "FAISS" in terms
        assert "Pinecone" in terms
        assert "Weaviate" in terms
        assert "Chroma" in terms
        assert "Milvus" in terms
        
        # Should extract other capitalized technical words
        assert "Vector" in terms or "Store" in terms or "Technologies" in terms

    def test_extract_from_code_like_text(self):
        """Test extraction from code-like text."""
        text = "Use System.IO.File for file operations and System.Net.Http for HTTP."
        terms = RAGService._extract_technical_terms(text)
        
        # Should extract dot-separated identifiers
        assert "System.IO.File" in terms or "System.IO" in terms
        assert "System.Net.Http" in terms or "System.Net" in terms

    def test_empty_text(self):
        """Test extraction from empty text."""
        terms = RAGService._extract_technical_terms("")
        assert len(terms) == 0

    def test_no_technical_terms(self):
        """Test extraction from text with no technical terms."""
        text = "the quick brown fox jumps over the lazy dog"
        terms = RAGService._extract_technical_terms(text)

        # Should have very few or no terms (all lowercase, common words)
        assert len(terms) == 0 or all(term.islower() for term in terms)


class TestPhoneticMatching:
    """Test phonetic similarity matching."""

    def test_find_phonetic_matches_exact(self):
        """Test phonetic matching with exact matches."""
        query = "Pinecone Weaviate Milvus"
        terms = {"Pinecone", "Weaviate", "Milvus", "FAISS", "Chroma"}

        matches = RAGService._find_phonetic_matches(query, terms)

        # Should find all exact matches
        assert "Pinecone" in matches
        assert "Weaviate" in matches
        assert "Milvus" in matches

    def test_find_phonetic_matches_similar(self):
        """Test phonetic matching with similar terms."""
        query = "Pine Code Milves"
        terms = {"Pinecone", "Weaviate", "Milvus", "FAISS", "Chroma"}

        matches = RAGService._find_phonetic_matches(query, terms)

        # Should find phonetically similar terms
        assert "Pinecone" in matches  # "Pine Code" → "Pinecone"
        assert "Milvus" in matches    # "Milves" → "Milvus"

    def test_find_phonetic_matches_case_insensitive(self):
        """Test that phonetic matching is case-insensitive."""
        query = "nuget packages"
        terms = {"NuGet", "AutoFixture", "FluentValidation"}

        matches = RAGService._find_phonetic_matches(query, terms)

        # Should find NuGet despite case difference
        assert "NuGet" in matches

    def test_find_phonetic_matches_no_matches(self):
        """Test phonetic matching with no similar terms."""
        query = "completely different words"
        terms = {"Pinecone", "Weaviate", "Milvus"}

        matches = RAGService._find_phonetic_matches(query, terms)

        # Should find no matches or very few
        assert len(matches) <= 1  # Might have one weak match

    def test_find_phonetic_matches_limit(self):
        """Test that phonetic matching limits results."""
        query = "test"
        # Create many similar terms
        terms = {f"Test{i}" for i in range(20)}

        matches = RAGService._find_phonetic_matches(query, terms)

        # Should limit to top 10
        assert len(matches) <= 10

    def test_find_phonetic_matches_vv8_to_weaviate(self):
        """Test that VV8 matches Weaviate using phonetic substitutions."""
        query = "For vector database we should use VV8"
        terms = {"Pinecone", "Weaviate", "Milvus", "FAISS", "Chroma", "Elasticsearch"}

        matches = RAGService._find_phonetic_matches(query, terms)

        # VV8 should match Weaviate through phonetic substitutions
        # VV → W/weav, 8 → ate/iate
        assert "Weaviate" in matches, f"Expected 'Weaviate' in matches, got: {matches}"

    def test_find_phonetic_matches_lowercase_vv8(self):
        """Test that lowercase vv8 matches Weaviate."""
        query = "vector database vv8"
        terms = {"Pinecone", "Weaviate", "Milvus", "FAISS", "Chroma"}

        matches = RAGService._find_phonetic_matches(query, terms)

        # vv8 should match Weaviate
        assert "Weaviate" in matches, f"Expected 'Weaviate' in matches, got: {matches}"

    def test_find_phonetic_matches_nvv8(self):
        """Test that NVV8 matches Weaviate."""
        query = "Pinecone NVV8"
        terms = {"Pinecone", "Weaviate", "Milvus", "FAISS", "Chroma"}

        matches = RAGService._find_phonetic_matches(query, terms)

        # NVV8 should match Weaviate (with some tolerance for the N prefix)
        assert "Weaviate" in matches, f"Expected 'Weaviate' in matches, got: {matches}"
        assert "Pinecone" in matches  # Should also match Pinecone

    def test_find_phonetic_matches_viviate(self):
        """Test that viviate matches Weaviate."""
        query = "Pinecone and viviate"
        terms = {"Pinecone", "Weaviate", "Milvus", "FAISS", "Chroma"}

        matches = RAGService._find_phonetic_matches(query, terms)

        # viviate should match Weaviate (high string similarity)
        assert "Weaviate" in matches, f"Expected 'Weaviate' in matches, got: {matches}"
        assert "Pinecone" in matches

    def test_phonetic_substitutions_number_to_word(self):
        """Test that number-to-word substitutions work."""
        # Test that "8" can be substituted with "ate", "eight", "iate"
        variations = RAGService._apply_phonetic_substitutions("vv8")

        # Should include variations with 8 → ate, eight, iate
        assert any("ate" in v for v in variations), f"Expected 'ate' variation in {variations}"

    def test_phonetic_substitutions_vv_to_w(self):
        """Test that VV to W substitutions work."""
        variations = RAGService._apply_phonetic_substitutions("vv8")

        # Should include variations with vv → w
        assert any("w" in v for v in variations), f"Expected 'w' variation in {variations}"

    def test_adaptive_threshold_short_words(self):
        """Test that short words have lower thresholds."""
        # Very short word (3 chars)
        threshold_3 = RAGService._get_adaptive_threshold(3)
        assert threshold_3 == 0.35

        # Short word (5 chars)
        threshold_5 = RAGService._get_adaptive_threshold(5)
        assert threshold_5 == 0.45

        # Medium word (7 chars)
        threshold_7 = RAGService._get_adaptive_threshold(7)
        assert threshold_7 == 0.55

        # Long word (10 chars)
        threshold_10 = RAGService._get_adaptive_threshold(10)
        assert threshold_10 == 0.65

    def test_calculate_phonetic_similarity_with_substitutions(self):
        """Test phonetic similarity calculation with substitutions."""
        # VV8 should have high similarity to Weaviate after substitutions
        similarity = RAGService._calculate_phonetic_similarity("vv8", "weaviate")

        # Should be above the threshold for short words (0.35)
        assert similarity >= 0.35, f"Expected similarity >= 0.35, got {similarity}"

