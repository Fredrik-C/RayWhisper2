"""Tests for DocumentParser."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from raywhisper.infrastructure.embeddings.document_parser import DocumentParser


def test_parse_markdown_word_chunks() -> None:
    """Test parsing markdown into word-level chunks."""
    # Create a temporary markdown file
    content = """# Test Header

This is a test document with some words.

## Another Section

More text here for testing."""

    with NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        # Parse with default chunk size (3 words, 2 overlap)
        docs = DocumentParser.parse_markdown(temp_path, chunk_size=3, chunk_overlap=2)

        # Should create multiple small chunks
        assert len(docs) > 0

        # Check metadata - all docs should have these fields
        assert all("header" in doc.metadata for doc in docs)
        assert all("chunk_index" in doc.metadata for doc in docs)
        assert all("word_count" in doc.metadata for doc in docs)

        # Check that we have different types of chunks
        types = {doc.metadata["type"] for doc in docs}
        assert "markdown" in types or "markdown_header" in types

    finally:
        temp_path.unlink()


def test_parse_markdown_single_word_chunks() -> None:
    """Test parsing markdown into single-word chunks."""
    content = """# Header

One two three four five"""

    with NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        # Parse with chunk size 1, no overlap
        docs = DocumentParser.parse_markdown(temp_path, chunk_size=1, chunk_overlap=0)

        # Should create one chunk per word
        assert len(docs) == 5

        # Each chunk should have exactly 1 word
        for doc in docs:
            assert len(doc.content.split()) == 1

        # Check the words
        words = [doc.content for doc in docs]
        assert words == ["One", "two", "three", "four", "five"]

    finally:
        temp_path.unlink()


def test_parse_markdown_with_overlap() -> None:
    """Test that overlapping chunks work correctly."""
    content = """# Test

One two three four five"""

    with NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        # Parse with chunk size 3, overlap 2
        docs = DocumentParser.parse_markdown(temp_path, chunk_size=3, chunk_overlap=2)

        # Extract chunk contents
        chunks = [doc.content for doc in docs]

        # With chunk_size=3 and overlap=2, we advance by 1 word each time
        # Expected chunks:
        # - "One two three"
        # - "two three four"
        # - "three four five"
        # - "four five" (last chunk may be shorter)
        # - "five" (last chunk may be shorter)

        assert len(chunks) >= 3
        assert "One two three" in chunks
        assert "two three four" in chunks
        assert "three four five" in chunks

    finally:
        temp_path.unlink()


def test_parse_markdown_strips_formatting() -> None:
    """Test that markdown formatting is stripped from chunks."""
    content = """# Header

This is **bold** and *italic* and `code` text."""

    with NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        docs = DocumentParser.parse_markdown(temp_path, chunk_size=10, chunk_overlap=0)

        # Check that markdown symbols are removed
        all_content = " ".join([doc.content for doc in docs])
        assert "**" not in all_content
        assert "*" not in all_content
        assert "`" not in all_content

        # But the words should still be there
        assert "bold" in all_content
        assert "italic" in all_content
        assert "code" in all_content

    finally:
        temp_path.unlink()


def test_parse_markdown_preserves_headers() -> None:
    """Test that section headers are preserved in metadata."""
    content = """# First Header

Some text here.

## Second Header

More text here."""

    with NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        docs = DocumentParser.parse_markdown(temp_path, chunk_size=2, chunk_overlap=1)

        # Check that headers are in metadata
        headers = set(doc.metadata["header"] for doc in docs)
        assert "# First Header" in headers
        assert "## Second Header" in headers

    finally:
        temp_path.unlink()


def test_parse_markdown_empty_file() -> None:
    """Test parsing an empty markdown file."""
    content = ""

    with NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        docs = DocumentParser.parse_markdown(temp_path, chunk_size=3, chunk_overlap=2)
        assert len(docs) == 0

    finally:
        temp_path.unlink()


def test_parse_file_markdown() -> None:
    """Test parse_file with markdown extension."""
    content = """# Test

Some content here."""

    with NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        docs = DocumentParser.parse_file(temp_path, chunk_size=3, chunk_overlap=2)
        assert len(docs) > 0

    finally:
        temp_path.unlink()


def test_parse_file_unsupported() -> None:
    """Test parse_file with unsupported extension."""
    with NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Some text")
        temp_path = Path(f.name)

    try:
        docs = DocumentParser.parse_file(temp_path, chunk_size=3, chunk_overlap=2)
        assert len(docs) == 0

    finally:
        temp_path.unlink()

