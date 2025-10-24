"""Tests for Document entity."""

from pathlib import Path

from raywhisper.domain.entities.document import Document


def test_document_creation() -> None:
    """Test creating a document."""
    doc = Document(
        content="Test content",
        source_path=Path("test.md"),
        metadata={"type": "markdown"},
    )

    assert doc.content == "Test content"
    assert doc.source_path == Path("test.md")
    assert doc.metadata == {"type": "markdown"}


def test_document_file_type_markdown() -> None:
    """Test file type property for markdown."""
    doc = Document(
        content="Test content",
        source_path=Path("test.md"),
        metadata={"type": "markdown"},
    )

    assert doc.file_type == ".md"


def test_document_file_type_no_extension() -> None:
    """Test file type property for file without extension."""
    doc = Document(
        content="Test content",
        source_path=Path("README"),
        metadata={"type": "text"},
    )

    assert doc.file_type == ""

