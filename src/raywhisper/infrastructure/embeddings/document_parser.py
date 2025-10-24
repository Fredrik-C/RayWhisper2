"""Document parser for extracting chunks from various file types."""

import re
from pathlib import Path

from loguru import logger

from ...domain.entities.document import Document


class DocumentParser:
    """Parser for extracting document chunks from files."""

    @staticmethod
    def parse_markdown(
        file_path: Path,
        chunk_size: int = 3,
        chunk_overlap: int = 2,
    ) -> list[Document]:
        """Parse markdown file into small word-based chunks for keyword/phrase matching.

        Args:
            file_path: Path to the markdown file.
            chunk_size: Number of words per chunk (default: 3).
            chunk_overlap: Number of overlapping words between chunks (default: 2).

        Returns:
            List of Document entities, one per word-chunk.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return []

        # Extract text content and track section headers for metadata
        sections: list[dict[str, str]] = []
        current_section_text: list[str] = []
        current_header = ""
        current_header_raw = ""  # Store original header with exact casing

        for line in content.split("\n"):
            if line.startswith("#"):
                # Save previous section
                if current_section_text:
                    sections.append(
                        {
                            "text": " ".join(current_section_text),
                            "header": current_header,
                            "header_raw": current_header_raw,
                        }
                    )
                # Start new section
                current_header = line.strip()
                # Extract header text without markdown symbols, preserving exact casing
                current_header_raw = re.sub(r"^#+\s*", "", line).strip()
                current_section_text = []
            else:
                # Remove markdown formatting but preserve technical terms
                # Be gentle with dots in technical names like "RichardSzalay.MockHttp"
                clean_line = re.sub(r"[*_`\[\]()]", " ", line)
                clean_line = re.sub(r"\s+", " ", clean_line).strip()
                if clean_line:
                    current_section_text.append(clean_line)

        # Save last section
        if current_section_text:
            sections.append(
                {
                    "text": " ".join(current_section_text),
                    "header": current_header,
                    "header_raw": current_header_raw,
                }
            )

        # Create word-based chunks from each section
        documents: list[Document] = []

        for section in sections:
            header_raw = section.get("header_raw", "")

            # Create a dedicated chunk for the header + description (high priority)
            # This ensures exact technical terms in headers are directly searchable
            if header_raw:
                # Combine header with beginning of section text (up to 15 words)
                section_words = section["text"].split()
                preview_text = " ".join(section_words[:15]) if section_words else ""
                header_chunk = f"{header_raw} {preview_text}".strip()

                documents.append(
                    Document(
                        content=header_chunk,
                        source_path=file_path,
                        metadata={
                            "header": section["header"],
                            "header_raw": header_raw,
                            "type": "markdown_header",
                            "source": str(file_path),
                            "chunk_index": -1,  # Special marker for header chunks
                            "word_count": len(header_chunk.split()),
                        },
                    )
                )

            # Split into words for sliding window chunks
            words = section["text"].split()

            if not words:
                continue

            # Create sliding window chunks with header context
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk_words = words[i : i + chunk_size]

                if not chunk_words:
                    continue

                # Include header name in chunk content for better context
                if header_raw:
                    chunk_content = f"{header_raw}: {' '.join(chunk_words)}"
                else:
                    chunk_content = " ".join(chunk_words)

                documents.append(
                    Document(
                        content=chunk_content,
                        source_path=file_path,
                        metadata={
                            "header": section["header"],
                            "header_raw": header_raw,
                            "type": "markdown",
                            "source": str(file_path),
                            "chunk_index": i,
                            "word_count": len(chunk_words),
                        },
                    )
                )

        logger.debug(
            f"Parsed {len(documents)} word-chunks from {file_path} "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )
        return documents

    @staticmethod
    def parse_file(
        file_path: Path,
        chunk_size: int = 3,
        chunk_overlap: int = 2,
    ) -> list[Document]:
        """Parse a file based on its extension.

        Args:
            file_path: Path to the file.
            chunk_size: Number of words per chunk (default: 3).
            chunk_overlap: Number of overlapping words between chunks (default: 2).

        Returns:
            List of Document entities.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".md":
            return DocumentParser.parse_markdown(file_path, chunk_size, chunk_overlap)
        else:
            logger.warning(f"Unsupported file type: {suffix} for {file_path}")
            return []

