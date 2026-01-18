"""
Document Processor

Handles parsing and chunking of PDF and Markdown documents.
Extracts text, metadata, and prepares chunks for embedding.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re

import fitz  # PyMuPDF
import tiktoken


class DocumentProcessor:
    """
    Processes documents (PDF, Markdown) into chunks for embedding.

    Features:
    - PDF text extraction with page tracking
    - Markdown parsing
    - Token-based chunking with overlap
    - Metadata extraction
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap tokens between chunks
            encoding_name: Tokenizer encoding (default for Claude/GPT-4)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def process_file(self, file_path: str) -> List[Dict]:
        """
        Process a single file and return chunks with metadata.

        Args:
            file_path: Path to the file

        Returns:
            List of chunk dictionaries with text and metadata

        Raises:
            ValueError: If file type not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type and process
        extension = path.suffix.lower()

        if extension == ".pdf":
            return self._process_pdf(path)
        elif extension in [".md", ".markdown", ".txt"]:
            return self._process_markdown(path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def _process_pdf(self, file_path: Path) -> List[Dict]:
        """
        Extract text from PDF and create chunks using PyMuPDF.

        Args:
            file_path: Path to PDF file

        Returns:
            List of chunks with metadata
        """
        chunks = []

        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(str(file_path))
            total_pages = len(doc)

            for page_num in range(total_pages):
                # Get page (0-indexed in PyMuPDF)
                page = doc[page_num]

                # Extract text from page
                text = page.get_text()

                if not text or not text.strip():
                    continue

                # Create chunks from page text
                page_chunks = self._chunk_text(text)

                # Add metadata to each chunk
                for chunk_idx, chunk_text in enumerate(page_chunks):
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": file_path.name,
                            "file_path": str(file_path.absolute()),
                            "file_type": "pdf",
                            "page": page_num + 1,  # 1-indexed for user display
                            "total_pages": total_pages,
                            "chunk_index": chunk_idx,
                            "timestamp": datetime.now().isoformat(),
                            "file_hash": self._compute_file_hash(file_path)
                        }
                    })

            # Close the PDF document
            doc.close()

        except Exception as e:
            raise RuntimeError(f"Error processing PDF {file_path}: {str(e)}")

        return chunks

    def _process_markdown(self, file_path: Path) -> List[Dict]:
        """
        Extract text from Markdown/text file and create chunks.

        Args:
            file_path: Path to markdown file

        Returns:
            List of chunks with metadata
        """
        chunks = []

        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                return chunks

            # Create chunks from text
            text_chunks = self._chunk_text(text)

            # Add metadata to each chunk
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": file_path.name,
                        "file_path": str(file_path.absolute()),
                        "file_type": file_path.suffix[1:],  # Remove dot
                        "chunk_index": chunk_idx,
                        "timestamp": datetime.now().isoformat(),
                        "file_hash": self._compute_file_hash(file_path)
                    }
                })

        except Exception as e:
            raise RuntimeError(f"Error processing file {file_path}: {str(e)}")

        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        # Encode text to tokens
        tokens = self.encoding.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            # Get chunk of tokens
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position with overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of file for change detection.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file hash
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))


# Utility functions

def process_directory(
    directory_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[Dict]:
    """
    Process all supported files in a directory.

    Args:
        directory_path: Path to directory
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Overlap tokens between chunks

    Returns:
        List of all chunks from all files
    """
    processor = DocumentProcessor(chunk_size, chunk_overlap)
    all_chunks = []

    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Supported extensions
    supported_extensions = {".pdf", ".md", ".markdown", ".txt"}

    # Find all supported files
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                chunks = processor.process_file(str(file_path))
                all_chunks.extend(chunks)
                print(f"Processed: {file_path.name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")

    return all_chunks


# TODO: Add support for more file types (DOCX, HTML, etc.)
# TODO: Add advanced chunking strategies (semantic chunking)
# TODO: Add text preprocessing (remove excessive whitespace, etc.)
