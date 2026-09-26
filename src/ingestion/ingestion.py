"""
Document Ingestion and Processing Pipeline
Handles loading, chunking, and preprocessing of various document formats.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from langchain_core.documents import Document
from unstructured.partition.auto import partition

from .parsers.base_parser import BaseParser
from .parsers.pdf_parser import PDFParser
from .parsers.ocr_parser import OCRParser
from .parsers.docx_parser import DocxParser
from .parsers.text_parser import TextParser, MarkdownParser
from .chunkers.base_chunker import BaseChunker
from .chunkers.recursive_chunker import RecursiveChunker

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting and processing documents."""

    def __init__(
        self,
        chunker: BaseChunker = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunker = chunker or RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self._parsers = {
            ".pdf": PDFParser(),
            ".docx": DocxParser(),
            ".txt": TextParser(),
            ".md": MarkdownParser(),
        }

    def run_pipeline(self, file_path: str) -> List[Document]:
        """
        Full ETL: load → parse → clean → chunk.
        Accepts a single file path OR a directory (all supported files in it).
        Returns List[Document] (chunks ready for embedding).
        Raises: ParseError, ChunkError.
        """
        if os.path.isdir(file_path):
            documents = self._load_directory(file_path)
        else:
            documents = self._load(file_path)
        cleaned = [self._clean(doc) for doc in documents]
        chunks = self.chunker.chunk(cleaned)
        return chunks

    def set_chunk_params(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Update chunking parameters at runtime.

        Rebuilds the chunker when it is the default RecursiveChunker.
        A custom chunker (e.g. SemanticChunker) is left untouched, since
        its parameters are set at construction.
        """
        if isinstance(self.chunker, RecursiveChunker):
            self.chunker = RecursiveChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        else:
            logger.warning(
                "set_chunk_params ignored: custom chunker %s in use",
                type(self.chunker).__name__,
            )

    def _load(self, file_path: str) -> List[Document]:
        """Select parser by extension. Fall back to unstructured."""
        ext = Path(file_path).suffix.lower()
        parser = self._parsers.get(ext)
        if parser is None:
            return self._unstructured_fallback(file_path)
        # Auto-detect scanned PDF
        if ext == ".pdf" and OCRParser.is_scanned(file_path):
            parser = OCRParser()
        return parser.parse(file_path)

    def _clean(self, doc: Document) -> Document:
        """
        Collapse whitespace. Strip non-printable chars.
        Preserve \n and \t (needed for table structure).
        """
        import re
        text = doc.page_content
        # Strip non-printable chars except \n and \t
        text = "".join(c for c in text if c.isprintable() or c in "\n\t")
        # Collapse runs of spaces/tabs (but not newlines) to a single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive blank lines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        return Document(page_content=text.strip(), metadata=doc.metadata)

    def _unstructured_fallback(self, file_path: str) -> List[Document]:
        elements = partition(file_path)
        return [Document(page_content=str(e), metadata={"source": file_path})
                for e in elements]

    # Legacy methods preserved for backward compatibility
    def load_documents(self, source_path: str) -> List[Document]:
        """Load documents from various sources."""
        documents = []

        if os.path.isfile(source_path):
            documents = self._load(source_path)
        elif os.path.isdir(source_path):
            documents = self._load_directory(source_path)
        else:
            raise ValueError(f"Invalid source path: {source_path}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def _load_directory(self, dir_path: str) -> List[Document]:
        """Load all documents from a directory."""
        documents = []
        for file_path in Path(dir_path).rglob("*"):
            if file_path.is_file():
                try:
                    documents.extend(self._load(str(file_path)))
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents: clean, chunk, and add metadata."""
        cleaned = [self._clean(doc) for doc in documents]
        chunks = self.chunker.chunk(cleaned)
        logger.info(f"Processed into {len(chunks)} chunks")
        return chunks
