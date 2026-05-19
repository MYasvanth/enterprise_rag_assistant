"""
Document Ingestion and Processing Pipeline
Handles loading, chunking, and preprocessing of various document formats.
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    """Pipeline for ingesting and processing documents."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, source_path: str) -> List[Document]:
        """Load documents from various sources."""
        documents = []

        if os.path.isfile(source_path):
            documents = self._load_single_file(source_path)
        elif os.path.isdir(source_path):
            documents = self._load_directory(source_path)
        else:
            raise ValueError(f"Invalid source path: {source_path}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def _load_single_file(self, file_path: str) -> List[Document]:
        """Load a single file."""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        else:
            # Use unstructured for other formats
            elements = partition(file_path)
            documents = []
            for element in elements:
                documents.append(Document(
                    page_content=str(element),
                    metadata={"source": file_path}
                ))
            return documents

        return loader.load()

    def _load_directory(self, dir_path: str) -> List[Document]:
        """Load all documents from a directory."""
        loader = DirectoryLoader(
            dir_path,
            glob="**/*.*",
            loader_cls=self._get_loader_for_file
        )
        return loader.load()

    def _get_loader_for_file(self, file_path: str):
        """Get appropriate loader for file type."""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension in ['.docx', '.doc']:
            return Docx2txtLoader(file_path)
        elif file_extension == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            return TextLoader(file_path)

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents: clean, chunk, and add metadata."""
        processed_docs = []

        for doc in documents:
            # Clean and preprocess text
            cleaned_content = self._clean_text(doc.page_content)

            # Create new document with cleaned content
            processed_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )

            processed_docs.append(processed_doc)

        # Split into chunks
        chunks = self.text_splitter.split_documents(processed_docs)

        logger.info(f"Processed into {len(chunks)} chunks")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text content."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
        return text.strip()

    def run_pipeline(self, source_path: str) -> List[Document]:
        """Run the complete ingestion pipeline."""
        logger.info(f"Starting document ingestion from {source_path}")

        # Load documents
        documents = self.load_documents(source_path)

        # Process documents
        processed_chunks = self.process_documents(documents)

        logger.info(f"Pipeline completed. Generated {len(processed_chunks)} chunks")
        return processed_chunks
