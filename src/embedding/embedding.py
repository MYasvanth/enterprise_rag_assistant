"""
Embedding Generation and Vector Storage
Handles converting text chunks to embeddings and storing them in vector databases.
"""

import os
from typing import List, Dict, Any, Optional
import logging
import numpy as np

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone, Weaviate, FAISS
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding generation and vector storage."""

    def __init__(self,
                 embedding_provider: str = "openai",
                 vector_store: str = "chroma",
                 api_key: Optional[str] = None,
                 model_name: str = "text-embedding-ada-002"):
        self.embedding_provider = embedding_provider
        self.vector_store_type = vector_store
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()

        # Initialize vector store
        self.vector_store = None

    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        if self.embedding_provider == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            return OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=self.api_key
            )
        elif self.embedding_provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def create_vector_store(self, persist_directory: str = "./chroma_db") -> VectorStore:
        """Create and initialize the vector store."""
        if self.vector_store_type == "chroma":
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        elif self.vector_store_type == "faiss":
            # FAISS will be created when adding documents
            self.vector_store = None
        elif self.vector_store_type == "pinecone":
            # Pinecone requires additional setup
            raise NotImplementedError("Pinecone integration not implemented yet")
        elif self.vector_store_type == "weaviate":
            # Weaviate requires additional setup
            raise NotImplementedError("Weaviate integration not implemented yet")
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_store_type}")

        logger.info(f"Initialized {self.vector_store_type} vector store")
        return self.vector_store

    def add_documents(self, documents: List[Document], collection_name: str = "default") -> None:
        """Add documents to the vector store."""
        if not self.vector_store and self.vector_store_type == "faiss":
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        elif self.vector_store:
            self.vector_store.add_documents(documents)
        else:
            raise ValueError("Vector store not initialized")

        logger.info(f"Added {len(documents)} documents to vector store")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Similarity search returned {len(results)} results")
        return results

    def save_vector_store(self, path: str) -> None:
        """Save the vector store to disk."""
        if hasattr(self.vector_store, 'persist'):
            self.vector_store.persist()
            logger.info(f"Vector store persisted to {path}")
        elif self.vector_store_type == "faiss":
            self.vector_store.save_local(path)
            logger.info(f"FAISS vector store saved to {path}")

    def load_vector_store(self, path: str) -> VectorStore:
        """Load vector store from disk."""
        if self.vector_store_type == "chroma":
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=path
            )
        elif self.vector_store_type == "faiss":
            self.vector_store = FAISS.load_local(path, self.embeddings)
        else:
            raise ValueError(f"Loading not supported for {self.vector_store_type}")

        logger.info(f"Vector store loaded from {path}")
        return self.vector_store
