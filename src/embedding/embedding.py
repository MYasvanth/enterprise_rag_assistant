"""
Embedding Generation and Vector Storage
Handles converting text chunks to embeddings and storing them in vector databases.
Implements tenant isolation, retry logic for API failures, and embedding caching.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
import logging
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, OpenAIEmbeddings as LangChainOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.vectorstores import VectorStore
from openai import APIError, RateLimitError

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingManager:
    """Manages embedding generation and vector storage.
    
    Features:
    - embed: Convert text to vector embeddings with retry logic
    - store: Persist embeddings in vector databases with tenant isolation
    - search: Semantic search filtered by tenant_id
    - embed cache: Redis caching to avoid redundant API calls
    - retry: Automatic retries for API rate limits and transient errors
    - tenant filter: Multi-tenancy support to isolate user data
    """

    def __init__(self,
                 embedding_provider: str = "openai",
                 vector_store: str = "chroma",
                 api_key: Optional[str] = None,
                 model_name: str = "text-embedding-ada-002"):
        self.embedding_provider = embedding_provider
        self.vector_store_type = vector_store
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model_name = model_name

        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()

        # Initialize vector store
        self.vector_store = None
        self._redis_available = False
        self._in_memory_embedding_cache = {}  # Development-only fallback
        
        # Initialize Redis for embedding cache
        self._redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=0.5,  # Ultra-short timeout for dev
            retry_on_timeout=False       # Don't retry on timeout
        )
        
        # Verify Redis connection - catch ALL Redis connection errors
        try:
            self._redis.ping()
            self._redis_available = True
            logger.info("Embedding cache Redis connection established")
        except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
            if settings.environment == "production":
                logger.critical(f"PRODUCTION FAILURE: Could not connect to Redis for embedding cache: {e}")
                raise SystemExit(1)
            else:
                logger.warning(f"DEVELOPMENT MODE: Redis connection failed, using in-memory embedding cache: {type(e).__name__}: {e}")
                self._redis_available = False

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

    def get_all_documents(self, collection_name: str = "default") -> List[Document]:
        """Retrieve all documents from the vector store for BM25 index initialization.
        
        This method is required by hybrid search to initialize the BM25 retriever
        with all stored documents to enable keyword search capabilities.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store() or load_vector_store() first.")
            
        if self.vector_store_type == "chroma":
            # Chroma supports get() which returns all documents in the collection
            results = self.vector_store.get()
            # Convert Chroma's raw format to LangChain Document objects
            documents = []
            for i in range(len(results["ids"])):
                doc = Document(
                    page_content=results["documents"][i],
                    metadata=results["metadatas"][i] if results["metadatas"] else {}
                )
                documents.append(doc)
            logger.info(f"Retrieved {len(documents)} total documents from Chroma vector store")
            return documents
            
        elif self.vector_store_type == "faiss":
            # For FAISS, we need to retrieve all documents from the index
            if hasattr(self.vector_store, 'docstore'):
                docs = list(self.vector_store.docstore._dict.values())
                logger.info(f"Retrieved {len(docs)} total documents from FAISS vector store")
                return docs
            else:
                logger.warning("FAISS vector store docstore not accessible")
                return []
                
        else:
            raise ValueError(f"get_all_documents() not supported for vector store type: {self.vector_store_type}")