from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.
    All chunkers must split a List[Document] into smaller chunks,
    preserving metadata from the parent document.
    """

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        Each chunk is a Document with page_content <= chunk_size.
        metadata preserved from parent document.
        """
        pass