from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Recursive character text splitter that splits on sentence/paragraph
    boundaries with overlap to preserve context across chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)