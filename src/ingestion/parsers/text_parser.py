from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

from .base_parser import BaseParser


class TextParser(BaseParser):
    """
    Basic plain-text parser using LangChain's TextLoader.
    """

    def parse(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        documents = loader.load()
        # Ensure the BaseParser metadata contract (source + format) is met.
        for doc in documents:
            doc.metadata.setdefault("format", "txt")
        return documents


class MarkdownParser(BaseParser):
    """
    Markdown parser using LangChain's UnstructuredMarkdownLoader,
    which preserves heading and element structure.
    """

    def parse(self, file_path: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        # Ensure the BaseParser metadata contract (source + format) is met.
        for doc in documents:
            doc.metadata.setdefault("format", "md")
        return documents


__all__ = ["TextParser", "MarkdownParser"]