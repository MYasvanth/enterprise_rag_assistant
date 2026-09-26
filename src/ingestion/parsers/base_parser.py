from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseParser(ABC):
    """
    Abstract base class for document parsers.
    All parsers must return a List[Document] with metadata
    containing at least: source, page, format.
    """

    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """
        Parse file into list of Documents.
        Each Document has page_content (str) and metadata (dict).
        metadata must include: source, page (if applicable), format.
        Raises: ParseError if file cannot be read.
        """
        pass