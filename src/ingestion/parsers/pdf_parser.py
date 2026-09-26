from typing import List

import pdfplumber
from langchain_core.documents import Document

from .base_parser import BaseParser


class PDFParser(BaseParser):
    """
    Table-aware PDF parser using pdfplumber.

    Uses pdfplumber instead of PyPDFLoader to preserve table structures
    by converting them to pipe-delimited text. Each page becomes one
    Document with metadata containing source, page, and format.
    """

    def parse(self, file_path: str) -> List[Document]:
        """
        Parse file into list of Documents.
        Each page becomes one Document with combined raw text and
        formatted table content.
        """
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                table_text = self._format_tables(tables)
                raw_text = page.extract_text() or ""
                content = f"{raw_text}\n{table_text}".strip()

                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                            "format": "pdf"
                        }
                    ))
        return documents

    def _format_tables(self, tables: List) -> str:
        """
        Convert pdfplumber table rows to pipe-delimited text.
        Preserves table structure for downstream chunking.
        """
        if not tables:
            return ""
        rows = []
        for table in tables:
            for row in table:
                cleaned = [str(cell or "").strip() for cell in row]
                rows.append("| " + " | ".join(cleaned) + " |")
        return "\n".join(rows)