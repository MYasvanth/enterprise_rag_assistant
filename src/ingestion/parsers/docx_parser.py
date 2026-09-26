from typing import List

from docx import Document as DocxDocument
from langchain_core.documents import Document

from .base_parser import BaseParser


class DocxParser(BaseParser):
    """
    DOCX parser using python-docx.

    Preserves heading hierarchy by flushing accumulated text into a
    new Document whenever a Heading paragraph is encountered. The
    heading style name is recorded in the document metadata.
    """

    def parse(self, file_path: str) -> List[Document]:
        """
        Parse DOCX file into list of Documents, splitting on headings.
        Each section between headings becomes one Document.
        """
        doc = DocxDocument(file_path)
        documents = []
        current_text = []

        for para in doc.paragraphs:
            if para.style.name.startswith('Heading') and current_text:
                # Flush previous section
                documents.append(Document(
                    page_content="\n".join(current_text),
                    metadata={
                        "source": file_path,
                        "format": "docx",
                        "heading": para.style.name
                    }
                ))
                current_text = []
            current_text.append(para.text)

        if current_text:
            documents.append(Document(
                page_content="\n".join(current_text),
                metadata={"source": file_path, "format": "docx"}
            ))
        return documents