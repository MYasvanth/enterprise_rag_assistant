from typing import List

import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

from .base_parser import BaseParser


class OCRParser(BaseParser):
    """
    OCR parser for scanned PDFs.

    Triggered when a PDF has no extractable text (scanned documents).
    Converts each page to an image and runs pytesseract OCR to extract
    text. Each page becomes one Document with format set to "pdf_ocr".
    """

    def parse(self, file_path: str) -> List[Document]:
        """
        Parse scanned PDF into list of Documents via OCR.
        Each page is converted to an image and processed with pytesseract.
        """
        documents = []
        images = convert_from_path(file_path)
        for page_num, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang="eng")
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_num + 1,
                        "format": "pdf_ocr"
                    }
                ))
        return documents

    @staticmethod
    def is_scanned(file_path: str) -> bool:
        """
        Returns True if PDF has no extractable text — needs OCR.
        A PDF is considered scanned if no page yields any extractable
        text via pdfplumber.
        """
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    return False
        return True