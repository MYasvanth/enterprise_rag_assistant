from typing import List

from langchain_core.documents import Document

from .base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Splits documents where embedding similarity between adjacent sentences
    drops below a threshold — indicating a topic/meaning shift.
    """

    def __init__(self, embedding_model, threshold: float = 0.3):
        self.embedding_model = embedding_model
        self.threshold = threshold  # cosine distance threshold for split

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Splits where embedding similarity between adjacent sentences drops
        below threshold — indicates topic/meaning shift.
        """
        from nltk.tokenize import sent_tokenize
        import numpy as np

        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

        chunks = []
        for doc in documents:
            sentences = sent_tokenize(doc.page_content)
            if len(sentences) <= 1:
                chunks.append(doc)
                continue
            embeddings = self.embedding_model.encode(sentences)
            current_group = [sentences[0]]
            for i in range(1, len(sentences)):
                sim = cosine_similarity(
                    [embeddings[i - 1]], [embeddings[i]]
                )
                if sim < self.threshold:
                    # Meaning shifted — start new chunk
                    chunks.append(Document(
                        page_content=" ".join(current_group),
                        metadata=doc.metadata
                    ))
                    current_group = [sentences[i]]
                else:
                    current_group.append(sentences[i])
            if current_group:
                chunks.append(Document(
                    page_content=" ".join(current_group),
                    metadata=doc.metadata
                ))
        return chunks