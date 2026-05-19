"""
Retrieval and Generation Pipeline
Handles query processing, retrieval from vector store, and LLM generation.
"""

import os
from typing import List, Dict, Any, Optional
import logging

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback

from ..embedding.embedding import EmbeddingManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self,
                 embedding_manager: EmbeddingManager,
                 llm_provider: str = "openai",
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo"):
        self.embedding_manager = embedding_manager
        self.llm_provider = llm_provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Initialize QA chain
        self.qa_chain = None

    def _initialize_llm(self):
        """Initialize the language model."""
        if self.llm_provider == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key required")
            return ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self.api_key,
                temperature=0.1
            )
        elif self.llm_provider == "huggingface":
            # For local models
            raise NotImplementedError("HuggingFace pipeline not implemented yet")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def create_qa_chain(self):
        """Create the retrieval QA chain using LCEL."""
        if not self.embedding_manager.vector_store:
            raise ValueError("Vector store not initialized")

        # Custom prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create retriever
        retriever = self.embedding_manager.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

        # Create the RAG chain using LCEL
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info("QA chain created")
        return self.qa_chain

    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")

        logger.info(f"Processing query: {question}")

        # Get relevant documents first
        retriever = self.embedding_manager.vector_store.as_retriever(search_kwargs={"k": 5})
        source_docs = retriever.get_relevant_documents(question)

        # Track token usage for OpenAI
        with get_openai_callback() as cb:
            answer = self.qa_chain.invoke(question)

        response = {
            "answer": answer,
            "source_documents": source_docs,
            "token_usage": {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost
            } if hasattr(cb, 'total_tokens') else None
        }

        logger.info(f"Query processed. Token usage: {response['token_usage']}")
        return response

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents for a query without generation."""
        return self.embedding_manager.similarity_search(query, k=k)

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer from query and context documents."""
        # Combine context
        context = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""

        return self.llm(prompt)
