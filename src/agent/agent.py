"""
ReAct Agent for Enterprise RAG Knowledge Assistant.
Wraps the existing RAG pipeline with multi-step reasoning capabilities.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from ..embedding.embedding import EmbeddingManager

logger = logging.getLogger(__name__)

_REACT_PROMPT = PromptTemplate.from_template("""You are an expert knowledge assistant with access to a document knowledge base.
Use the tools to find accurate answers. For complex questions, decompose them and search multiple times.

Tools available:
{tools}

Tool names: {tool_names}

Format:
Question: the input question
Thought: reason about what to do
Action: tool name
Action Input: input to the tool
Observation: tool result
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Final Answer: comprehensive answer with source references

Question: {input}
Thought: {agent_scratchpad}""")


class RAGAgent:
    """ReAct agent that orchestrates multi-step retrieval and reasoning over the knowledge base."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        max_iterations: int = 6,
    ):
        self.embedding_manager = embedding_manager
        self.llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            temperature=0.1,
        )
        self.max_iterations = max_iterations
        self._executor: Optional[AgentExecutor] = None

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _vector_search(self, query: str) -> str:
        """Search the vector store and return formatted chunks with sources."""
        docs = self.embedding_manager.similarity_search(query, k=5)
        if not docs:
            return "No relevant documents found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
        return "\n\n".join(parts)

    def _document_summarizer(self, query: str) -> str:
        """Retrieve and summarize documents most relevant to the query."""
        docs = self.embedding_manager.similarity_search(query, k=3)
        if not docs:
            return "No documents found to summarize."
        combined = "\n\n".join(doc.page_content for doc in docs)
        summary_prompt = (
            f"Summarize the following content concisely, focusing on: {query}\n\n{combined}"
        )
        return self.llm.invoke(summary_prompt).content

    def _query_decomposer(self, complex_query: str) -> str:
        """Break a complex question into focused sub-questions and answer each."""
        decompose_prompt = (
            f"Break this question into 2-4 focused sub-questions that together answer it fully.\n"
            f"Return only a numbered list.\n\nQuestion: {complex_query}"
        )
        sub_questions_text = self.llm.invoke(decompose_prompt).content
        lines = [l.strip() for l in sub_questions_text.splitlines() if l.strip()]

        results: List[str] = []
        for line in lines:
            # Strip leading numbering like "1." or "1)"
            q = line.lstrip("0123456789.)- ").strip()
            if not q:
                continue
            docs = self.embedding_manager.similarity_search(q, k=3)
            context = "\n".join(doc.page_content for doc in docs) if docs else "No context found."
            answer_prompt = f"Answer this question using the context below.\nQuestion: {q}\nContext: {context}"
            answer = self.llm.invoke(answer_prompt).content
            results.append(f"Q: {q}\nA: {answer}")

        return "\n\n".join(results) if results else "Could not decompose the query."

    def _answer_validator(self, query_and_answer: str) -> str:
        """
        Validate whether an answer actually addresses the question.
        Input format: 'QUESTION: <q> | ANSWER: <a>'
        Returns 'VALID', 'PARTIAL: <reason>', or 'INVALID: <reason> | SUGGESTION: <refined_query>'.
        """
        validate_prompt = (
            f"Evaluate if the answer fully addresses the question.\n"
            f"Respond with one of:\n"
            f"  VALID\n"
            f"  PARTIAL: <reason>\n"
            f"  INVALID: <reason> | SUGGESTION: <refined search query>\n\n"
            f"{query_and_answer}"
        )
        return self.llm.invoke(validate_prompt).content

    # ------------------------------------------------------------------
    # Agent setup
    # ------------------------------------------------------------------

    def _build_tools(self) -> List[Tool]:
        return [
            Tool(
                name="vector_search",
                func=self._vector_search,
                description=(
                    "Search the knowledge base for relevant document chunks. "
                    "Use for direct factual lookups. Input: a search query string."
                ),
            ),
            Tool(
                name="document_summarizer",
                func=self._document_summarizer,
                description=(
                    "Retrieve and summarize documents relevant to a topic. "
                    "Use when you need a condensed overview. Input: topic or question."
                ),
            ),
            Tool(
                name="query_decomposer",
                func=self._query_decomposer,
                description=(
                    "Break a complex multi-part question into sub-questions and answer each. "
                    "Use for comparative or analytical questions. Input: the full complex question."
                ),
            ),
            Tool(
                name="answer_validator",
                func=self._answer_validator,
                description=(
                    "Validate whether a retrieved answer actually addresses the question. "
                    "Use before giving a final answer if unsure. "
                    "Input format: 'QUESTION: <q> | ANSWER: <a>'"
                ),
            ),
        ]

    def initialize(self) -> None:
        """Build the ReAct agent executor."""
        if not self.embedding_manager.vector_store:
            raise ValueError("Vector store not initialized in EmbeddingManager.")

        tools = self._build_tools()
        agent = create_react_agent(llm=self.llm, tools=tools, prompt=_REACT_PROMPT)
        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            verbose=True,
        )
        logger.info("RAGAgent initialized with %d tools", len(tools))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the agent on a question.

        Returns a dict compatible with RAGPipeline.query() output:
            {
                "answer": str,
                "source_documents": List[Document],
                "agent_steps": list,
                "token_usage": None   # tracked externally if needed
            }
        """
        if self._executor is None:
            raise RuntimeError("RAGAgent not initialized. Call initialize() first.")

        logger.info("Agent processing: %s", question)

        try:
            result = self._executor.invoke({"input": question})
            answer = result.get("output", "Agent did not produce an answer.")
        except Exception as e:
            logger.error("Agent execution error: %s", e)
            raise

        # Fetch source docs for UI display (best-effort after agent run)
        source_docs = self.embedding_manager.similarity_search(question, k=5)

        return {
            "answer": answer,
            "source_documents": source_docs,
            "agent_steps": result.get("intermediate_steps", []),
            "token_usage": None,
        }
