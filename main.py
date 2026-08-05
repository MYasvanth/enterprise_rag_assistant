"""
Main entry point for the Enterprise RAG Knowledge Assistant
Provides command-line interface and orchestrates all components.
"""

import os
import logging
import argparse
from pathlib import Path

from src.ingestion.ingestion import DocumentIngestionPipeline
from src.embedding.embedding import EmbeddingManager
from src.retrieval.retrieval import RAGPipeline
from src.agent.agent import RAGAgent
from src.api.api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnterpriseRAGAssistant:
    """Main application class."""

    def __init__(self):
        self.ingestion_pipeline = None
        self.embedding_manager = None
        self.rag_pipeline = None
        self.rag_agent = None
        self.initialized = False

    def initialize(self,
                   embedding_provider: str = "openai",
                   vector_store: str = "chroma",
                   llm_provider: str = "openai",
                   api_key: str = None):
        """Initialize all components."""
        try:
            logger.info("Initializing Enterprise RAG Knowledge Assistant...")

            # Initialize ingestion pipeline
            self.ingestion_pipeline = DocumentIngestionPipeline()

            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                api_key=api_key
            )

            # Create vector store
            persist_dir = "./data/chroma_db"
            os.makedirs(persist_dir, exist_ok=True)
            self.embedding_manager.create_vector_store(persist_dir)

            # Initialize RAG pipeline
            self.rag_pipeline = RAGPipeline(
                embedding_manager=self.embedding_manager,
                llm_provider=llm_provider,
                api_key=api_key
            )

            # Create QA chain
            self.rag_pipeline.create_qa_chain()

            # Initialize agent
            self.rag_agent = RAGAgent(
                embedding_manager=self.embedding_manager,
                api_key=api_key,
            )
            self.rag_agent.initialize()

            self.initialized = True
            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    def ingest_documents(self, source_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Ingest documents from file or directory."""
        if not self.initialized:
            raise RuntimeError("Assistant not initialized")

        logger.info(f"Ingesting documents from {source_path}")

        # Update pipeline parameters
        self.ingestion_pipeline.chunk_size = chunk_size
        self.ingestion_pipeline.chunk_overlap = chunk_overlap

        # Process documents
        chunks = self.ingestion_pipeline.run_pipeline(source_path)

        # Add to vector store
        self.embedding_manager.add_documents(chunks)

        # Save vector store
        self.embedding_manager.save_vector_store("./data/chroma_db")

        logger.info(f"Successfully ingested {len(chunks)} chunks")
        return len(chunks)

    def query(self, question: str, use_agent: bool = False):
        """Query the knowledge base."""
        if not self.initialized:
            raise RuntimeError("Assistant not initialized")

        logger.info(f"Processing query: {question}")
        if use_agent:
            return self.rag_agent.run(question)
        return self.rag_pipeline.query(question)

    def save_state(self, path: str = "./data"):
        """Save the current state."""
        if not self.initialized:
            return

        os.makedirs(path, exist_ok=True)
        self.embedding_manager.save_vector_store(f"{path}/chroma_db")
        logger.info(f"State saved to {path}")

    def load_state(self, path: str = "./data"):
        """Load saved state."""
        if not os.path.exists(f"{path}/chroma_db"):
            logger.warning(f"No saved state found at {path}")
            return

        self.embedding_manager.load_vector_store(f"{path}/chroma_db")
        self.rag_pipeline.create_qa_chain()
        logger.info(f"State loaded from {path}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Enterprise RAG Knowledge Assistant")
    parser.add_argument("--mode", choices=["cli", "api", "ingest", "agent"], default="cli",
                       help="Run mode: cli, api, ingest, or agent (interactive agent mode)")
    parser.add_argument("--source", help="Source path for ingestion (file or directory)")
    parser.add_argument("--query", help="Query to process")
    parser.add_argument("--embedding-provider", default="openai", help="Embedding provider")
    parser.add_argument("--vector-store", default="chroma", help="Vector store type")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider")
    parser.add_argument("--api-key", help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Document chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and (args.embedding_provider == "openai" or args.llm_provider == "openai"):
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key")
        return

    if args.mode == "api":
        # Run the FastAPI server
        import uvicorn
        logger.info("Starting FastAPI server on http://localhost:8000")
        logger.info("API documentation available at http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)

    elif args.mode == "ingest":
        if not args.source:
            print("Error: --source required for ingest mode")
            return

        assistant = EnterpriseRAGAssistant()
        assistant.initialize(
            embedding_provider=args.embedding_provider,
            vector_store=args.vector_store,
            llm_provider=args.llm_provider,
            api_key=api_key
        )

        chunks = assistant.ingest_documents(
            args.source,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        print(f"Successfully ingested {chunks} chunks from {args.source}")
        assistant.save_state()

    elif args.mode == "cli":
        assistant = EnterpriseRAGAssistant()
        assistant.initialize(
            embedding_provider=args.embedding_provider,
            vector_store=args.vector_store,
            llm_provider=args.llm_provider,
            api_key=api_key
        )

        # Try to load existing state
        assistant.load_state()

        print("Enterprise RAG Knowledge Assistant")
        print("Commands: ingest <path>, query <question>, save, load, quit")

        while True:
            try:
                cmd = input("> ").strip()

                if cmd.startswith("ingest "):
                    path = cmd[7:].strip()
                    if os.path.exists(path):
                        chunks = assistant.ingest_documents(path)
                        print(f"Ingested {chunks} chunks")
                    else:
                        print(f"Path not found: {path}")

                elif cmd.startswith("query "):
                    question = cmd[6:].strip()
                    result = assistant.query(question)
                    print(f"\nAnswer: {result['answer']}\n")
                    print("Sources:")
                    for i, doc in enumerate(result['source_documents'][:3], 1):
                        print(f"{i}. {doc.page_content[:200]}...")

                elif cmd.startswith("agent "):
                    question = cmd[6:].strip()
                    result = assistant.query(question, use_agent=True)
                    print(f"\nAnswer: {result['answer']}\n")
                    steps = result.get('agent_steps', [])
                    if steps:
                        print(f"Agent used {len(steps)} reasoning step(s).")

                elif cmd == "save":
                    assistant.save_state()
                    print("State saved")

                elif cmd == "load":
                    assistant.load_state()
                    print("State loaded")

                elif cmd in ["quit", "exit", "q"]:
                    break

                else:
                    print("Unknown command. Use: ingest <path>, query <question>, agent <question>, save, load, quit")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("Goodbye!")

if __name__ == "__main__":
    main()
