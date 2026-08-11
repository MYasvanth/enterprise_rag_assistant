"""
FastAPI REST API for the Enterprise RAG Knowledge Assistant
Provides endpoints for document ingestion, querying, and management.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..ingestion.ingestion import DocumentIngestionPipeline
from ..embedding.embedding import EmbeddingManager
from ..retrieval.retrieval import RAGPipeline
from ..agent.agent import RAGAgent

logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    token_usage: Optional[Dict[str, Any]] = None

class IngestionResponse(BaseModel):
    message: str
    chunks_processed: int

class HealthResponse(BaseModel):
    status: str
    vector_store_initialized: bool
    qa_chain_initialized: bool
    agent_initialized: bool

class AgentQueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    agent_steps: List[Any]

# Global instances (in production, use dependency injection)
ingestion_pipeline = None
embedding_manager = None
rag_pipeline = None
rag_agent = None

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Enterprise RAG Knowledge Assistant API",
        description="REST API for document ingestion and Q&A using RAG",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

app = create_app()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global ingestion_pipeline, embedding_manager, rag_pipeline

    try:
        # Initialize ingestion pipeline
        ingestion_pipeline = DocumentIngestionPipeline()

        # Initialize embedding manager
        embedding_manager = EmbeddingManager(
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            vector_store=os.getenv("VECTOR_STORE", "chroma"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Create vector store
        embedding_manager.create_vector_store("./chroma_db")

        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            embedding_manager=embedding_manager,
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        )

        # Create QA chain
        rag_pipeline.create_qa_chain()

        # Initialize agent
        rag_agent = RAGAgent(
            embedding_manager=embedding_manager,
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        )
        rag_agent.initialize()

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        vector_store_initialized=embedding_manager.vector_store is not None if embedding_manager else False,
        qa_chain_initialized=rag_pipeline.qa_chain is not None if rag_pipeline else False,
        agent_initialized=rag_agent is not None and rag_agent._executor is not None,
    )

@app.post("/ingest/file", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Ingest a single file."""
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Update pipeline parameters
        ingestion_pipeline.chunk_size = chunk_size
        ingestion_pipeline.chunk_overlap = chunk_overlap

        # Process the file
        chunks = ingestion_pipeline.run_pipeline(temp_path)

        # Add to vector store
        embedding_manager.add_documents(chunks)

        # Save vector store
        embedding_manager.save_vector_store("./chroma_db")

        # Clean up
        os.remove(temp_path)

        return IngestionResponse(
            message=f"Successfully ingested {file.filename}",
            chunks_processed=len(chunks)
        )

    except Exception as e:
        logger.error(f"Error ingesting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/directory", response_model=IngestionResponse)
async def ingest_directory(
    path: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Ingest all documents from a directory."""
    try:
        # Update pipeline parameters
        ingestion_pipeline.chunk_size = chunk_size
        ingestion_pipeline.chunk_overlap = chunk_overlap

        # Process the directory
        chunks = ingestion_pipeline.run_pipeline(path)

        # Add to vector store
        embedding_manager.add_documents(chunks)

        # Save vector store
        embedding_manager.save_vector_store("./chroma_db")

        return IngestionResponse(
            message=f"Successfully ingested directory {path}",
            chunks_processed=len(chunks)
        )

    except Exception as e:
        logger.error(f"Error ingesting directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge base."""
    try:
        result = rag_pipeline.query(request.question)

        # Format source documents
        source_docs = []
        for doc in result["source_documents"]:
            source_docs.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata
            })

        return QueryResponse(
            answer=result["answer"],
            source_documents=source_docs,
            token_usage=result["token_usage"]
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(request: QueryRequest):
    """Query the knowledge base using the ReAct agent."""
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        result = rag_agent.run(request.question)
        source_docs = [
            {"content": doc.page_content[:500], "metadata": doc.metadata}
            for doc in result["source_documents"]
        ]
        return AgentQueryResponse(
            answer=result["answer"],
            source_documents=source_docs,
            agent_steps=[
                {"tool": s[0].tool, "input": s[0].tool_input, "output": s[1]}
                for s in result["agent_steps"]
            ],
        )
    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents(limit: int = 10):
    """Get list of documents in the vector store."""
    try:
        # This is a simplified implementation
        # In a real system, you'd track documents separately
        return {"message": "Document listing not implemented yet"}

    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
