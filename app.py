"""
Streamlit Web UI for Enterprise RAG Knowledge Assistant
Provides an intuitive interface for document ingestion and Q&A.
"""

import streamlit as st
import os
import tempfile
import logging
from pathlib import Path

# Import our RAG components
from src.ingestion.ingestion import DocumentIngestionPipeline
from src.embedding.embedding import EmbeddingManager
from src.retrieval.retrieval import RAGPipeline
from src.agent.agent import RAGAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'assistant_initialized' not in st.session_state:
    st.session_state.assistant_initialized = False
if 'ingestion_pipeline' not in st.session_state:
    st.session_state.ingestion_pipeline = None
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'rag_agent' not in st.session_state:
    st.session_state.rag_agent = None
if 'documents_ingested' not in st.session_state:
    st.session_state.documents_ingested = 0

def initialize_assistant():
    """Initialize the RAG assistant components."""
    try:
        with st.spinner("Initializing RAG Assistant..."):
            # Get API key
            api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

            if not api_key:
                st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
                return False

            # Initialize components
            st.session_state.ingestion_pipeline = DocumentIngestionPipeline()
            st.session_state.embedding_manager = EmbeddingManager(
                embedding_provider="openai",
                vector_store="chroma",
                api_key=api_key
            )

            # Create vector store
            persist_dir = "./data/chroma_db"
            os.makedirs(persist_dir, exist_ok=True)
            st.session_state.embedding_manager.create_vector_store(persist_dir)

            # Initialize RAG pipeline
            st.session_state.rag_pipeline = RAGPipeline(
                embedding_manager=st.session_state.embedding_manager,
                llm_provider="openai",
                api_key=api_key
            )

            # Create QA chain
            st.session_state.rag_pipeline.create_qa_chain()

            # Initialize agent
            st.session_state.rag_agent = RAGAgent(
                embedding_manager=st.session_state.embedding_manager,
                api_key=api_key,
            )
            st.session_state.rag_agent.initialize()

            st.session_state.assistant_initialized = True
            st.success(" RAG Assistant initialized successfully!")
            return True

    except Exception as e:
        st.error(f" Failed to initialize assistant: {str(e)}")
        logger.error(f"Initialization error: {e}")
        return False

def ingest_documents(uploaded_files, chunk_size=1000, chunk_overlap=200):
    """Ingest uploaded documents."""
    if not st.session_state.assistant_initialized:
        st.error("Assistant not initialized")
        return 0

    total_chunks = 0

    for uploaded_file in uploaded_files:
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Update pipeline parameters
                st.session_state.ingestion_pipeline.chunk_size = chunk_size
                st.session_state.ingestion_pipeline.chunk_overlap = chunk_overlap

                # Process the file
                chunks = st.session_state.ingestion_pipeline.run_pipeline(tmp_path)

                # Add to vector store
                st.session_state.embedding_manager.add_documents(chunks)

                # Clean up
                os.unlink(tmp_path)

                total_chunks += len(chunks)
                st.success(f"✅ {uploaded_file.name}: {len(chunks)} chunks processed")

        except Exception as e:
            st.error(f" Error processing {uploaded_file.name}: {str(e)}")
            logger.error(f"Ingestion error for {uploaded_file.name}: {e}")

    # Save vector store
    try:
        st.session_state.embedding_manager.save_vector_store("./data/chroma_db")
        st.success(f" Total chunks ingested: {total_chunks}")
        st.session_state.documents_ingested += total_chunks
    except Exception as e:
        st.error(f" Error saving vector store: {str(e)}")

    return total_chunks

def query_assistant(question, use_agent=False):
    """Query the RAG assistant."""
    if not st.session_state.assistant_initialized:
        return None

    try:
        if use_agent:
            with st.spinner("Agent reasoning over knowledge base..."):
                result = st.session_state.rag_agent.run(question)
        else:
            with st.spinner("Searching knowledge base..."):
                result = st.session_state.rag_pipeline.query(question)
        return result
    except Exception as e:
        st.error(f" Query error: {str(e)}")
        logger.error(f"Query error: {e}")
        return None

def main():
    """Main Streamlit application."""

    # Title and description
    st.title(" Enterprise RAG Knowledge Assistant")
    st.markdown("""
    **Intelligent Knowledge Management System**

    Upload documents and ask questions to get accurate, cited answers from your knowledge base.
    Built with modern RAG (Retrieval-Augmented Generation) technology.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key input (if not in environment)
        if not os.getenv("OPENAI_API_KEY"):
            api_key = st.text_input("OpenAI API Key", type="password",
                                  help="Enter your OpenAI API key")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        # Initialization button
        if st.button("🚀 Initialize Assistant", type="primary"):
            initialize_assistant()

        # Status
        st.header("📊 Status")
        if st.session_state.assistant_initialized:
            st.success("✅ Assistant Ready")
            st.info(f"📄 Documents: {st.session_state.documents_ingested} chunks")
        else:
            st.warning("⚠️ Assistant Not Initialized")

        # Chunking parameters
        st.header("🔧 Ingestion Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000,
                              help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200,
                                 help="Overlap between chunks")

        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

    # Main content
    if not st.session_state.assistant_initialized:
        st.info("👆 Please initialize the assistant using the sidebar first.")

        # Sample data ingestion
        st.header("📚 Quick Start")
        st.markdown("Try ingesting the sample knowledge base:")

        if st.button("📖 Load Sample Data"):
            if initialize_assistant():
                sample_path = "./data/sample_knowledge.txt"
                if os.path.exists(sample_path):
                    chunks = ingest_documents(
                        [type('MockFile', (), {
                            'name': 'sample_knowledge.txt',
                            'getvalue': lambda: open(sample_path, 'rb').read()
                        })()],
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    st.success(f"✅ Sample data loaded: {chunks} chunks")
                else:
                    st.error("Sample data file not found")

        return

    # Document ingestion section
    st.header("📤 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, MD, DOCX)",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'md', 'docx'],
        help="Select one or more documents to add to the knowledge base"
    )

    if uploaded_files and st.button("🔄 Ingest Documents", type="primary"):
        if len(uploaded_files) > 0:
            chunks = ingest_documents(
                uploaded_files,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            st.success(f"🎉 Successfully ingested {len(uploaded_files)} files ({chunks} total chunks)")
        else:
            st.warning("Please select files to upload")

    # Query section
    st.header("❓ Ask Questions")

    use_agent = st.toggle(
        " Agent Mode (multi-step reasoning)",
        value=False,
        help="Agent mode decomposes complex questions and validates answers using ReAct reasoning.",
    )

    question = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="What is RAG? How does it work? What are the benefits?",
        help="Ask any question about the ingested documents"
    )

    if st.button("🔍 Search & Answer", type="primary") and question.strip():
        result = query_assistant(question.strip(), use_agent=use_agent)

        if result:
            # Display answer
            st.subheader("💡 Answer")
            st.write(result['answer'])

            # Display agent steps (agent mode only)
            if use_agent and result.get('agent_steps'):
                with st.expander(" Agent Reasoning Steps"):
                    for i, (action, observation) in enumerate(result['agent_steps'], 1):
                        st.markdown(f"**Step {i} — Tool:** `{action.tool}`")
                        st.markdown(f"**Input:** {action.tool_input}")
                        st.markdown(f"**Observation:** {str(observation)[:300]}")
                        st.divider()

            # Display sources
            if result['source_documents']:
                st.subheader("📚 Sources")

                for i, doc in enumerate(result['source_documents'][:3], 1):
                    with st.expander(f"Source {i}"):
                        st.markdown(f"**Content:** {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
                        if doc.metadata:
                            st.markdown(f"**Metadata:** {doc.metadata}")

            # Display token usage
            if result.get('token_usage'):
                st.subheader("📊 Token Usage")
                usage = result['token_usage']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tokens", usage.get('total_tokens', 'N/A'))
                with col2:
                    st.metric("Prompt Tokens", usage.get('prompt_tokens', 'N/A'))
                with col3:
                    st.metric("Completion Tokens", usage.get('completion_tokens', 'N/A'))
                with col4:
                    if usage.get('total_cost'):
                        st.metric("Cost ($)", f"{usage['total_cost']:.4f}")
        else:
            st.error("Failed to get answer. Please try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Enterprise RAG Knowledge Assistant** | Built with Streamlit, LangChain, and OpenAI

    *Features: Multi-format document ingestion, intelligent chunking, vector embeddings,
    similarity search, and LLM-powered Q&A with source citations.*
    """)

if __name__ == "__main__":
    main()
