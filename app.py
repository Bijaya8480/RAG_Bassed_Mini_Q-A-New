"""
Streamlit User Interface for RAG-based Mini Q&A System.

Provides a clean, beginner-friendly interface for:
- Document upload (PDF/TXT)
- Document processing
- Question answering with source references
"""

import streamlit as st
import os
from pathlib import Path

from rag_pipeline import RAGSystem, create_rag_system
from document_loader import is_supported_format, get_file_extension


# Page configuration
st.set_page_config(
    page_title="RAG Mini Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .stat-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .source-chunk {
        background: #f3f4f6;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .answer-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #991b1b;
    }
    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #166534;
    }
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1e40af;
    }
    .stButton > button {
        width: 100%;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def load_config_from_env():
    """Load configuration from environment variables."""
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", "300")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
        "top_k": int(os.getenv("TOP_K", "3")),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "openai_api_key": os.getenv("OPENAI_API_KEY") or None,
        "openai_model": "gpt-4o-mini"
    }


def render_header():
    """Render application header."""
    st.markdown('<div class="main-header">📚 RAG Mini Q&A System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a document and ask questions. '
        'Answers are grounded in your document with source references.</div>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render sidebar with configuration and info."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        config = load_config_from_env()
        
        # Show current config
        with st.expander("Current Settings", expanded=False):
            st.write(f"**Chunk Size:** {config['chunk_size']} words")
            st.write(f"**Chunk Overlap:** {config['chunk_overlap']} words")
            st.write(f"**Top-K Retrieval:** {config['top_k']}")
            st.write(f"**Similarity Threshold:** {config['similarity_threshold']}")
            st.write(f"**Embedding Model:** {config['embedding_model']}")
            st.write(f"**OpenAI Configured:** {'Yes' if config['openai_api_key'] else 'No (using extractive fallback)'}")
        
        st.markdown("---")
        
        # System status
        st.markdown("### 📊 System Status")
        if st.session_state.rag_system:
            stats = st.session_state.rag_system.get_stats()
            if stats.get("initialized"):
                st.markdown('<div class="success-box">✅ System Ready</div>', unsafe_allow_html=True)
                st.write(f"**Chunks Indexed:** {stats['chunk_count']}")
                st.write(f"**Embedding Dim:** {stats['embedding_dim']}")
                st.write(f"**LLM Mode:** {'OpenAI' if stats['openai_configured'] else 'Extractive Fallback'}")
            else:
                st.markdown('<div class="info-box">⏳ Waiting for document...</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">⏳ Not initialized</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) system:
        1. **Loads** your PDF/TXT document
        2. **Chunks** it into overlapping segments
        3. **Embeds** chunks using Sentence Transformers
        4. **Stores** embeddings in FAISS vector DB
        5. **Retrieves** top-3 relevant chunks for your query
        6. **Generates** answers using only document context
        
        **Anti-hallucination:** Answers are strictly grounded in retrieved content.
        """)


def render_document_upload():
    """Render document upload and processing section."""
    st.markdown('<div class="section-header">📄 Document Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Maximum file size: 200MB"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        process_btn = st.button(
            "🔄 Process Document",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True
        )
    
    with col2:
        if st.session_state.document_processed:
            st.markdown('<div class="success-box">✅ Document processed and indexed</div>', unsafe_allow_html=True)
    
    if process_btn and uploaded_file:
        process_document(uploaded_file)
    
    # Show processing stats if available
    if st.session_state.processing_stats:
        stats = st.session_state.processing_stats
        st.markdown('<div class="section-header">📈 Processing Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Chunks Created", stats["chunk_count"])
        with col2:
            st.metric("Total Words", f"{stats['total_words']:,}")
        with col3:
            st.metric("Avg Words/Chunk", f"{stats['avg_chunk_words']:.0f}")
        with col4:
            st.metric("Embedding Dim", stats["embedding_dim"])


def process_document(uploaded_file):
    """Process uploaded document and build index."""
    config = load_config_from_env()
    
    # Create RAG system with config
    rag = create_rag_system(**config)
    st.session_state.rag_system = rag
    
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load document
        status_text.text("📖 Loading document...")
        progress_bar.progress(20)
        
        file_ext = get_file_extension(uploaded_file.name)
        
        # Step 2: Process (load, chunk, embed, index)
        status_text.text("✂️ Chunking text...")
        progress_bar.progress(40)
        
        status_text.text("🧠 Generating embeddings...")
        progress_bar.progress(60)
        
        status_text.text("💾 Building vector index...")
        progress_bar.progress(80)
        
        stats = rag.load_and_process(uploaded_file, file_extension)
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        st.session_state.document_processed = True
        st.session_state.processing_stats = stats
        
        st.success(f"Successfully processed **{uploaded_file.name}**")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def render_question_interface():
    """Render question input and answer display."""
    if not st.session_state.document_processed:
        st.markdown('<div class="info-box">👆 Please upload and process a document first.</div>', unsafe_allow_html=True)
        return
    
    st.markdown('<div class="section-header">❓ Ask a Question</div>', unsafe_allow_html=True)
    
    # Question input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How many days of annual leave do employees get?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        ask_btn = st.button(
            "🔍 Get Answer",
            type="primary",
            disabled=not query.strip(),
            use_container_width=True
        )
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.last_query = ""
            st.session_state.last_result = None
            st.rerun()
    
    # Process question
    if ask_btn and query.strip():
        with st.spinner("🔍 Searching document and generating answer..."):
            result = st.session_state.rag_system.ask_question(query)
            st.session_state.last_query = query
            st.session_state.last_result = result
        st.rerun()
    
    # Display result
    if st.session_state.last_result:
        render_answer(st.session_state.last_result)


def render_answer(result):
    """Render answer with sources."""
    if result.success:
        st.markdown('<div class="section-header">💡 Answer</div>', unsafe_allow_html=True)
        
        # Answer box
        st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)
        
        # Sources
        if result.sources:
            st.markdown('<div class="section-header">📎 Source Chunks</div>', unsafe_allow_html=True)
            
            for i, source in enumerate(result.sources):
                with st.expander(f"Chunk {source['chunk_id']} — Similarity: {source['similarity_score']:.4f}", expanded=(i==0)):
                    st.markdown(f'<div class="source-chunk">{source["text_preview"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Words: {source['word_count']} | Chunk ID: {source['chunk_id']}")
    else:
        st.markdown(f'<div class="error-box">❌ Error: {result.error}</div>', unsafe_allow_html=True)


def render_footer():
    """Render footer with tips."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        <strong>Tips for better results:</strong><br>
        • Ask specific questions about the document content<br>
        • Use keywords from the document<br>
        • The system only answers from the uploaded document<br>
        • Configure OpenAI API key in .env for LLM-generated answers
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    render_header()
    render_sidebar()
    
    # Main content area
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        render_document_upload()
    
    with col_right:
        render_question_interface()
    
    render_footer()


if __name__ == "__main__":
    main()