"""
RAG Pipeline Module

Complete Retrieval-Augmented Generation pipeline integrating:
- Document loading
- Text chunking
- Embedding generation
- FAISS vector storage
- Query retrieval
- LLM answer generation (OpenAI or extractive fallback)
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from document_loader import load_document, get_file_extension, is_supported_format, DocumentLoadError
from text_splitter import TextSplitter, split_text_into_chunks_with_metadata, TextSplitterError
from embeddings import EmbeddingGenerator, EmbeddingError
from vector_store import FAISSVectorStore, VectorStoreError


@dataclass
class RAGResult:
    """Result of a RAG query."""
    answer: str
    sources: List[Dict]
    query: str
    success: bool
    error: Optional[str] = None


class RAGSystem:
    """
    End-to-End RAG System for document question answering.
    
    Pipeline:
    1. Load document (PDF/TXT)
    2. Split into chunks
    3. Generate embeddings
    4. Store in FAISS
    5. Retrieve relevant chunks for query
    6. Generate answer using LLM or extractive fallback
    
    Example:
        rag = RAGSystem()
        rag.load_and_process("document.pdf")
        result = rag.ask_question("What is the leave policy?")
        print(result.answer)
    """
    
    # System prompt for anti-hallucination
    SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer using ONLY the supplied document context.

Never use external knowledge.

Do not invent, assume, or fabricate information.

If the answer is not available in the supplied context, respond:

'I could not find this information in the provided document.'

Keep answers concise and provide source chunk references."""
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        top_k: int = 3,
        similarity_threshold: float = 0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini"
    ):
        """
        Initialize RAG system.
        
        Args:
            chunk_size: Words per chunk (default: 300).
            chunk_overlap: Overlap words between chunks (default: 50).
            top_k: Number of chunks to retrieve (default: 3).
            similarity_threshold: Minimum similarity for retrieval (default: 0.3).
            embedding_model: Sentence Transformer model name.
            openai_api_key: OpenAI API key (optional, uses fallback if not provided).
            openai_model: OpenAI model to use (default: gpt-4o-mini).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model
        
        # Initialize components
        self.text_splitter = TextSplitter(chunk_size, chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_store: Optional[FAISSVectorStore] = None
        self.document_text: Optional[str] = None
        self.is_initialized = False
        
        # Initialize OpenAI client if key provided
        self._openai_client = None
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except Exception:
                self._openai_client = None
    
    def load_and_process(
        self,
        file_input,
        file_extension: str = None
    ) -> Dict:
        """
        Load document and build vector index.
        
        Args:
            file_input: File path, Path object, or file-like object.
            file_extension: File extension (required for file-like objects).
            
        Returns:
            Dict with processing stats: chunk_count, embedding_dim, etc.
            
        Raises:
            DocumentLoadError: If document loading fails.
            TextSplitterError: If chunking fails.
            EmbeddingError: If embedding generation fails.
            VectorStoreError: If index creation fails.
        """
        # Load document
        if file_extension is None:
            if hasattr(file_input, 'name'):
                file_extension = get_file_extension(file_input.name)
            else:
                raise DocumentLoadError("file_extension required for file-like objects")
                
        if not is_supported_format(file_extension):
            raise DocumentLoadError(f"Unsupported format: {file_extension}")
            
        self.document_text = load_document(file_input, file_extension)
        
        # Split into chunks
        chunks_with_metadata = split_text_into_chunks_with_metadata(
            self.document_text,
            self.chunk_size,
            self.chunk_overlap
        )
        chunk_texts = [c["text"] for c in chunks_with_metadata]
        chunk_metadata = chunks_with_metadata
        
        # Generate embeddings
        embeddings = self.embedding_generator.encode_batch(chunk_texts)
        
        # Create vector store
        self.vector_store = FAISSVectorStore(self.embedding_generator.embedding_dim)
        self.vector_store.add(embeddings, chunk_texts, chunk_metadata)
        
        self.is_initialized = True
        
        return {
            "chunk_count": len(chunk_texts),
            "embedding_dim": self.embedding_generator.embedding_dim,
            "total_words": len(self.document_text.split()),
            "avg_chunk_words": np.mean([c["word_count"] for c in chunk_metadata])
        }
    
    def retrieve(self, query: str) -> List[Tuple[int, float, str, dict]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: User question.
            
        Returns:
            List of (chunk_id, score, text, metadata) tuples.
        """
        if not self.is_initialized or self.vector_store is None:
            raise VectorStoreError("System not initialized. Call load_and_process() first.")
            
        if not query or not query.strip():
            raise VectorStoreError("Query cannot be empty")
            
        # Generate query embedding
        query_embedding = self.embedding_generator.encode(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )
        
        return results
    
    def _build_context(self, retrieved_chunks: List[Tuple]) -> str:
        """Build context string from retrieved chunks."""
        if not retrieved_chunks:
            return ""
            
        context_parts = []
        for chunk_id, score, text, metadata in retrieved_chunks:
            context_parts.append(f"[Chunk {chunk_id} (similarity: {score:.3f})]\n{text}")
            
        return "\n\n---\n\n".join(context_parts)
    
    def _build_sources(self, retrieved_chunks: List[Tuple]) -> List[Dict]:
        """Build source references for response."""
        sources = []
        for chunk_id, score, text, metadata in retrieved_chunks:
            sources.append({
                "chunk_id": chunk_id,
                "similarity_score": round(score, 4),
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
                "word_count": metadata.get("word_count", 0)
            })
        return sources
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using LLM or extractive fallback.
        
        Args:
            query: User question.
            context: Retrieved document context.
            
        Returns:
            Generated answer.
        """
        if not context.strip():
            return "I could not find this information in the provided document."
        
        # Try OpenAI if available
        if self._openai_client:
            try:
                return self._generate_with_openai(query, context)
            except Exception:
                pass  # Fall through to extractive fallback
        
        # Extractive fallback
        return self._generate_extractive(query, context)
    
    def _generate_with_openai(self, query: str, context: str) -> str:
        """Generate answer using OpenAI API."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Document Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = self._openai_client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_extractive(self, query: str, context: str) -> str:
        """
        Extractive fallback: return most relevant sentences from context.
        Simple keyword-based extraction without external API.
        """
        # Split context into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Score sentences by keyword overlap with query
        query_words = set(query.lower().split())
        scored_sentences = []
        
        for sent in sentences:
            if len(sent.strip()) < 20:  # Skip very short fragments
                continue
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            if overlap > 0:
                scored_sentences.append((overlap, sent.strip()))
        
        if not scored_sentences:
            return "I could not find this information in the provided document."
        
        # Sort by overlap and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:3]]
        
        answer = " ".join(top_sentences)
        return f"{answer}\n\n[Note: Answer generated using extractive fallback - no OpenAI API key configured]"
    
    def ask_question(self, query: str) -> RAGResult:
        """
        Ask a question and get a complete RAG result.
        
        Args:
            query: User question.
            
        Returns:
            RAGResult with answer, sources, and metadata.
        """
        try:
            if not self.is_initialized:
                return RAGResult(
                    answer="",
                    sources=[],
                    query=query,
                    success=False,
                    error="System not initialized. Please load a document first."
                )
                
            # Retrieve relevant chunks
            retrieved = self.retrieve(query)
            
            if not retrieved:
                return RAGResult(
                    answer="I could not find this information in the provided document.",
                    sources=[],
                    query=query,
                    success=True
                )
            
            # Build context and generate answer
            context = self._build_context(retrieved)
            answer = self.generate_answer(query, context)
            sources = self._build_sources(retrieved)
            
            return RAGResult(
                answer=answer,
                sources=sources,
                query=query,
                success=True
            )
            
        except Exception as e:
            return RAGResult(
                answer="",
                sources=[],
                query=query,
                success=False,
                error=str(e)
            )
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        if not self.is_initialized or self.vector_store is None:
            return {"initialized": False}
            
        return {
            "initialized": True,
            "chunk_count": len(self.vector_store),
            "embedding_dim": self.embedding_generator.embedding_dim,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "openai_configured": self._openai_client is not None
        }


def create_rag_system(**kwargs) -> RAGSystem:
    """
    Factory function to create a configured RAG system.
    
    Args:
        **kwargs: Configuration parameters for RAGSystem.
        
    Returns:
        Configured RAGSystem instance.
    """
    return RAGSystem(**kwargs)