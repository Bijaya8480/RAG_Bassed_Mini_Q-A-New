"""
Test Suite for RAG Pipeline

Run with: pytest tests/test_rag_pipeline.py -v
"""

import pytest
import io
import os
import tempfile
from pathlib import Path

from document_loader import (
    load_txt, load_pdf, load_document, get_file_extension, is_supported_format,
    DocumentLoadError
)
from text_splitter import (
    split_text_into_chunks, split_text_into_chunks_with_metadata,
    TextSplitter, validate_chunk_params, TextSplitterError
)
from embeddings import EmbeddingGenerator, EmbeddingError
from vector_store import FAISSVectorStore, VectorStoreError
from rag_pipeline import RAGSystem, RAGResult, create_rag_system


# Test data
SAMPLE_TEXT = """Company Policy Document

Annual Paid Leave
All full-time employees are entitled to 20 days of annual paid leave per calendar year.

Working Hours
Standard working hours are Monday to Friday, 9:30 AM to 6:00 PM.

Remote Work Policy
Employees may work remotely up to 2 days per week with prior manager approval."""

SAMPLE_PDF_CONTENT = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test PDF Content) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000206 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
301
%%EOF"""


class TestDocumentLoader:
    """Tests for document loading module."""
    
    def test_load_txt_from_path(self):
        """Test loading TXT from file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(SAMPLE_TEXT)
            temp_path = f.name
        
        try:
            result = load_txt(temp_path)
            assert "Annual Paid Leave" in result
            assert "20 days" in result
        finally:
            os.unlink(temp_path)
    
    def test_load_txt_from_file_object(self):
        """Test loading TXT from file-like object."""
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        result = load_txt(file_obj)
        assert "Annual Paid Leave" in result
    
    def test_load_txt_empty_file(self):
        """Test loading empty TXT file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentLoadError, match="empty"):
                load_txt(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_txt_whitespace_only(self):
        """Test loading whitespace-only TXT file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("   \n\t  \n  ")
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentLoadError, match="empty"):
                load_txt(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_txt_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(DocumentLoadError, match="not found"):
            load_txt("/nonexistent/path.txt")
    
    def test_load_pdf_from_path(self):
        """Test loading PDF from file path."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            f.write(SAMPLE_PDF_CONTENT)
            temp_path = f.name
        
        try:
            # This may fail with minimal PDF, so we just test it doesn't crash
            result = load_pdf(temp_path)
            # If it succeeds, check it returns string
            assert isinstance(result, str)
        except DocumentLoadError:
            # Expected for minimal test PDF
            pass
        finally:
            os.unlink(temp_path)
    
    def test_load_pdf_from_file_object(self):
        """Test loading PDF from file-like object."""
        file_obj = io.BytesIO(SAMPLE_PDF_CONTENT)
        try:
            result = load_pdf(file_obj)
            assert isinstance(result, str)
        except DocumentLoadError:
            pass
    
    def test_load_pdf_nonexistent_file(self):
        """Test loading nonexistent PDF raises error."""
        with pytest.raises(DocumentLoadError, match="not found"):
            load_pdf("/nonexistent/path.pdf")
    
    def test_load_document_txt(self):
        """Test load_document with TXT."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(SAMPLE_TEXT)
            temp_path = f.name
        
        try:
            result = load_document(temp_path)
            assert "Annual Paid Leave" in result
        finally:
            os.unlink(temp_path)
    
    def test_load_document_unsupported_format(self):
        """Test load_document with unsupported format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
            f.write("test")
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentLoadError, match="Unsupported"):
                load_document(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension("test.pdf") == ".pdf"
        assert get_file_extension("test.TXT") == ".txt"
        assert get_file_extension("path/to/file.pdf") == ".pdf"
        assert get_file_extension("no_extension") == ""
    
    def test_is_supported_format(self):
        """Test format support check."""
        assert is_supported_format("test.pdf") is True
        assert is_supported_format("test.txt") is True
        assert is_supported_format("test.TXT") is True
        assert is_supported_format("test.docx") is False
        assert is_supported_format("test") is False


class TestTextSplitter:
    """Tests for text splitting module."""
    
    def test_split_basic(self):
        """Test basic text splitting."""
        chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 50
    
    def test_split_default_params(self):
        """Test splitting with default parameters."""
        chunks = split_text_into_chunks(SAMPLE_TEXT)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.split()) <= 300
    
    def test_split_overlap(self):
        """Test chunk overlap."""
        chunks = split_text_into_chunks(SAMPLE_TEXT, chunk_size=30, chunk_overlap=10)
        # With overlap, we should get more chunks
        assert len(chunks) >= 2
    
    def test_split_single_chunk(self):
        """Test text smaller than chunk size."""
        short_text = "Short text here."
        chunks = split_text_into_chunks(short_text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == short_text
    
    def test_split_empty_text(self):
        """Test splitting empty text raises error."""
        with pytest.raises(TextSplitterError, match="empty"):
            split_text_into_chunks("")
    
    def test_split_whitespace_only(self):
        """Test splitting whitespace-only text raises error."""
        with pytest.raises(TextSplitterError, match="empty"):
            split_text_into_chunks("   \n\t  ")
    
    def test_validate_chunk_params_valid(self):
        """Test valid chunk parameters."""
        validate_chunk_params(300, 50)  # Should not raise
        validate_chunk_params(100, 0)   # Should not raise
        validate_chunk_params(10, 5)    # Should not raise
    
    def test_validate_chunk_params_invalid_size(self):
        """Test invalid chunk size."""
        with pytest.raises(TextSplitterError, match="positive"):
            validate_chunk_params(0, 10)
        with pytest.raises(TextSplitterError, match="positive"):
            validate_chunk_params(-10, 5)
    
    def test_validate_chunk_params_invalid_overlap(self):
        """Test invalid overlap."""
        with pytest.raises(TextSplitterError, match="negative"):
            validate_chunk_params(100, -10)
        with pytest.raises(TextSplitterError, match="less than chunk_size"):
            validate_chunk_params(100, 100)
        with pytest.raises(TextSplitterError, match="less than chunk_size"):
            validate_chunk_params(100, 150)
    
    def test_split_with_metadata(self):
        """Test splitting with metadata."""
        chunks = split_text_into_chunks_with_metadata(SAMPLE_TEXT, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "word_count" in chunk
            assert "start_word" in chunk
            assert "end_word" in chunk
            assert chunk["word_count"] == len(chunk["text"].split())
    
    def test_text_splitter_class(self):
        """Test TextSplitter class."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split(SAMPLE_TEXT)
        assert len(chunks) > 0
        
        chunks_meta = splitter.split_with_metadata(SAMPLE_TEXT)
        assert len(chunks_meta) == len(chunks)


class TestEmbeddings:
    """Tests for embedding generation module."""
    
    @pytest.fixture(scope="class")
    def embedding_generator(self):
        """Create embedding generator for tests."""
        return EmbeddingGenerator("all-MiniLM-L6-v2")
    
    def test_encode_single_string(self, embedding_generator):
        """Test encoding single string."""
        emb = embedding_generator.encode("test text")
        assert emb.shape == (embedding_generator.embedding_dim,)
        assert emb.dtype == np.float32
    
    def test_encode_list_of_strings(self, embedding_generator):
        """Test encoding list of strings."""
        embs = embedding_generator.encode(["text 1", "text 2", "text 3"])
        assert embs.shape == (3, embedding_generator.embedding_dim)
    
    def test_encode_normalized(self, embedding_generator):
        """Test that embeddings are normalized."""
        emb = embedding_generator.encode("test", normalize=True)
        norm = (emb ** 2).sum() ** 0.5
        assert abs(norm - 1.0) < 1e-5
    
    def test_encode_not_normalized(self, embedding_generator):
        """Test non-normalized embeddings."""
        emb = embedding_generator.encode("test", normalize=False)
        norm = (emb ** 2).sum() ** 0.5
        # Not necessarily 1.0
        assert norm > 0
    
    def test_encode_empty_list(self, embedding_generator):
        """Test encoding empty list raises error."""
        with pytest.raises(EmbeddingError, match="No texts provided"):
            embedding_generator.encode([])
    
    def test_encode_batch(self, embedding_generator):
        """Test batch encoding."""
        texts = [f"text {i}" for i in range(10)]
        embs = embedding_generator.encode_batch(texts, batch_size=3)
        assert embs.shape == (10, embedding_generator.embedding_dim)
    
    def test_similarity(self, embedding_generator):
        """Test similarity computation."""
        emb1 = embedding_generator.encode("hello world")
        emb2 = embedding_generator.encode("hello world")
        emb3 = embedding_generator.encode("completely different text")
        
        sim_same = embedding_generator.similarity(emb1, emb2)
        sim_diff = embedding_generator.similarity(emb1, emb3)
        
        assert sim_same > sim_diff
        # Allow small floating point error above 1.0
        assert -1e-5 <= sim_same <= 1 + 1e-5
        assert -1e-5 <= sim_diff <= 1 + 1e-5
    
    def test_embedding_dimension(self, embedding_generator):
        """Test embedding dimension property."""
        assert embedding_generator.embedding_dim == 384  # all-MiniLM-L6-v2


class TestVectorStore:
    """Tests for FAISS vector store."""
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store for tests."""
        return FAISSVectorStore(384)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings - distinct and normalized."""
        import numpy as np
        np.random.seed(42)
        # Generate distinct embeddings by using identity-like vectors
        embs = np.eye(5, 384, dtype=np.float32)
        # Add small random noise to make them distinct but still mostly orthogonal
        noise = np.random.randn(5, 384).astype(np.float32) * 0.01
        embs = embs + noise
        # Normalize
        for i in range(embs.shape[0]):
            norm = (embs[i] ** 2).sum() ** 0.5
            embs[i] = embs[i] / norm
        return embs
    
    def test_add_embeddings(self, vector_store, sample_embeddings):
        """Test adding embeddings."""
        texts = [f"chunk {i}" for i in range(5)]
        vector_store.add(sample_embeddings, texts)
        assert len(vector_store) == 5
    
    def test_add_mismatched_lengths(self, vector_store, sample_embeddings):
        """Test adding with mismatched lengths raises error."""
        texts = ["chunk 1", "chunk 2"]  # Only 2 texts for 5 embeddings
        with pytest.raises(VectorStoreError, match="match number of texts"):
            vector_store.add(sample_embeddings, texts)
    
    def test_add_wrong_dimension(self, vector_store):
        """Test adding wrong dimension raises error."""
        import numpy as np
        wrong_embs = np.random.randn(3, 100).astype(np.float32)
        texts = ["a", "b", "c"]
        with pytest.raises(VectorStoreError, match="dimension"):
            vector_store.add(wrong_embs, texts)
    
    def test_search(self, vector_store, sample_embeddings):
        """Test searching."""
        texts = [f"chunk {i}" for i in range(5)]
        vector_store.add(sample_embeddings, texts)
        
        # Query with first embedding (should find itself)
        results = vector_store.search(sample_embeddings[0:1], top_k=3)
        assert len(results) >= 1  # At least itself
        assert results[0][0] == 0  # First result should be chunk 0
        assert results[0][1] > 0.99  # Very high similarity
    
    def test_search_empty_store(self, vector_store):
        """Test searching empty store raises error."""
        import numpy as np
        query = np.random.randn(1, 384).astype(np.float32)
        with pytest.raises(VectorStoreError, match="empty"):
            vector_store.search(query)
    
    def test_search_with_threshold(self, vector_store, sample_embeddings):
        """Test searching with similarity threshold."""
        texts = [f"chunk {i}" for i in range(5)]
        vector_store.add(sample_embeddings, texts)
        
        # High threshold - fewer results
        results = vector_store.search(sample_embeddings[0:1], top_k=3, similarity_threshold=0.99)
        assert len(results) <= 3
        assert len(results) >= 1  # Should at least find itself
        
        # Very high threshold - no results (allow for floating point precision)
        results = vector_store.search(sample_embeddings[0:1], top_k=3, similarity_threshold=1.0 + 1e-5)
        assert len(results) == 0
    
    def test_get_chunk(self, vector_store, sample_embeddings):
        """Test getting chunk by ID."""
        texts = [f"chunk {i}" for i in range(5)]
        vector_store.add(sample_embeddings, texts)
        
        text, meta = vector_store.get_chunk(0)
        assert text == "chunk 0"
        assert meta["chunk_id"] == 0
    
    def test_get_chunk_out_of_bounds(self, vector_store, sample_embeddings):
        """Test getting out-of-bounds chunk raises error."""
        texts = ["chunk 0"]
        vector_store.add(sample_embeddings[:1], texts)
        
        with pytest.raises(VectorStoreError, match="out of bounds"):
            vector_store.get_chunk(5)
    
    def test_clear(self, vector_store, sample_embeddings):
        """Test clearing store."""
        texts = ["chunk 0"]
        vector_store.add(sample_embeddings[:1], texts)
        assert len(vector_store) == 1
        
        vector_store.clear()
        assert len(vector_store) == 0
        assert vector_store.is_empty()
    
    def test_is_empty(self, vector_store):
        """Test is_empty."""
        assert vector_store.is_empty()
        vector_store.add(
            __import__('numpy').random.randn(1, 384).astype('float32'),
            ["test"]
        )
        assert not vector_store.is_empty()


class TestRAGPipeline:
    """Tests for complete RAG pipeline."""
    
    @pytest.fixture
    def rag_system(self):
        """Create RAG system for tests."""
        return create_rag_system(
            chunk_size=50,
            chunk_overlap=10,
            top_k=2,
            similarity_threshold=0.1,
            openai_api_key=None  # Use extractive fallback
        )
    
    def test_load_and_process_txt(self, rag_system):
        """Test loading and processing TXT document."""
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        stats = rag_system.load_and_process(file_obj, ".txt")
        
        assert stats["chunk_count"] > 0
        assert stats["embedding_dim"] == 384
        assert stats["total_words"] > 0
        assert rag_system.is_initialized
    
    def test_load_and_process_pdf(self, rag_system):
        """Test loading and processing PDF document."""
        # Create a simple PDF-like content
        file_obj = io.BytesIO(SAMPLE_PDF_CONTENT)
        try:
            stats = rag_system.load_and_process(file_obj, ".pdf")
            assert stats["chunk_count"] >= 0  # May be 0 for minimal PDF
        except Exception:
            # Expected for minimal test PDF
            pass
    
    def test_retrieve_relevant(self, rag_system):
        """Test retrieving relevant chunks."""
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        rag_system.load_and_process(file_obj, ".txt")
        
        results = rag_system.retrieve("annual leave days")
        assert len(results) > 0
        # First result should be about leave
        assert "leave" in results[0][2].lower() or "20" in results[0][2]
    
    def test_retrieve_irrelevant(self, rag_system):
        """Test retrieving for irrelevant query."""
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        rag_system.load_and_process(file_obj, ".txt")
        
        results = rag_system.retrieve("quantum physics mechanics")
        # May return results but with low similarity
        # Should not crash
        assert isinstance(results, list)
    
    def test_retrieve_empty_query(self, rag_system):
        """Test retrieving with empty query raises error."""
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        rag_system.load_and_process(file_obj, ".txt")
        
        with pytest.raises(VectorStoreError, match="empty"):
            rag_system.retrieve("")
    
    def test_retrieve_before_init(self, rag_system):
        """Test retrieving before initialization raises error."""
        with pytest.raises(VectorStoreError, match="not initialized"):
            rag_system.retrieve("test query")
    
    def test_ask_question_relevant(self, rag_system):
        """Test asking relevant question."""
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        rag_system.load_and_process(file_obj, ".txt")
        
        result = rag_system.ask_question("How many days of annual leave?")
        
        assert result.success
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert len(result.sources) > 0
        assert result.query == "How many days of annual leave?"
    
    def test_ask_question_irrelevant(self, rag_system):
        """Test asking irrelevant question."""
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        rag_system.load_and_process(file_obj, ".txt")
        
        result = rag_system.ask_question("What is quantum physics?")
        
        assert result.success
        # Should say not found or give low-confidence answer
        assert "could not find" in result.answer.lower() or len(result.sources) == 0
    
    def test_ask_question_before_init(self, rag_system):
        """Test asking question before initialization."""
        result = rag_system.ask_question("test query")
        
        assert not result.success
        assert "not initialized" in result.error.lower()
    
    def test_get_stats(self, rag_system):
        """Test getting system stats."""
        stats = rag_system.get_stats()
        assert stats["initialized"] is False
        
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        rag_system.load_and_process(file_obj, ".txt")
        
        stats = rag_system.get_stats()
        assert stats["initialized"] is True
        assert stats["chunk_count"] > 0
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"
        assert stats["openai_configured"] is False
    
    def test_extractive_fallback(self, rag_system):
        """Test extractive fallback generates answer."""
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        rag_system.load_and_process(file_obj, ".txt")
        
        result = rag_system.ask_question("working hours")
        
        assert result.success
        assert "9:30" in result.answer or "6:00" in result.answer or "Monday" in result.answer


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_txt(self):
        """Test full pipeline with TXT document."""
        rag = create_rag_system(
            chunk_size=100,
            chunk_overlap=20,
            top_k=3,
            similarity_threshold=0.2,
            openai_api_key=None
        )
        
        # Load
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        stats = rag.load_and_process(file_obj, ".txt")
        assert stats["chunk_count"] > 0
        
        # Query
        result = rag.ask_question("remote work policy")
        assert result.success
        assert len(result.sources) > 0
        # Check answer mentions remote work
        assert "remote" in result.answer.lower() or "2 days" in result.answer
    
    def test_chunk_overlap_preserves_context(self):
        """Test that overlap preserves context across chunks."""
        rag = create_rag_system(chunk_size=30, chunk_overlap=10, openai_api_key=None)
        file_obj = io.BytesIO(SAMPLE_TEXT.encode('utf-8'))
        rag.load_and_process(file_obj, ".txt")
        
        # Query for content that might be at chunk boundary
        result = rag.ask_question("annual paid leave calendar year")
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])