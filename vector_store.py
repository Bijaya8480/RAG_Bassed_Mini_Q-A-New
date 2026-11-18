"""
Vector Store Module

FAISS-based vector database for storing and retrieving document embeddings.
Uses inner product (IndexFlatIP) with normalized vectors for cosine similarity.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Union
import pickle
from pathlib import Path


class VectorStoreError(Exception):
    """Custom exception for vector store errors."""
    pass


class FAISSVectorStore:
    """
    FAISS vector store for document embeddings.
    
    Uses IndexFlatIP (inner product) with L2-normalized vectors
    for efficient cosine similarity search.
    
    Example:
        store = FAISSVectorStore(dimension=384)
        store.add(embeddings, chunk_texts)
        results = store.search(query_embedding, top_k=3)
    """
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension (e.g., 384 for all-MiniLM-L6-v2).
        """
        if dimension <= 0:
            raise VectorStoreError("Dimension must be positive")
            
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_texts: List[str] = []
        self.chunk_metadata: List[dict] = []
        
    def add(self, embeddings: np.ndarray, texts: List[str], metadata: Optional[List[dict]] = None) -> None:
        """
        Add embeddings and associated texts to the store.
        
        Args:
            embeddings: numpy array of shape (n_chunks, dimension).
            texts: List of chunk texts corresponding to embeddings.
            metadata: Optional list of metadata dicts for each chunk.
            
        Raises:
            VectorStoreError: If dimensions don't match or inputs are invalid.
        """
        if len(embeddings) != len(texts):
            raise VectorStoreError("Number of embeddings must match number of texts")
            
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match store dimension {self.dimension}"
            )
            
        if len(embeddings) == 0:
            return
            
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.chunk_texts.extend(texts)
        
        if metadata:
            self.chunk_metadata.extend(metadata)
        else:
            # Generate default metadata
            start_idx = len(self.chunk_metadata)
            for i in range(len(texts)):
                self.chunk_metadata.append({
                    "chunk_id": start_idx + i,
                    "text_length": len(texts[i]),
                    "word_count": len(texts[i].split())
                })
                
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[int, float, str, dict]]:
        """
        Search for most similar chunks.
        
        Args:
            query_embedding: Query embedding vector of shape (dimension,) or (1, dimension).
            top_k: Number of top results to return (default: 3).
            similarity_threshold: Minimum similarity score (default: 0.0).
            
        Returns:
            List of tuples: (chunk_id, similarity_score, chunk_text, metadata).
            
        Raises:
            VectorStoreError: If index is empty or query dimension mismatch.
        """
        if self.index.ntotal == 0:
            raise VectorStoreError("Vector store is empty - no documents indexed")
            
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        if query_embedding.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query dimension {query_embedding.shape[1]} doesn't match store dimension {self.dimension}"
            )
            
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, actual_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue
            if score < similarity_threshold:
                continue
            results.append((
                int(idx),
                float(score),
                self.chunk_texts[idx],
                self.chunk_metadata[idx]
            ))
            
        return results
    
    def get_chunk(self, chunk_id: int) -> Tuple[str, dict]:
        """
        Retrieve chunk text and metadata by ID.
        
        Args:
            chunk_id: Chunk index.
            
        Returns:
            Tuple of (chunk_text, metadata).
            
        Raises:
            VectorStoreError: If chunk_id is out of bounds.
        """
        if chunk_id < 0 or chunk_id >= len(self.chunk_texts):
            raise VectorStoreError(f"Chunk ID {chunk_id} out of bounds")
        return self.chunk_texts[chunk_id], self.chunk_metadata[chunk_id]
    
    def __len__(self) -> int:
        """Return number of indexed chunks."""
        return len(self.chunk_texts)
    
    def is_empty(self) -> bool:
        """Check if store is empty."""
        return len(self.chunk_texts) == 0
    
    def clear(self) -> None:
        """Clear all data from the store."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_texts = []
        self.chunk_metadata = []
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save vector store to disk.
        
        Args:
            filepath: Path to save file (without extension).
        """
        filepath = Path(filepath)
        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix(".faiss")))
        # Save texts and metadata
        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump({
                "chunk_texts": self.chunk_texts,
                "chunk_metadata": self.chunk_metadata,
                "dimension": self.dimension
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FAISSVectorStore":
        """
        Load vector store from disk.
        
        Args:
            filepath: Path to load file (without extension).
            
        Returns:
            Loaded FAISSVectorStore instance.
        """
        filepath = Path(filepath)
        # Load FAISS index
        index = faiss.read_index(str(filepath.with_suffix(".faiss")))
        # Load texts and metadata
        with open(filepath.with_suffix(".pkl"), "rb") as f:
            data = pickle.load(f)
            
        store = cls(data["dimension"])
        store.index = index
        store.chunk_texts = data["chunk_texts"]
        store.chunk_metadata = data["chunk_metadata"]
        return store