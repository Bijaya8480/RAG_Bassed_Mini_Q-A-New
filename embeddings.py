"""
Embeddings Module

Generates semantic embeddings using Sentence Transformers (all-MiniLM-L6-v2).
Provides normalized embeddings for cosine similarity via inner product.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingError(Exception):
    """Custom exception for embedding generation errors."""
    pass


class EmbeddingGenerator:
    """
    Generates normalized embeddings using Sentence Transformers.
    
    Uses all-MiniLM-L6-v2 by default (384-dimensional embeddings).
    Embeddings are L2-normalized for cosine similarity via inner product.
    
    Example:
        generator = EmbeddingGenerator()
        embeddings = generator.encode(["text 1", "text 2"])
        query_embedding = generator.encode("query text")
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence Transformer model name (default: all-MiniLM-L6-v2).
        """
        self.model_name = model_name
        self._model = None
        self._embedding_dim = None
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
                # Use get_embedding_dimension (new API) with fallback
                if hasattr(self._model, 'get_embedding_dimension'):
                    self._embedding_dim = self._model.get_embedding_dimension()
                else:
                    self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                raise EmbeddingError(f"Failed to load model '{self.model_name}': {e}")
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            _ = self.model  # triggers lazy load
        return self._embedding_dim
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings.
            normalize: Whether to L2-normalize embeddings (default: True).
            show_progress_bar: Whether to show progress bar (default: False).
            
        Returns:
            numpy array of shape (n_texts, embedding_dim) or (embedding_dim,) for single text.
            
        Raises:
            EmbeddingError: If encoding fails.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                single_input = True
            else:
                single_input = False
                
            if not texts:
                raise EmbeddingError("No texts provided for encoding")
                
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            
            if single_input:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a large batch of texts.
        
        Args:
            texts: List of text strings.
            batch_size: Batch size for encoding (default: 32).
            normalize: Whether to L2-normalize embeddings (default: True).
            
        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
            
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch, normalize=normalize)
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings)
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.
            
        Returns:
            Cosine similarity score (0-1 for normalized vectors).
        """
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
            
        return float(np.dot(emb1, emb2.T)[0, 0])


def create_embedding_generator(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingGenerator:
    """
    Factory function to create an embedding generator.
    
    Args:
        model_name: Sentence Transformer model name.
        
    Returns:
        Configured EmbeddingGenerator instance.
    """
    return EmbeddingGenerator(model_name)