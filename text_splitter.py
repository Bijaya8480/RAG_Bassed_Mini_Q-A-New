"""
Text Splitter Module

Splits documents into overlapping word-based chunks for better retrieval.
Default chunk size: 300 words, overlap: 50 words.
"""

from typing import List


class TextSplitterError(Exception):
    """Custom exception for text splitting errors."""
    pass


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text.
        
    Returns:
        Number of words.
    """
    return len(text.split())


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """
    Validate chunk size and overlap parameters.
    
    Args:
        chunk_size: Target chunk size in words.
        chunk_overlap: Overlap between chunks in words.
        
    Raises:
        TextSplitterError: If parameters are invalid.
    """
    if chunk_size <= 0:
        raise TextSplitterError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise TextSplitterError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise TextSplitterError("chunk_overlap must be less than chunk_size")


def split_text_into_chunks(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    
    Args:
        text: Input text to split.
        chunk_size: Target number of words per chunk (default: 300).
        chunk_overlap: Number of overlapping words between chunks (default: 50).
        
    Returns:
        List of text chunks.
        
    Raises:
        TextSplitterError: If parameters are invalid or text is empty.
    """
    validate_chunk_params(chunk_size, chunk_overlap)
    
    if not text or not text.strip():
        raise TextSplitterError("Input text is empty")
        
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        raise TextSplitterError("Input text contains no words")
    
    chunks = []
    start = 0
    
    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
            
        if end == total_words:
            break
            
        start += chunk_size - chunk_overlap
        
    if not chunks:
        raise TextSplitterError("No valid chunks generated")
        
    return chunks


def split_text_into_chunks_with_metadata(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50
) -> List[dict]:
    """
    Split text into chunks with metadata (chunk index, word count, etc.).
    
    Args:
        text: Input text to split.
        chunk_size: Target number of words per chunk (default: 300).
        chunk_overlap: Number of overlapping words between chunks (default: 50).
        
    Returns:
        List of dictionaries containing chunk text and metadata.
    """
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
    
    result = []
    for idx, chunk in enumerate(chunks):
        result.append({
            "chunk_id": idx,
            "text": chunk,
            "word_count": count_words(chunk),
            "start_word": idx * (chunk_size - chunk_overlap),
            "end_word": idx * (chunk_size - chunk_overlap) + count_words(chunk)
        })
        
    return result


class TextSplitter:
    """
    Configurable text splitter for document chunking.
    
    Example:
        splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split("Your document text here...")
    """
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Target chunk size in words (default: 300).
            chunk_overlap: Overlap between chunks in words (default: 50).
        """
        validate_chunk_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text.
            
        Returns:
            List of text chunks.
        """
        return split_text_into_chunks(text, self.chunk_size, self.chunk_overlap)
        
    def split_with_metadata(self, text: str) -> List[dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Input text.
            
        Returns:
            List of chunk dictionaries with metadata.
        """
        return split_text_into_chunks_with_metadata(text, self.chunk_size, self.chunk_overlap)