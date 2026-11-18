"""
Document Loader Module

Handles loading and text extraction from PDF and TXT documents.
Supports file-like objects (e.g., from Streamlit uploaders) and file paths.
"""

from pathlib import Path
from typing import Union, BinaryIO
import pypdf


class DocumentLoadError(Exception):
    """Custom exception for document loading errors."""
    pass


def load_txt(file_input: Union[str, Path, BinaryIO]) -> str:
    """
    Load text from a TXT file.
    
    Args:
        file_input: File path (str/Path) or file-like object.
        
    Returns:
        Extracted text content.
        
    Raises:
        DocumentLoadError: If file cannot be read or is empty.
    """
    try:
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                raise DocumentLoadError(f"File not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = file_input.read().decode("utf-8")
            
        if not content or not content.strip():
            raise DocumentLoadError("Document is empty or contains only whitespace")
            
        return content.strip()
        
    except UnicodeDecodeError as e:
        raise DocumentLoadError(f"Failed to decode file as UTF-8: {e}")
    except Exception as e:
        if isinstance(e, DocumentLoadError):
            raise
        raise DocumentLoadError(f"Error reading TXT file: {e}")


def load_pdf(file_input: Union[str, Path, BinaryIO]) -> str:
    """
    Load text from a PDF file.
    
    Args:
        file_input: File path (str/Path) or file-like object.
        
    Returns:
        Extracted text content from all pages.
        
    Raises:
        DocumentLoadError: If PDF cannot be read or contains no extractable text.
    """
    try:
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                raise DocumentLoadError(f"File not found: {path}")
            reader = pypdf.PdfReader(path)
        else:
            reader = pypdf.PdfReader(file_input)
            
        if len(reader.pages) == 0:
            raise DocumentLoadError("PDF has no pages")
            
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                raise DocumentLoadError(f"Error extracting text from page {page_num + 1}: {e}")
                
        if not text_parts:
            raise DocumentLoadError("No extractable text found in PDF")
            
        full_text = "\n\n".join(text_parts).strip()
        
        if not full_text:
            raise DocumentLoadError("PDF contains no readable text content")
            
        return full_text
        
    except pypdf.errors.PdfReadError as e:
        raise DocumentLoadError(f"Invalid or corrupted PDF file: {e}")
    except Exception as e:
        if isinstance(e, DocumentLoadError):
            raise
        raise DocumentLoadError(f"Error reading PDF file: {e}")


def load_document(file_input: Union[str, Path, BinaryIO], file_extension: str = None) -> str:
    """
    Load document text based on file extension.
    
    Args:
        file_input: File path (str/Path) or file-like object.
        file_extension: Optional file extension (e.g., '.pdf', '.txt').
                        If not provided, inferred from file path.
                        
    Returns:
        Extracted text content.
        
    Raises:
        DocumentLoadError: If format is unsupported or loading fails.
    """
    if file_extension is None:
        if isinstance(file_input, (str, Path)):
            file_extension = Path(file_input).suffix.lower()
        else:
            raise DocumentLoadError("File extension must be provided for file-like objects")
            
    if file_extension == ".txt":
        return load_txt(file_input)
    elif file_extension == ".pdf":
        return load_pdf(file_input)
    else:
        raise DocumentLoadError(f"Unsupported file format: {file_extension}. Supported formats: .txt, .pdf")


def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename.
    
    Args:
        filename: Name of the file.
        
    Returns:
        Lowercase file extension including dot (e.g., '.pdf').
    """
    return Path(filename).suffix.lower()


def is_supported_format(filename: str) -> bool:
    """
    Check if file format is supported.
    
    Args:
        filename: Name of the file or file extension (e.g., 'test.txt' or '.txt').
        
    Returns:
        True if format is supported, False otherwise.
    """
    ext = get_file_extension(filename)
    # Also check if the input itself is a valid extension
    if not ext and filename.startswith('.'):
        ext = filename.lower()
    return ext in {".txt", ".pdf"}