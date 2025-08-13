"""Base parser interface for document parsing."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from pathlib import Path

class BaseParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document and extract text and structured data.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - tables: List of extracted tables
                - metadata: Additional metadata about the document
        """
        pass
    
    @abstractmethod
    def supports_file_type(self, file_path: str) -> bool:
        """
        Check if this parser supports the given file type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if this parser can handle the file type
        """
        pass
    
    def get_file_extension(self, file_path: str) -> str:
        """
        Get the file extension.
        """
        return Path(file_path).suffix.lower()
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate that the file exists and is readable.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid and readable
        """
        path = Path(file_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0