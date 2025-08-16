# financial_kg/__init__.py
"""Financial Knowledge Graph - Open Source LLM Version."""

__version__ = "1.0.0"
__author__ = "Abdul Basit Tonmoy"
__description__ = "Open Source Financial Knowledge Graph with RAG capabilities"

from .engine.processing_engine import FinancialKGEngine
from .models.data_models import Entity, Relationship, Document, QueryResult

__all__ = ["FinancialKGEngine", "Entity", "Relationship", "Document", "QueryResult"]

# financial_kg/models/__init__.py
"""Data models and model management."""

from .data_models import Entity, Relationship, Document, QueryResult, AuditIssue
from .model_manager import OpenSourceModelManager

__all__ = ["Entity", "Relationship", "Document", "QueryResult", "AuditIssue", "OpenSourceModelManager"]

# financial_kg/parsers/__init__.py
"""Document parsers for various file formats."""

from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .excel_parser import ExcelParser

__all__ = ["BaseParser", "PDFParser", "ExcelParser"]

# financial_kg/extractors/__init__.py
"""Entity and relationship extractors."""

from .entity_extractor import FinancialEntityExtractor

__all__ = ["FinancialEntityExtractor"]

# financial_kg/storage/__init__.py
"""Data storage components."""

from .knowledge_graph import SimpleKnowledgeGraph

__all__ = ["SimpleKnowledgeGraph"]

# financial_kg/rag/__init__.py
"""Retrieval Augmented Generation components."""

from .generator import OpenSourceRAG

__all__ = ["OpenSourceRAG"]

# financial_kg/api/__init__.py
"""FastAPI web service components."""

from .routes import router
from .schemas import QueryRequest, QueryResponse

__all__ = ["router", "QueryRequest", "QueryResponse"]

# financial_kg/engine/__init__.py
"""Core processing engine."""

from .processing_engine import FinancialKGEngine

__all__ = ["FinancialKGEngine"]

# financial_kg/cli/__init__.py
"""Command line interface components."""

from .commands import CLIHandler

__all__ = ["CLIHandler"]

# financial_kg/utils/__init__.py
"""Utility functions and classes."""

from .text_utils import TextProcessor
from .audit_utils import AuditEngine

__all__ = ["TextProcessor", "AuditEngine"]