# financial_kg/__init__.py
"""Financial Knowledge Graph - Open Source LLM Version."""

__version__ = "1.0.0"
__author__ = "Abdul Basit Tonmoy"
__description__ = "Open Source Financial Knowledge Graph with RAG capabilities"

from .engine.processing_engine import FinancialKGEngine
from .models.data_models import Entity, Relationship, Document, QueryResult

__all__ = ["FinancialKGEngine", "Entity", "Relationship", "Document", "QueryResult"]





