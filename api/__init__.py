# financial_kg/api/__init__.py
"""FastAPI web service components."""

from .routes import router
from .schemas import QueryRequest, QueryResponse

__all__ = ["router", "QueryRequest", "QueryResponse"]
