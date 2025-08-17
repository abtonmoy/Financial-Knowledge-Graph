"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """Request schema for document queries."""
    question: str = Field(..., description="Question to ask about the documents", min_length=1)

class QueryResponse(BaseModel):
    """Response schema for document queries."""
    answer: str
    sources: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence: Optional[float] = None

class EntityFilter(BaseModel):
    """Filter parameters for entity queries."""
    entity_type: Optional[str] = None
    source_doc: Optional[str] = None
    text_contains: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    limit: int = Field(100, ge=1, le=1000)

class EntityResponse(BaseModel):
    """Response schema for entity queries."""
    entities: List[Dict[str, Any]]
    count: int
    total_available: Optional[int] = None

class DocumentInfo(BaseModel):
    """Document information schema."""
    id: str
    filename: str
    file_type: str
    processed_at: datetime
    entity_count: int
    metadata: Dict[str, Any]

class DocumentResponse(BaseModel):
    """Response schema for document information."""
    document: Optional[DocumentInfo] = None
    found: bool

class AuditIssueResponse(BaseModel):
    """Response schema for individual audit issues."""
    type: str
    severity: str
    description: str
    entity_count: int
    recommendations: List[str]
    metadata: Dict[str, Any]

class AuditResponse(BaseModel):
    """Response schema for audit results."""
    issues: List[AuditIssueResponse]
    summary: Dict[str, Any]
    total_issues: int

class StatisticsResponse(BaseModel):
    """Response schema for system statistics."""
    knowledge_graph: Dict[str, Any]
    vector_store: Dict[str, Any]
    models: Dict[str, Any]

class HealthStatus(BaseModel):
    """Health check response schema."""
    overall_status: str
    components: Dict[str, Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)

class UploadResponse(BaseModel):
    """Response schema for document uploads."""
    document_id: str
    status: str
    filename: str
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelStatusResponse(BaseModel):
    """Response schema for model status."""
    device: str
    loaded_models: List[str]
    available_models: Dict[str, Dict[str, str]]

class ExportRequest(BaseModel):
    """Request schema for data export."""
    entity_type: Optional[str] = None
    format: str = Field("json", regex="^(json|csv)$")
    include_metadata: bool = True

class ClearDataRequest(BaseModel):
    """Request schema for clearing data."""
    confirm: bool = Field(..., description="Must be true to confirm data deletion")
    clear_vector_store: bool = True
    clear_knowledge_graph: bool = True