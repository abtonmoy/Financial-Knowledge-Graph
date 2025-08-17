"""FastAPI routes for the Financial Knowledge Graph API."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from pathlib import Path
from typing import Optional, List
import tempfile
import os

from .schemas import (
    QueryRequest, QueryResponse, EntityFilter, EntityResponse,
    DocumentResponse, AuditResponse, StatisticsResponse,
    HealthStatus, UploadResponse, ModelStatusResponse,
    ExportRequest, ClearDataRequest, ErrorResponse
)
from ..engine.processing_engine import FinancialKGEngine

# Global engine instance (initialized in main app)
engine: Optional[FinancialKGEngine] = None

def get_engine() -> FinancialKGEngine:
    """Get the engine instance or raise an error if not initialized."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    return engine

# Create router
router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and process a financial document."""
    current_engine = get_engine()
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ['.pdf', '.xlsx', '.xls']:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Supported types: .pdf, .xlsx, .xls"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_path = temp_file.name
        try:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Process the document
            doc_id = current_engine.process_document(temp_path)
            
            return UploadResponse(
                document_id=doc_id, 
                status="processed", 
                filename=file.filename,
                message=f"Successfully processed {file.filename}"
            )
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Query the financial knowledge base."""
    current_engine = get_engine()
    
    try:
        result = current_engine.ask_question(request.question)
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            metadata=result.metadata,
            confidence=result.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing query: {str(e)}")

@router.get("/entities", response_model=EntityResponse)
async def get_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    source_doc: Optional[str] = Query(None, description="Filter by source document ID"),
    text_contains: Optional[str] = Query(None, description="Filter by text content"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of entities to return")
) -> EntityResponse:
    """Get entities from the knowledge graph."""
    current_engine = get_engine()
    
    try:
        entities = current_engine.knowledge_graph.query_entities(
            entity_type=entity_type,
            source_doc=source_doc,
            text_contains=text_contains,
            min_confidence=min_confidence,
            limit=limit
        )
        
        return EntityResponse(
            entities=entities, 
            count=len(entities),
            total_available=None  # Could implement total count if needed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving entities: {str(e)}")

@router.get("/entities/{entity_id}")
async def get_entity_by_id(entity_id: str):
    """Get a specific entity by ID."""
    current_engine = get_engine()
    
    try:
        entity = current_engine.get_entity_by_id(entity_id)
        
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        return {"entity": entity, "found": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving entity: {str(e)}")

@router.get("/entities/{entity_id}/related")
async def get_related_entities(entity_id: str):
    """Get entities related to a specific entity."""
    current_engine = get_engine()
    
    try:
        related = current_engine.get_related_entities(entity_id)
        return {"related_entities": related, "count": len(related)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving related entities: {str(e)}")

@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document_by_id(doc_id: str) -> DocumentResponse:
    """Get a specific document by ID."""
    current_engine = get_engine()
    
    try:
        document = current_engine.get_document_by_id(doc_id)
        
        if not document:
            return DocumentResponse(document=None, found=False)
        
        # Count entities for this document
        entities = current_engine.get_entities(source_doc=doc_id)
        
        from datetime import datetime
        doc_info = {
            "id": document["id"],
            "filename": document["filename"],
            "file_type": document["file_type"],
            "processed_at": datetime.fromisoformat(document["processed_at"]),
            "entity_count": len(entities),
            "metadata": document.get("metadata", {})
        }
        
        return DocumentResponse(document=doc_info, found=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@router.post("/audit", response_model=AuditResponse)
async def run_audit() -> AuditResponse:
    """Run audit checks on the knowledge base."""
    current_engine = get_engine()
    
    try:
        issues = current_engine.run_audit()
        
        # Convert issues to response format
        issue_responses = []
        for issue in issues:
            issue_responses.append({
                "type": issue.type,
                "severity": issue.severity,
                "description": issue.description,
                "entity_count": len(issue.entities),
                "recommendations": issue.recommendations,
                "metadata": issue.metadata
            })
        
        # Generate summary
        summary = {
            "high_severity": len([i for i in issues if i.severity == "high"]),
            "medium_severity": len([i for i in issues if i.severity == "medium"]),
            "low_severity": len([i for i in issues if i.severity == "low"])
        }
        
        return AuditResponse(
            issues=issue_responses,
            summary=summary,
            total_issues=len(issues)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running audit: {str(e)}")

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics() -> StatisticsResponse:
    """Get comprehensive system statistics."""
    current_engine = get_engine()
    
    try:
        stats = current_engine.get_statistics()
        return StatisticsResponse(
            knowledge_graph=stats["knowledge_graph"],
            vector_store=stats["vector_store"],
            models=stats["models"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Perform a health check of all system components."""
    try:
        current_engine = get_engine()
        health = current_engine.health_check()
        return HealthStatus(
            overall_status=health["overall_status"],
            components=health["components"]
        )
    except Exception as e:
        # If engine is not available, return degraded status
        from datetime import datetime
        return HealthStatus(
            overall_status="error",
            components={"engine": {"status": "error", "error": str(e)}},
            timestamp=datetime.now()
        )

@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """Check which models are loaded."""
    current_engine = get_engine()
    
    try:
        loaded_models = current_engine.model_manager.get_loaded_models()
        device = current_engine.model_manager.device
        available_models = current_engine.config.MODELS
        
        return ModelStatusResponse(
            device=device,
            loaded_models=loaded_models,
            available_models=available_models
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model status: {str(e)}")

@router.post("/export/entities")
async def export_entities(request: ExportRequest):
    """Export entities to a file."""
    current_engine = get_engine()
    
    try:
        # Create temporary file for export
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{request.format}', delete=False) as temp_file:
            temp_path = temp_file.name
            
            success = current_engine.export_entities(temp_path, request.entity_type)
            
            if success:
                return {"export_path": temp_path, "status": "success"}
            else:
                raise HTTPException(status_code=500, detail="Export failed")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting entities: {str(e)}")

@router.post("/admin/clear")
async def clear_all_data(request: ClearDataRequest):
    """Clear all data from the system (admin only)."""
    current_engine = get_engine()
    
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Must confirm data deletion")
    
    try:
        success = current_engine.clear_all_data()
        
        if success:
            return {"status": "success", "message": "All data cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear all data")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@router.get("/documents")
async def list_documents(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of documents to return")
):
    """List all processed documents."""
    current_engine = get_engine()
    
    try:
        # Get all entities to find unique documents
        all_entities = current_engine.get_entities(limit=10000)
        doc_ids = list(set(e.get('source_doc') for e in all_entities if e.get('source_doc')))
        
        documents = []
        for doc_id in doc_ids[:limit]:
            doc = current_engine.get_document_by_id(doc_id)
            if doc:
                entity_count = len([e for e in all_entities if e.get('source_doc') == doc_id])
                documents.append({
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "processed_at": doc["processed_at"],
                    "entity_count": entity_count
                })
        
        return {"documents": documents, "count": len(documents), "total_available": len(doc_ids)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

def set_engine(app_engine: FinancialKGEngine):
    """Set the global engine instance."""
    global engine
    engine = app_engine