"""Data models for the Financial Knowledge Graph."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List, Literal
import uuid


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""

    type: str  # PERSON, COMPANY, AMOUNT, DATE, ACCOUNT
    text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: Optional[float] = None
    source_doc: str = ""
    position: Tuple[int, int] = (0, 0)  # start, end
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        entity_type: str,
        text: str,
        confidence: float,
        source_doc: str,
        position: Tuple[int, int],
        properties: Optional[Dict[str, Any]] = None
    ) -> "Entity":
        """Create a new entity with auto-generated ID."""
        return cls(
            type=entity_type,
            text=text,
            id=str(uuid.uuid4()),
            confidence=confidence,
            source_doc=source_doc,
            position=position,
            properties=properties or {}
        )


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    source_entity_id: str
    target_entity_id: str
    type: str  # OWNS, PAYS, TRANSFERS_TO, etc.
    confidence: float
    source_doc: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        confidence: float,
        source_doc: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> "Relationship":
        """Create a new relationship with auto-generated ID."""
        return cls(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            type=relationship_type,
            confidence=confidence,
            source_doc=source_doc,
            id=str(uuid.uuid4()),
            properties=properties or {}
        )


@dataclass
class Document:
    """Represents a processed document."""

    filename: str
    file_type: str
    text_content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tables: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        filename: str,
        file_type: str,
        text_content: str,
        tables: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "Document":
        """Create a new document with default empty lists for entities and relationships."""
        return cls(
            filename=filename,
            file_type=file_type,
            text_content=text_content,
            tables=tables or [],
            metadata=metadata or {}
        )


@dataclass
class QueryResult:
    """Result of a RAG query."""

    answer: str
    sources: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence: Optional[float] = None

    @classmethod
    def create(
        cls,
        answer: str,
        documents: Optional[List[str]] = None,
        graph_facts: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None
    ) -> "QueryResult":
        """Create a query result."""
        return cls(
            answer=answer,
            sources={
                "documents": documents or [],
                "graph_facts": graph_facts or []
            },
            metadata=metadata or {},
            confidence=confidence
        )


@dataclass
class AuditIssue:
    """Represents an audit issue found in the data."""

    type: str
    severity: Literal["low", "medium", "high"]
    description: str
    entities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
