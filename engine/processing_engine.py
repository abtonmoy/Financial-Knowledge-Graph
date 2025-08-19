"""Main processing engine that orchestrates all components."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..models.model_manager import OpenSourceModelManager
from ..models.data_models import Document, QueryResult, AuditIssue
from ..parsers.pdf_parser import PDFParser
from ..parsers.excel_parser import ExcelParser
from ..extractors.entity_extractor import FinancialEntityExtractor
from ..extractors.relationship_extractor import HybridRelationshipExtractor
from ..storage.knowledge_graph import SimpleKnowledgeGraph
from ..rag.generator import OpenSourceRAG
from ..utils.text_utils import TextProcessor
from ..utils.audit_utils import AuditEngine
from ..config import get_config


class FinancialKGEngine:
    """Main engine that coordinates all components for financial knowledge graph processing."""

    def __init__(self, device: str = "auto"):
        print("Initializing Financial Knowledge Graph...")

        self.config = get_config()

        # Initialize core components
        self.model_manager = OpenSourceModelManager(device)
        self.text_processor = TextProcessor()

        # Initialize parsers
        self.parsers = {
            "pdf": PDFParser(),
            "excel": ExcelParser(),
        }

        # Initialize extractors
        self.entity_extractor = FinancialEntityExtractor(self.model_manager)
        self.relationship_extractor = HybridRelationshipExtractor(self.model_manager)

        # Initialize storage
        self.knowledge_graph = SimpleKnowledgeGraph()

        # Initialize RAG system
        self.rag_system = OpenSourceRAG(self.model_manager, self.knowledge_graph)

        # Initialize audit engine
        self.audit_engine = AuditEngine(self.knowledge_graph)

        print("Financial KG Engine ready!")

    def process_document(self, file_path: str, doc_id: Optional[str] = None) -> str:
        """
        Process a document end-to-end.

        Args:
            file_path: Path to the document file
            doc_id: Optional document ID (auto-generated if not provided)

        Returns:
            Document ID of the processed document

        Raises:
            ValueError: If file type is not supported
            RuntimeError: If processing fails
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        print(f"Processing Document: {file_path}")

        try:
            # Step 1: Determine parser and parse document
            parser = self._get_parser_for_file(file_path)
            if not parser:
                raise ValueError(f"Unsupported file type: {file_path}")

            parsed = parser.parse(file_path)
            text = parsed.get("text", "") or ""
            tables = parsed.get("tables", []) or []
            print(f"Extracted {len(text)} characters of text and {len(tables)} tables")

            # Step 2: Extract entities from text
            entities = self.entity_extractor.extract_entities(text, doc_id)
            print(f"Extracted {len(entities)} entities from text")

            # Step 3: Extract entities from tables
            for table in tables:
                table_entities = self.entity_extractor.extract_from_table(table, doc_id)
                entities.extend(table_entities)
                print(f"Extracted {len(table_entities)} entities from table")

            # Step 4: Extract relationships using hybrid approach
            relationships = self.relationship_extractor.extract_relationships(
                entities=entities,
                text=text,
                doc_id=doc_id,
                tables=tables
            )
            print(f"Extracted {len(relationships)} relationships")

            # Step 5: Create document object
            document = Document.create(
                filename=Path(file_path).name,
                file_type=Path(file_path).suffix.lstrip(".").lower(),
                text_content=text,
                tables=tables,
                metadata=parsed.get("metadata", {}) or {},
            )
            document.id = doc_id
            document.entities = entities
            document.relationships = relationships

            # Step 6: Store in knowledge graph
            success = self.knowledge_graph.add_document(document=document)
            if not success:
                raise RuntimeError("Failed to store document in KG")

            # Store entities
            for entity in entities:
                self.knowledge_graph.add_entity(entity)

            # Store relationships
            for relationship in relationships:
                self.knowledge_graph.add_relationship(relationship)

            # Add to vector store for RAG
            self.rag_system.add_document_to_vector_store(document)
            print(f"Document processed successfully: {doc_id}")

            return doc_id

        except Exception as e:
            print(f"Error processing document: {e}")
            raise RuntimeError(f"Document processing failed: {e}")

    def _get_parser_for_file(self, file_path: str):
        """
        Get the appropriate parser for a file.

        Args:
            file_path: Path to the file

        Returns:
            Parser instance or None if no suitable parser found
        """
        for parser in self.parsers.values():
            try:
                if parser.supports_file_type(file_path):
                    return parser
            except Exception:
                # Ignore individual parser errors during capability check
                continue
        return None

    def get_entities(
        self,
        entity_type: Optional[str] = None,
        source_doc: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get entities from the knowledge graph.

        Args:
            entity_type: Filter by entity type
            source_doc: Filter by source document
            limit: Maximum number of entities to return

        Returns:
            List of entity dictionaries
        """
        return self.knowledge_graph.query_entities(
            entity_type=entity_type, source_doc=source_doc, limit=limit
        )

    def get_relationships(
        self,
        relationship_type: Optional[str] = None,
        source_entity_id: Optional[str] = None,
        target_entity_id: Optional[str] = None,
        source_doc: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get relationships from the knowledge graph.

        Args:
            relationship_type: Filter by relationship type
            source_entity_id: Filter by source entity
            target_entity_id: Filter by target entity
            source_doc: Filter by source document
            limit: Maximum number of relationships to return

        Returns:
            List of relationship dictionaries
        """
        return self.knowledge_graph.query_relationships(
            relationship_type=relationship_type,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            source_doc=source_doc,
            limit=limit
        )

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Get a document by its ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dictionary or None if not found
        """
        return self.knowledge_graph.get_document_by_id(doc_id)

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get an entity by its ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity dictionary or None if not found
        """
        return self.knowledge_graph.get_entity_by_id(entity_id)

    def get_related_entities(self, entity_id: str) -> List[Dict]:
        """
        Get entities related to a given entity.

        Args:
            entity_id: ID of the entity to find relations for

        Returns:
            List of related entities
        """
        return self.knowledge_graph.query_related_entities(entity_id)

    def run_audit(self) -> List[AuditIssue]:
        """
        Run audit checks on the knowledge base.

        Returns:
            List of audit issues found
        """
        print("Running audit checks...")

        try:
            issues = self.audit_engine.run_all_checks()
            print(f"Audit completed. Found {len(issues)} issues.")
            return issues
        except Exception as e:
            print(f"Error running audit: {e}")
            return [
                AuditIssue(
                    type="audit_error",
                    severity="high",
                    description=f"Audit failed: {str(e)}",
                )
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with various statistics
        """
        kg_stats = self.knowledge_graph.get_statistics()
        vector_stats = self.rag_system.get_vector_store_stats()
        model_stats = {
            "loaded_models": self.model_manager.get_loaded_models(),
            "device": self.model_manager.device,
        }

        return {
            "knowledge_graph": kg_stats,
            "vector_store": vector_stats,
            "models": model_stats,
        }

    def clear_all_data(self) -> bool:
        """
        Clear all data from the knowledge graph and vector store.

        Returns:
            True if successful
        """
        print("Clearing all data...")

        try:
            kg_cleared = self.knowledge_graph.clear_all_data()
            vector_cleared = self.rag_system.clear_vector_store()

            if kg_cleared and vector_cleared:
                print("All data cleared successfully")
                return True
            else:
                print("Some data may not have been cleared")
                return False
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False

    def export_entities(self, file_path: str, entity_type: Optional[str] = None) -> bool:
        """
        Export entities to a JSON file.

        Args:
            file_path: Path to save the export file
            entity_type: Optional entity type filter

        Returns:
            True if successful
        """
        try:
            import json

            entities = self.get_entities(entity_type=entity_type, limit=10000)
            relationships = self.get_relationships(limit=10000)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "export_timestamp": datetime.now().isoformat(),
                        "entity_type_filter": entity_type,
                        "total_entities": len(entities),
                        "total_relationships": len(relationships),
                        "entities": entities,
                        "relationships": relationships,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"Exported {len(entities)} entities and {len(relationships)} relationships to {file_path}")
            return True

        except Exception as e:
            print(f"Error exporting entities: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all system components.

        Returns:
            Dictionary with health status of each component
        """
        health = {
            "overall_status": "healthy",
            "components": {},
        }

        try:
            # Check model manager
            loaded_models = self.model_manager.get_loaded_models()
            health["components"]["model_manager"] = {
                "status": "healthy",
                "loaded_models": loaded_models,
                "device": self.model_manager.device,
            }
        except Exception as e:
            health["components"]["model_manager"] = {"status": "error", "error": str(e)}
            health["overall_status"] = "degraded"

        try:
            # Check knowledge graph
            kg_stats = self.knowledge_graph.get_statistics()
            health["components"]["knowledge_graph"] = {
                "status": "healthy",
                "stats": kg_stats,
            }
        except Exception as e:
            health["components"]["knowledge_graph"] = {"status": "error", "error": str(e)}
            health["overall_status"] = "degraded"

        try:
            # Check vector store
            vector_stats = self.rag_system.get_vector_store_stats()
            # If vector store returns an error, mark degraded
            if isinstance(vector_stats, dict) and "error" in vector_stats:
                health["components"]["vector_store"] = {
                    "status": "error",
                    "error": vector_stats["error"],
                }
                health["overall_status"] = "degraded"
            else:
                health["components"]["vector_store"] = {
                    "status": "healthy",
                    "stats": vector_stats,
                }
        except Exception as e:
            health["components"]["vector_store"] = {"status": "error", "error": str(e)}
            health["overall_status"] = "degraded"

        return health
    

    def ask_question(self, question: str, top_k: int = 5) -> QueryResult:
        """
        Ask a natural language question using the RAG system.

        Args:
            question: The question string
            top_k: Number of top documents/entities to consider

        Returns:
            QueryResult object with answer and sources
        """
        if not self.rag_system:
            raise RuntimeError("RAG system is not initialized")

        result = self.rag_system.query(question, top_k=top_k)
        return result