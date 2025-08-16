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
            'pdf': PDFParser(),
            'excel': ExcelParser()
        }
        
        # Initialize extractors
        self.entity_extractor = FinancialEntityExtractor(self.model_manager)
        
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
            # s1 -> Determine parser and parse document
            parser = self._get_parser_for_file(file_path)
            if not parser:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            parsed = parser.parse(file_path)
            print(f"Extracted {len(parsed['text'])} char of text")

            # s2 -> extract entities from text
            entities = self._entity_extractor.extract_entities(parsed['text'], doc_id)
            print(f"Extracted {len(entities)} entities")

            # s3 -> extract entries from tables
            tables = parsed.get('tables', [])
            for table in tables:
                table_entities = self.entity_extractor.extract_from_table(table, doc_id)
                entities.extend(table_entities)
                print(f"Extracted {len(table_entities)} entities from table")
            
            # s4 -> create document obj
            document = Document.create(
                filename=Path(file_path).name,
                file_type=Path(file_path).suffix,
                text_content=parsed['text'],
                tables=parsed['tables'],
                metadata=parsed.get('metadata', {})
            )
            document.id = doc_id
            document.entities = entities

            # s5 -> store in kg
            success = self.knowledge_graph.add_document(document=document)
            if not success:
                raise RuntimeError("Failed to store document in KG")
            
            # store entities
            for entity in entities:
                self.knowledge_graph.add_entity(entity)

            # add to vector store for RAG
            self.rag_system.add_document_to_vector_store(document)
            print(f"Document processed successfully: {doc_id}")
        except Exception as e:
            print(f"Error processing document: {e}")
            raise RuntimeError(f"Document processing failed: {e}")
        