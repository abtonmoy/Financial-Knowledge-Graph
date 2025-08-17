"""Command line interface handlers."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..engine.processing_engine import FinancialKGEngine
from ..models.data_models import Document
from ..config import get_config

class CLIHandler:
    """Handles command line interface operations."""
    
    def __init__(self):
        self.config = get_config()
        self.engine = None
    
    def _get_engine(self) -> FinancialKGEngine:
        """Get or create engine instance."""
        if self.engine is None:
            print(" Initializing engine...")
            self.engine = FinancialKGEngine(device=self.config.DEVICE)
        return self.engine
    
    def process_document(self, file_path: str):
        """Process a document via CLI."""
        try:
            if not Path(file_path).exists():
                print(f" File not found: {file_path}")
                return
            
            engine = self._get_engine()
            doc_id = engine.process_document(file_path)
            
            print(f" Document processed successfully!")
            print(f"   Document ID: {doc_id}")
            print(f"   File: {Path(file_path).name}")
            
            # Show basic stats
            entities = engine.get_entities(source_doc=doc_id)
            print(f"   Entities extracted: {len(entities)}")
            
            # Show entity breakdown
            entity_types = {}
            for entity in entities:
                entity_type = entity['type']
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            if entity_types:
                print("   Entity breakdown:")
                for entity_type, count in entity_types.items():
                    print(f"     {entity_type}: {count}")
            
        except Exception as e:
            print(f" Error processing document: {e}")
    
    def query_documents(self, question: str):
        """Query documents via CLI."""
        try:
            engine = self._get_engine()
            result = engine.ask_question(question)
            
            print(f"\n Question: {question}")
            print(f" Answer: {result.answer}")
            
            # Show sources
            doc_sources = result.sources.get('documents', [])
            graph_sources = result.sources.get('graph_facts', [])
            
            if doc_sources:
                print(f"\n Document sources ({len(doc_sources)}):")
                for i, source in enumerate(doc_sources[:3], 1):
                    preview = source[:100] + "..." if len(source) > 100 else source
                    print(f"   {i}. {preview}")
            
            if graph_sources:
                print(f"\n Knowledge graph facts ({len(graph_sources)}):")
                for i, fact in enumerate(graph_sources[:5], 1):
                    fact_text = f"{fact.get('type', 'Entity')}: {fact.get('text', 'Unknown')}"
                    print(f"   {i}. {fact_text}")
            
            # Show metadata
            metadata = result.metadata
            if metadata.get('question_entities'):
                print(f"\n Detected entities: {', '.join(metadata['question_entities'])}")
            
        except Exception as e:
            print(f" Error querying documents: {e}")
    
    def list_entities(self, entity_type: Optional[str] = None):
        """List entities via CLI."""
        try:
            engine = self._get_engine()
            entities = engine.get_entities(entity_type=entity_type, limit=50)
            
            if entity_type:
                print(f" Found {len(entities)} {entity_type} entities:")
            else:
                print(f" Found {len(entities)} entities:")
            
            if not entities:
                print("   No entities found. Process some documents first.")
                return
            
            # Group by type for display
            entities_by_type = {}
            for entity in entities:
                etype = entity['type']
                if etype not in entities_by_type:
                    entities_by_type[etype] = []
                entities_by_type[etype].append(entity)
            
            for etype, type_entities in entities_by_type.items():
                print(f"\n {etype} ({len(type_entities)}):")
                for entity in type_entities[:10]:  # Limit display
                    confidence = entity.get('confidence', 0)
                    text = entity.get('text', 'Unknown')
                    
                    # Show properties if available
                    props_text = ""
                    if entity.get('properties'):
                        props = entity['properties']
                        if isinstance(props, str):
                            try:
                                props = json.loads(props)
                            except:
                                props = {}
                        
                        if props:
                            key_props = []
                            for key in ['amount', 'percentage', 'masked_number']:
                                if key in props:
                                    key_props.append(f"{key}:{props[key]}")
                            if key_props:
                                props_text = f" ({', '.join(key_props)})"
                    
                    print(f"   • {text}{props_text} (confidence: {confidence:.2f})")
                
                if len(type_entities) > 10:
                    print(f"   ... and {len(type_entities) - 10} more")
        
        except Exception as e:
            print(f" Error listing entities: {e}")
    
    def run_audit(self):
        """Run audit checks via CLI."""
        try:
            engine = self._get_engine()
            issues = engine.run_audit()
            
            print(f" Audit Results: Found {len(issues)} issues")
            
            if not issues:
                print(" No issues found. Your financial data looks good!")
                return
            
            # Group by severity
            by_severity = {'high': [], 'medium': [], 'low': []}
            for issue in issues:
                by_severity[issue.severity].append(issue)
            
            for severity in ['high', 'medium', 'low']:
                severity_issues = by_severity[severity]
                if severity_issues:
                    icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                    print(f"\n{icon} {severity.upper()} SEVERITY ({len(severity_issues)}):")
                    
                    for issue in severity_issues:
                        print(f"   • {issue.description}")
                        if issue.recommendations:
                            print(f"     Recommendations: {'; '.join(issue.recommendations[:2])}")
                        print()
        
        except Exception as e:
            print(f" Error running audit: {e}")
    
    def show_statistics(self):
        """Show system statistics via CLI."""
        try:
            engine = self._get_engine()
            stats = engine.get_statistics()
            
            print(" System Statistics")
            print("=" * 50)
            
            # Knowledge Graph stats
            kg_stats = stats.get('knowledge_graph', {})
            print("\n Knowledge Graph:")
            print(f"   Total entities: {kg_stats.get('total_entities', 0)}")
            print(f"   Total relationships: {kg_stats.get('total_relationships', 0)}")
            print(f"   Total documents: {kg_stats.get('total_documents', 0)}")
            
            entities_by_type = kg_stats.get('entities_by_type', {})
            if entities_by_type:
                print("   Entities by type:")
                for entity_type, count in entities_by_type.items():
                    print(f"     {entity_type}: {count}")
            
            # Vector Store stats
            vector_stats = stats.get('vector_store', {})
            print(f"\n Vector Store:")
            print(f"   Total chunks: {vector_stats.get('total_chunks', 0)}")
            print(f"   Collection: {vector_stats.get('collection_name', 'N/A')}")
            
            # Model stats
            model_stats = stats.get('models', {})
            print(f"\n Models:")
            print(f"   Device: {model_stats.get('device', 'unknown')}")
            loaded_models = model_stats.get('loaded_models', [])
            if loaded_models:
                print(f"   Loaded models: {', '.join(loaded_models)}")
            else:
                print("   No models currently loaded")
        
        except Exception as e:
            print(f"Error retrieving statistics: {e}")
    
    def run_test(self):
        """Run test with sample financial data."""
        print("🧪 Running test with sample financial text...")
        
        # Create sample financial document content
        sample_text = """
        FINANCIAL STATEMENT - ACME CORPORATION
        Statement Date: March 31, 2024
        
        ACCOUNT SUMMARY:
        Checking Account: 123456789
        Current Balance: $25,847.32
        
        RECENT TRANSACTIONS:
        03/28/2024    Office Supplies Inc.     -$1,245.67
        03/29/2024    Client Payment - ABC Co   +$5,500.00
        03/30/2024    Salary - John Smith       -$3,200.00
        03/31/2024    Rent Payment             -$2,800.00
        
        SUMMARY:
        Total Income: $5,500.00
        Total Expenses: $7,245.67
        Net Change: -$1,745.67
        
        Account Holder: Jane Doe (CEO)
        Interest Rate: 2.5%
        Service Fee: $25.00
        """
        
        try:
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(sample_text)
                temp_path = f.name
            
            try:
                engine = self._get_engine()
                
                # Process the sample document
                print(" Processing sample financial document...")
                doc_id = str(uuid.uuid4())
                
                # Create document and extract entities
                from ..extractors.entity_extractor import FinancialEntityExtractor
                extractor = FinancialEntityExtractor(engine.model_manager)
                entities = extractor.extract_entities(sample_text, doc_id)
                
                print(f" Extracted {len(entities)} entities:")
                
                # Display entities by type
                entity_types = {}
                for entity in entities:
                    entity_type = entity.type
                    if entity_type not in entity_types:
                        entity_types[entity_type] = []
                    entity_types[entity_type].append(entity)
                
                for entity_type, type_entities in entity_types.items():
                    print(f"\n {entity_type} ({len(type_entities)}):")
                    for entity in type_entities[:10]:
                        props_text = ""
                        if entity.properties:
                            key_props = []
                            for key in ['amount', 'percentage', 'masked_number']:
                                if key in entity.properties:
                                    key_props.append(f"{key}:{entity.properties[key]}")
                            if key_props:
                                props_text = f" ({', '.join(key_props)})"
                        
                        print(f"   • {entity.text}{props_text} (confidence: {entity.confidence:.2f})")
                
                # Create and store document for testing queries
                document = Document.create(
                    filename="sample_statement.txt",
                    file_type=".txt",
                    text_content=sample_text
                )
                document.id = doc_id
                document.entities = entities
                
                # Store in knowledge graph
                engine.knowledge_graph.add_document(document)
                for entity in entities:
                    engine.knowledge_graph.add_entity(entity)
                
                # Add to vector store
                engine.rag_system.add_document_to_vector_store(document)
                
                # Test some questions
                print("\n Testing sample questions...")
                
                test_questions = [
                    "What is the account number?",
                    "What is the current balance?",
                    "How much was paid for rent?",
                    "Who is the account holder?"
                ]
                
                for question in test_questions:
                    try:
                        result = engine.ask_question(question)
                        answer = result.answer[:100] + "..." if len(result.answer) > 100 else result.answer
                        print(f"   Q: {question}")
                        print(f"   A: {answer}")
                        print()
                    except Exception as e:
                        print(f"   Q: {question}")
                        print(f"   A: Error - {e}")
                        print()
                
                print("Test completed successfully!")
            
            finally:
                # Clean up temp file
                import os
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except Exception as e:
            print(f"Error running test: {e}")
    
    def clear_data(self):
        """Clear all data via CLI."""
        try:
            print("This will permanently delete all processed documents, entities, and vector data.")
            response = input("Are you sure? (yes/no): ").lower().strip()
            
            if response != 'yes':
                print("Operation cancelled.")
                return
            
            engine = self._get_engine()
            success = engine.clear_all_data()
            
            if success:
                print("All data cleared successfully!")
            else:
                print("Some data may not have been cleared completely.")
        
        except Exception as e:
            print(f"Error clearing data: {e}")