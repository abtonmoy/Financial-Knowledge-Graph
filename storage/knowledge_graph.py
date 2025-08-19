"""SQLite-based knowledge graph storage with relationship support."""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from ..models.data_models import Entity, Relationship, Document
from ..config import get_config

class SimpleKnowledgeGraph:
    """A simple knowledge graph implementation using SQLite with enhanced relationship support."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config()
        self.db_path = db_path or self.config.DATABASE_PATH
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Entities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                text TEXT NOT NULL,
                confidence REAL,
                source_doc TEXT,
                position_start INTEGER,
                position_end INTEGER,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Relationships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT,
                target_entity_id TEXT,
                type TEXT,
                confidence REAL,
                source_doc TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES entities (id),
                FOREIGN KEY (target_entity_id) REFERENCES entities (id)
            )
        ''')
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT,
                file_type TEXT,
                text_content TEXT,
                tables TEXT,
                processed_at TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_doc ON entities(source_doc)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(text)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_doc ON relationships(source_doc)')
        
        conn.commit()
        conn.close()
    
    def add_entity(self, entity: Entity) -> bool:
        """
        Add an entity to the graph.
        
        Args:
            entity: Entity to add
            
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO entities 
                (id, type, text, confidence, source_doc, position_start, position_end, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entity.id, entity.type, entity.text, entity.confidence,
                entity.source_doc, entity.position[0], entity.position[1],
                json.dumps(entity.properties)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error adding entity: {e}")
            return False
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """
        Add a relationship to the graph.
        
        Args:
            relationship: Relationship to add
            
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO relationships 
                (id, source_entity_id, target_entity_id, type, confidence, source_doc, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                relationship.id, relationship.source_entity_id, relationship.target_entity_id,
                relationship.type, relationship.confidence, relationship.source_doc,
                json.dumps(relationship.properties)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error adding relationship: {e}")
            return False
    
    def add_document(self, document: Document) -> bool:
        """
        Add a document to the graph.
        
        Args:
            document: Document to add
            
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, filename, file_type, text_content, tables, processed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                document.id, document.filename, document.file_type,
                document.text_content, json.dumps(document.tables),
                document.processed_at.isoformat(), json.dumps(document.metadata)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def query_entities(self, 
                      entity_type: Optional[str] = None, 
                      source_doc: Optional[str] = None,
                      text_contains: Optional[str] = None,
                      min_confidence: Optional[float] = None,
                      limit: int = 100) -> List[Dict]:
        """
        Query entities from the graph with various filters.
        
        Args:
            entity_type: Filter by entity type
            source_doc: Filter by source document
            text_contains: Filter by text content
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of entity dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM entities WHERE 1=1'
        params = []
        
        if entity_type:
            query += ' AND type = ?'
            params.append(entity_type)
        
        if source_doc:
            query += ' AND source_doc = ?'
            params.append(source_doc)
        
        if text_contains:
            query += ' AND text LIKE ?'
            params.append(f'%{text_contains}%')
        
        if min_confidence is not None:
            query += ' AND confidence >= ?'
            params.append(min_confidence)
        
        query += ' ORDER BY confidence DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            entity_dict = dict(zip(columns, row))
            # Parse JSON properties
            if entity_dict['properties']:
                try:
                    entity_dict['properties'] = json.loads(entity_dict['properties'])
                except:
                    entity_dict['properties'] = {}
            else:
                entity_dict['properties'] = {}
            results.append(entity_dict)
        
        conn.close()
        return results

    def query_relationships(self, 
                          relationship_type: Optional[str] = None,
                          source_entity_id: Optional[str] = None,
                          target_entity_id: Optional[str] = None,
                          source_doc: Optional[str] = None,
                          min_confidence: Optional[float] = None,
                          limit: int = 100) -> List[Dict]:
        """
        Query relationships from the graph with various filters.
        
        Args:
            relationship_type: Filter by relationship type
            source_entity_id: Filter by source entity ID
            target_entity_id: Filter by target entity ID
            source_doc: Filter by source document
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of relationship dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Join with entities to get entity details
        query = '''
            SELECT r.*, 
                   se.text as source_entity_text, se.type as source_entity_type,
                   te.text as target_entity_text, te.type as target_entity_type
            FROM relationships r
            LEFT JOIN entities se ON r.source_entity_id = se.id
            LEFT JOIN entities te ON r.target_entity_id = te.id
            WHERE 1=1
        '''
        params = []
        
        if relationship_type:
            query += ' AND r.type = ?'
            params.append(relationship_type)
        
        if source_entity_id:
            query += ' AND r.source_entity_id = ?'
            params.append(source_entity_id)
        
        if target_entity_id:
            query += ' AND r.target_entity_id = ?'
            params.append(target_entity_id)
        
        if source_doc:
            query += ' AND r.source_doc = ?'
            params.append(source_doc)
        
        if min_confidence is not None:
            query += ' AND r.confidence >= ?'
            params.append(min_confidence)
        
        query += ' ORDER BY r.confidence DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            rel_dict = dict(zip(columns, row))
            # Parse JSON properties
            if rel_dict['properties']:
                try:
                    rel_dict['properties'] = json.loads(rel_dict['properties'])
                except:
                    rel_dict['properties'] = {}
            else:
                rel_dict['properties'] = {}
            results.append(rel_dict)
        
        conn.close()
        return results

    def query_money_entities_by_amount(self, 
                                     order: str = 'DESC',
                                     limit: int = 100,
                                     source_doc: Optional[str] = None) -> List[Dict]:
        """
        Query MONEY entities ordered by their amount value.
        
        Args:
            order: 'ASC' for ascending, 'DESC' for descending
            limit: Maximum number of results
            source_doc: Optional filter by source document
            
        Returns:
            List of money entities ordered by amount
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all money entities first
        query = "SELECT * FROM entities WHERE type = 'MONEY'"
        params = []
        
        if source_doc:
            query += " AND source_doc = ?"
            params.append(source_doc)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            entity_dict = dict(zip(columns, row))
            # Parse JSON properties
            if entity_dict['properties']:
                try:
                    entity_dict['properties'] = json.loads(entity_dict['properties'])
                except:
                    entity_dict['properties'] = {}
            else:
                entity_dict['properties'] = {}
            
            # Only include entities with valid amount values
            if 'amount' in entity_dict['properties']:
                try:
                    amount = float(entity_dict['properties']['amount'])
                    entity_dict['_sort_amount'] = amount  # Add for sorting
                    results.append(entity_dict)
                except (ValueError, TypeError):
                    continue
        
        # Sort by amount
        reverse_order = (order.upper() == 'DESC')
        results.sort(key=lambda x: x['_sort_amount'], reverse=reverse_order)
        
        # Remove the temporary sort key and limit results
        for result in results[:limit]:
            result.pop('_sort_amount', None)
        
        conn.close()
        return results[:limit]

    def get_amount_statistics(self, source_doc: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistical information about monetary amounts.
        
        Args:
            source_doc: Optional filter by source document
            
        Returns:
            Dictionary with amount statistics
        """
        money_entities = self.query_entities(entity_type="MONEY", source_doc=source_doc, limit=10000)
        amounts = []
        
        for entity in money_entities:
            properties = entity.get('properties', {})
            if 'amount' in properties:
                try:
                    amount = float(properties['amount'])
                    amounts.append(amount)
                except (ValueError, TypeError):
                    continue
        
        if not amounts:
            return {
                "total_money_entities": len(money_entities),
                "parseable_amounts": 0,
                "statistics": None
            }
        
        amounts.sort()
        stats = {
            "total_money_entities": len(money_entities),
            "parseable_amounts": len(amounts),
            "min_amount": min(amounts),
            "max_amount": max(amounts),
            "sum_amount": sum(amounts),
            "avg_amount": sum(amounts) / len(amounts),
        }
        
        # Add median
        n = len(amounts)
        if n % 2 == 0:
            stats["median_amount"] = (amounts[n//2 - 1] + amounts[n//2]) / 2
        else:
            stats["median_amount"] = amounts[n//2]
        
        # Add percentiles
        stats["percentile_25"] = amounts[int(0.25 * n)]
        stats["percentile_75"] = amounts[int(0.75 * n)]
        
        return stats
    
    def query_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Dict]:
        """
        Find entities related to a given entity through relationships.
        
        Args:
            entity_id: ID of the entity to find relations for
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            List of related entities with relationship information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get direct relationships with entity details
        cursor.execute('''
            SELECT e.*, r.type as relationship_type, r.confidence as rel_confidence,
                   r.id as relationship_id,
                   CASE 
                       WHEN r.source_entity_id = ? THEN 'outgoing'
                       ELSE 'incoming'
                   END as direction,
                   r.properties as relationship_properties
            FROM entities e
            JOIN relationships r ON (
                (e.id = r.target_entity_id AND r.source_entity_id = ?) OR
                (e.id = r.source_entity_id AND r.target_entity_id = ?)
            )
            WHERE e.id != ?
            ORDER BY r.confidence DESC
        ''', (entity_id, entity_id, entity_id, entity_id))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            entity_dict = dict(zip(columns, row))
            # Parse JSON properties
            if entity_dict['properties']:
                try:
                    entity_dict['properties'] = json.loads(entity_dict['properties'])
                except:
                    entity_dict['properties'] = {}
            else:
                entity_dict['properties'] = {}
                
            # Parse relationship properties
            if entity_dict['relationship_properties']:
                try:
                    entity_dict['relationship_properties'] = json.loads(entity_dict['relationship_properties'])
                except:
                    entity_dict['relationship_properties'] = {}
            else:
                entity_dict['relationship_properties'] = {}
                
            results.append(entity_dict)
        
        conn.close()
        return results

    def get_relationship_statistics(self, source_doc: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about relationships in the graph.
        
        Args:
            source_doc: Optional filter by source document
            
        Returns:
            Dictionary with relationship statistics
        """
        relationships = self.query_relationships(source_doc=source_doc, limit=10000)
        
        stats = {
            "total_relationships": len(relationships),
            "relationships_by_type": {},
            "relationships_by_extraction_method": {},
            "average_confidence": 0.0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0}
        }
        
        if not relationships:
            return stats
        
        # Analyze relationship types
        for rel in relationships:
            rel_type = rel.get('type', 'UNKNOWN')
            stats['relationships_by_type'][rel_type] = stats['relationships_by_type'].get(rel_type, 0) + 1
            
            # Analyze extraction methods
            properties = rel.get('properties', {})
            methods = properties.get('extraction_methods', [properties.get('extraction_method', 'unknown')])
            if not isinstance(methods, list):
                methods = [methods]
            
            for method in methods:
                stats['relationships_by_extraction_method'][method] = stats['relationships_by_extraction_method'].get(method, 0) + 1
            
            # Confidence analysis
            confidence = rel.get('confidence', 0.0)
            if confidence >= 0.7:
                stats['confidence_distribution']['high'] += 1
            elif confidence >= 0.4:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        # Calculate average confidence
        total_confidence = sum(rel.get('confidence', 0.0) for rel in relationships)
        stats['average_confidence'] = total_confidence / len(relationships) if relationships else 0.0
        
        return stats
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get a specific entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity dictionary or None if not found
        """
        entities = self.query_entities(limit=1)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM entities WHERE id = ?', (entity_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            entity_dict = dict(zip(columns, row))
            # Parse JSON properties
            if entity_dict['properties']:
                try:
                    entity_dict['properties'] = json.loads(entity_dict['properties'])
                except:
                    entity_dict['properties'] = {}
            else:
                entity_dict['properties'] = {}
            
            conn.close()
            return entity_dict
        
        conn.close()
        return None
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Get a specific document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            doc_dict = dict(zip(columns, row))
            
            # Parse JSON fields
            for json_field in ['tables', 'metadata']:
                if doc_dict[json_field]:
                    try:
                        doc_dict[json_field] = json.loads(doc_dict[json_field])
                    except:
                        doc_dict[json_field] = {} if json_field == 'metadata' else []
                else:
                    doc_dict[json_field] = {} if json_field == 'metadata' else []
            
            conn.close()
            return doc_dict
        
        conn.close()
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Dictionary with various statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Entity counts
        cursor.execute('SELECT COUNT(*) FROM entities')
        stats['total_entities'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT type, COUNT(*) FROM entities GROUP BY type')
        stats['entities_by_type'] = dict(cursor.fetchall())
        
        # Relationship counts
        cursor.execute('SELECT COUNT(*) FROM relationships')
        stats['total_relationships'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT type, COUNT(*) FROM relationships GROUP BY type')
        stats['relationships_by_type'] = dict(cursor.fetchall())
        
        # Document counts
        cursor.execute('SELECT COUNT(*) FROM documents')
        stats['total_documents'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT file_type, COUNT(*) FROM documents GROUP BY file_type')
        stats['documents_by_type'] = dict(cursor.fetchall())
        
        conn.close()
        
        # Add additional statistics
        stats['amount_statistics'] = self.get_amount_statistics()
        stats['relationship_statistics'] = self.get_relationship_statistics()
        
        return stats
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and all its relationships.
        
        Args:
            entity_id: ID of entity to delete
            
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete relationships first
            cursor.execute('DELETE FROM relationships WHERE source_entity_id = ? OR target_entity_id = ?', 
                          (entity_id, entity_id))
            
            # Delete entity
            cursor.execute('DELETE FROM entities WHERE id = ?', (entity_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error deleting entity: {e}")
            return False
    
    def clear_all_data(self) -> bool:
        """
        Clear all data from the knowledge graph.
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM relationships')
            cursor.execute('DELETE FROM entities')
            cursor.execute('DELETE FROM documents')
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False