"""SQLite-based knowledge graph storage."""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from ..models.data_models import Entity, Relationship, Document
from ..config import get_config

class SimpleKnowledgeGraph:
    """A simple knowledge graph implementation using SQLite."""
    
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
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id)')
        
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
    
    def query_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Dict]:
        """
        Find entities related to a given entity.
        
        Args:
            entity_id: ID of the entity to find relations for
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            List of related entities with relationship information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Direct relationships (depth 1)
        cursor.execute('''
            SELECT e.*, r.type as relationship_type, r.confidence as rel_confidence,
                   CASE 
                       WHEN r.source_entity_id = ? THEN 'outgoing'
                       ELSE 'incoming'
                   END as direction
            FROM entities e
            JOIN relationships r ON (e.id = r.target_entity_id OR e.id = r.source_entity_id)
            WHERE (r.source_entity_id = ? OR r.target_entity_id = ?)
            AND e.id != ?
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
            results.append(entity_dict)
        
        conn.close()
        return results
    
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
        Get database statistics.
        
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