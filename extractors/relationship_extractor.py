"""Hybrid relationship extraction for financial entities."""

import re
import json
from typing import List, Dict, Any, Tuple, Optional
from ..models.data_models import Entity, Relationship
from ..models.model_manager import OpenSourceModelManager
from ..config import get_config


class HybridRelationshipExtractor:
    """Extract relationships between entities using multiple approaches."""
    
    def __init__(self, model_manager: OpenSourceModelManager):
        self.model_manager = model_manager
        self.config = get_config()
        
        # Financial relationship patterns
        self.relationship_patterns = {
            "PAYMENT": [
                r"(?P<source>\w+(?:\s+\w+)*)\s+(?:paid|pays|transferred)\s+(?P<target>\$[\d,]+\.?\d*)",
                r"(?P<source>\$[\d,]+\.?\d*)\s+(?:from|paid by)\s+(?P<target>\w+(?:\s+\w+)*)",
                r"payment\s+of\s+(?P<source>\$[\d,]+\.?\d*)\s+(?:to|from)\s+(?P<target>\w+(?:\s+\w+)*)"
            ],
            "OWNERSHIP": [
                r"(?P<source>\w+(?:\s+\w+)*)\s+(?:owns|has|holds)\s+(?:account\s+)?(?P<target>\d{8,20})",
                r"account\s+(?P<target>\d{8,20})\s+(?:owned by|belongs to)\s+(?P<source>\w+(?:\s+\w+)*)",
                r"(?P<source>\w+(?:\s+\w+)*)'s\s+(?:account|balance)"
            ],
            "TRANSACTION": [
                r"(?P<source>\w+(?:\s+\w+)*)\s+(?:transferred|sent|deposited)\s+(?P<target>\$[\d,]+\.?\d*)",
                r"(?P<source>\$[\d,]+\.?\d*)\s+(?:transferred|sent)\s+(?:to|from)\s+(?P<target>\w+(?:\s+\w+)*)",
                r"transaction\s+(?:of\s+)?(?P<source>\$[\d,]+\.?\d*)\s+(?:between|with)\s+(?P<target>\w+(?:\s+\w+)*)"
            ],
            "EMPLOYMENT": [
                r"(?P<source>\w+(?:\s+\w+)*)\s+(?:works for|employed by|employee of)\s+(?P<target>\w+(?:\s+\w+)*)",
                r"(?P<target>\w+(?:\s+\w+)*)\s+(?:employs|employee)\s+(?P<source>\w+(?:\s+\w+)*)"
            ],
            "ASSOCIATION": [
                r"(?P<source>\w+(?:\s+\w+)*)\s+(?:is associated with|related to|connected to)\s+(?P<target>\w+(?:\s+\w+)*)",
                r"(?P<source>\w+(?:\s+\w+)*)\s+(?:and|&)\s+(?P<target>\w+(?:\s+\w+)*)\s+(?:partnership|joint|together)"
            ]
        }
    
    def extract_relationships(self, entities: List[Entity], text: str, doc_id: str, 
                            tables: Optional[List[Dict[str, Any]]] = None) -> List[Relationship]:
        """
        Extract relationships using hybrid approach.
        
        Args:
            entities: List of entities to find relationships between
            text: Source text content
            doc_id: Document ID
            tables: Optional table data for table-based extraction
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Method 1: Rule-based pattern matching
        rule_relationships = self._extract_rule_based_relationships(entities, text, doc_id)
        relationships.extend(rule_relationships)
        
        # Method 2: Table-based relationships
        if tables:
            table_relationships = self._extract_table_relationships(entities, tables, doc_id)
            relationships.extend(table_relationships)
        
        # Method 3: LLM-powered relationship extraction
        llm_relationships = self._extract_llm_relationships(entities, text, doc_id)
        relationships.extend(llm_relationships)
        
        # Method 4: Proximity-based relationships
        proximity_relationships = self._extract_proximity_relationships(entities, text, doc_id)
        relationships.extend(proximity_relationships)
        
        # Deduplicate relationships
        relationships = self._deduplicate_relationships(relationships)
        
        print(f"Extracted {len(relationships)} relationships using hybrid approach")
        return relationships
    
    def _extract_rule_based_relationships(self, entities: List[Entity], text: str, doc_id: str) -> List[Relationship]:
        """Extract relationships using predefined patterns."""
        relationships = []
        
        # Create entity lookup maps
        entity_by_text = {entity.text.lower(): entity for entity in entities}
        entity_by_id = {entity.id: entity for entity in entities}
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    source_text = match.group('source').strip().lower()
                    target_text = match.group('target').strip().lower()
                    
                    # Find matching entities
                    source_entity = self._find_best_entity_match(source_text, entities)
                    target_entity = self._find_best_entity_match(target_text, entities)
                    
                    if source_entity and target_entity and source_entity.id != target_entity.id:
                        relationship = Relationship.create(
                            source_entity_id=source_entity.id,
                            target_entity_id=target_entity.id,
                            relationship_type=rel_type,
                            confidence=0.8,  # High confidence for pattern matches
                            source_doc=doc_id,
                            properties={
                                "extraction_method": "rule_based",
                                "pattern_match": match.group(0),
                                "match_start": match.start(),
                                "match_end": match.end()
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_table_relationships(self, entities: List[Entity], tables: List[Dict[str, Any]], doc_id: str) -> List[Relationship]:
        """Extract relationships from structured table data."""
        relationships = []
        
        for table in tables:
            if 'data' not in table or not isinstance(table['data'], list):
                continue
            
            headers = table.get('headers', [])
            if len(headers) < 2:
                continue
            
            # Look for common financial table patterns
            for row_idx, row in enumerate(table['data']):
                if not isinstance(row, dict):
                    continue
                
                # Find entities in this row
                row_entities = []
                for col_name, value in row.items():
                    if value is None:
                        continue
                    
                    # Find entities that match this cell value
                    matching_entities = [e for e in entities if e.text.lower() in str(value).lower() or str(value).lower() in e.text.lower()]
                    for entity in matching_entities:
                        row_entities.append((entity, col_name))
                
                # Create relationships between entities in the same row
                for i, (entity1, col1) in enumerate(row_entities):
                    for j, (entity2, col2) in enumerate(row_entities):
                        if i >= j or entity1.id == entity2.id:
                            continue
                        
                        # Determine relationship type based on column names
                        rel_type = self._infer_table_relationship_type(col1, col2)
                        
                        relationship = Relationship.create(
                            source_entity_id=entity1.id,
                            target_entity_id=entity2.id,
                            relationship_type=rel_type,
                            confidence=0.7,  # Medium confidence for table relationships
                            source_doc=doc_id,
                            properties={
                                "extraction_method": "table_based",
                                "source_column": col1,
                                "target_column": col2,
                                "row_index": row_idx,
                                "table_sheet": table.get('sheet_name', 'unknown')
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_llm_relationships(self, entities: List[Entity], text: str, doc_id: str) -> List[Relationship]:
        """Extract relationships using LLM for complex narrative understanding."""
        relationships = []
        
        # Limit entities to avoid overwhelming the model
        important_entities = [e for e in entities if e.confidence > 0.6][:10]
        
        if len(important_entities) < 2:
            return relationships
        
        # Create entity context
        entity_context = []
        for entity in important_entities:
            entity_context.append(f"- {entity.type}: {entity.text} (ID: {entity.id})")
        
        # Prepare prompt for relationship extraction
        prompt = f"""
Text: {text[:800]}

Entities found:
{chr(10).join(entity_context)}

Based on the text and entities above, identify relationships between the entities. 
Focus on financial relationships like payments, ownership, transactions, employment, etc.

Format your response as JSON with this structure:
[
  {{
    "source_id": "entity_id",
    "target_id": "entity_id", 
    "relationship": "PAYMENT|OWNERSHIP|TRANSACTION|EMPLOYMENT|ASSOCIATION",
    "confidence": 0.0-1.0,
    "evidence": "text snippet that supports this relationship"
  }}
]

Only include relationships you are confident about. Respond with valid JSON only.
"""
        
        try:
            response = self.model_manager.generate_text(
                prompt=prompt,
                max_length=400,
                temperature=0.1  # Low temperature for more deterministic output
            )
            
            # Try to parse JSON response
            relationships_data = self._parse_llm_response(response)
            
            for rel_data in relationships_data:
                if self._validate_llm_relationship(rel_data, important_entities):
                    relationship = Relationship.create(
                        source_entity_id=rel_data['source_id'],
                        target_entity_id=rel_data['target_id'],
                        relationship_type=rel_data['relationship'],
                        confidence=min(float(rel_data['confidence']), 0.6),  # Cap LLM confidence
                        source_doc=doc_id,
                        properties={
                            "extraction_method": "llm_based",
                            "evidence": rel_data.get('evidence', ''),
                            "model_response": response[:200]
                        }
                    )
                    relationships.append(relationship)
        
        except Exception as e:
            print(f"Error in LLM relationship extraction: {e}")
        
        return relationships
    
    def _extract_proximity_relationships(self, entities: List[Entity], text: str, doc_id: str) -> List[Relationship]:
        """Extract relationships based on entity proximity in text."""
        relationships = []
        
        # Sort entities by position
        positioned_entities = [e for e in entities if e.position[0] >= 0 and e.position[1] > e.position[0]]
        positioned_entities.sort(key=lambda e: e.position[0])
        
        # Create relationships between nearby entities
        for i, entity1 in enumerate(positioned_entities):
            for j, entity2 in enumerate(positioned_entities[i+1:], i+1):
                # Check if entities are close enough
                distance = entity2.position[0] - entity1.position[1]
                
                if distance > 200:  # Skip if too far apart
                    break
                
                # Extract text between entities for context
                between_text = text[entity1.position[1]:entity2.position[0]].strip()
                
                # Infer relationship type from context
                rel_type, confidence = self._infer_proximity_relationship(entity1, entity2, between_text)
                
                if rel_type and confidence > 0.3:
                    relationship = Relationship.create(
                        source_entity_id=entity1.id,
                        target_entity_id=entity2.id,
                        relationship_type=rel_type,
                        confidence=confidence,
                        source_doc=doc_id,
                        properties={
                            "extraction_method": "proximity_based",
                            "distance": distance,
                            "between_text": between_text,
                            "source_position": entity1.position,
                            "target_position": entity2.position
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _find_best_entity_match(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find the best matching entity for a text string."""
        text = text.strip().lower()
        
        # Exact match first
        for entity in entities:
            if entity.text.lower() == text:
                return entity
        
        # Partial match
        for entity in entities:
            if text in entity.text.lower() or entity.text.lower() in text:
                return entity
        
        # Pattern match for special cases (amounts, accounts)
        for entity in entities:
            if entity.type == "MONEY" and re.search(r'\$?[\d,]+\.?\d*', text):
                return entity
            if entity.type == "ACCOUNT_NUMBER" and re.search(r'\d{8,20}', text):
                return entity
        
        return None
    
    def _infer_table_relationship_type(self, col1: str, col2: str) -> str:
        """Infer relationship type from table column names."""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Payment relationships
        if any(word in col1_lower for word in ['amount', 'payment', 'cost']) and any(word in col2_lower for word in ['name', 'person', 'company']):
            return "PAYMENT"
        if any(word in col2_lower for word in ['amount', 'payment', 'cost']) and any(word in col1_lower for word in ['name', 'person', 'company']):
            return "PAYMENT"
        
        # Ownership relationships
        if any(word in col1_lower for word in ['account', 'number']) and any(word in col2_lower for word in ['owner', 'name', 'customer']):
            return "OWNERSHIP"
        if any(word in col2_lower for word in ['account', 'number']) and any(word in col1_lower for word in ['owner', 'name', 'customer']):
            return "OWNERSHIP"
        
        # Default association
        return "ASSOCIATION"
    
    def _infer_proximity_relationship(self, entity1: Entity, entity2: Entity, between_text: str) -> Tuple[Optional[str], float]:
        """Infer relationship type and confidence from proximity context."""
        between_lower = between_text.lower()
        
        # Payment indicators
        payment_indicators = ['paid', 'payment', 'to', 'from', 'transferred', 'sent']
        if any(indicator in between_lower for indicator in payment_indicators):
            if entity1.type == "PERSON" and entity2.type == "MONEY":
                return "PAYMENT", 0.6
            if entity1.type == "MONEY" and entity2.type == "PERSON":
                return "PAYMENT", 0.6
        
        # Ownership indicators
        ownership_indicators = ['owns', 'has', 'account', 'balance', "'s"]
        if any(indicator in between_lower for indicator in ownership_indicators):
            if entity1.type == "PERSON" and entity2.type == "ACCOUNT_NUMBER":
                return "OWNERSHIP", 0.7
            if entity1.type == "COMPANY" and entity2.type == "ACCOUNT_NUMBER":
                return "OWNERSHIP", 0.7
        
        # General association for entities close together
        if len(between_text.strip()) < 50:  # Very close entities
            return "ASSOCIATION", 0.4
        
        return None, 0.0
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract relationship data."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Try to parse the entire response as JSON
            return json.loads(response)
        
        except json.JSONDecodeError:
            # Fallback: extract relationships using regex
            relationships = []
            lines = response.split('\n')
            
            for line in lines:
                # Look for relationship patterns in text
                if 'relationship' in line.lower() and 'confidence' in line.lower():
                    # This is a simple fallback - in practice you might want more sophisticated parsing
                    continue
            
            return relationships
    
    def _validate_llm_relationship(self, rel_data: Dict[str, Any], entities: List[Entity]) -> bool:
        """Validate LLM-extracted relationship data."""
        required_fields = ['source_id', 'target_id', 'relationship', 'confidence']
        
        # Check required fields
        for field in required_fields:
            if field not in rel_data:
                return False
        
        # Check that entity IDs exist
        entity_ids = {e.id for e in entities}
        if rel_data['source_id'] not in entity_ids or rel_data['target_id'] not in entity_ids:
            return False
        
        # Check relationship type is valid
        valid_types = ['PAYMENT', 'OWNERSHIP', 'TRANSACTION', 'EMPLOYMENT', 'ASSOCIATION']
        if rel_data['relationship'] not in valid_types:
            return False
        
        # Check confidence is valid
        try:
            confidence = float(rel_data['confidence'])
            if not (0.0 <= confidence <= 1.0):
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships, keeping the highest confidence ones."""
        # Group by source-target-type tuple
        relationship_groups = {}
        
        for rel in relationships:
            key = (rel.source_entity_id, rel.target_entity_id, rel.type)
            if key not in relationship_groups:
                relationship_groups[key] = []
            relationship_groups[key].append(rel)
        
        # Keep the highest confidence relationship from each group
        final_relationships = []
        for group in relationship_groups.values():
            best_rel = max(group, key=lambda r: r.confidence)
            
            # Merge properties from all relationships in the group
            merged_properties = {}
            extraction_methods = []
            
            for rel in group:
                merged_properties.update(rel.properties)
                method = rel.properties.get('extraction_method', 'unknown')
                if method not in extraction_methods:
                    extraction_methods.append(method)
            
            best_rel.properties['extraction_methods'] = extraction_methods
            best_rel.properties['duplicate_count'] = len(group)
            
            final_relationships.append(best_rel)
        
        return final_relationships