"""Entity extraction for financial documents."""

import re
import uuid
from typing import List, Dict, Any
from ..models.data_models import Entity
from ..models.model_manager import OpenSourceModelManager
from ..config import get_config

class FinancialEntityExtractor:
    """Extract financial entities using open-source NER + patterns."""
    
    def __init__(self, model_manager: OpenSourceModelManager):
        self.model_manager = model_manager
        self.config = get_config()
    
    def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        """
        Extract all financial entities from text using NER + patterns.
        
        Args:
            text: Text to extract entities from
            doc_id: Source document ID
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Step 1: Use NER model for general entities (PERSON, ORG, etc.)
        try:
            ner_entities = self._extract_ner_entities(text, doc_id)
            entities.extend(ner_entities)
        except Exception as e:
            print(f"Error in NER extraction: {e}")
        
        # Step 2: Use regex patterns for financial-specific entities
        pattern_entities = self._extract_pattern_entities(text, doc_id)
        entities.extend(pattern_entities)
        
        # Step 3: Deduplicate overlapping entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_ner_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using NER model."""
        entities = []
        
        ner_pipeline = self.model_manager.get_ner_pipeline()
        ner_results = ner_pipeline(text)
        
        for entity_data in ner_results:
            # Map NER labels to our financial types
            entity_type = self._map_ner_label(entity_data["entity_group"])
            
            entity = Entity.create(
                entity_type=entity_type,
                text=entity_data["word"].strip(),
                confidence=entity_data["score"],
                source_doc=doc_id,
                position=(entity_data["start"], entity_data["end"])
            )
            entities.append(entity)
        
        return entities
    
    def _extract_pattern_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.config.FINANCIAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    properties = self._extract_properties(match.group(), entity_type)
                    
                    entity = Entity.create(
                        entity_type=entity_type,
                        text=match.group().strip(),
                        confidence=0.9,  # High confidence for pattern matches
                        source_doc=doc_id,
                        position=(match.start(), match.end()),
                        properties=properties
                    )
                    entities.append(entity)
        
        return entities
    
    def _map_ner_label(self, ner_label: str) -> str:
        """Map NER labels to our financial entity types."""
        return self.config.NER_LABEL_MAPPING.get(ner_label.upper(), "OTHER")
    
    def _extract_properties(self, text: str, entity_type: str) -> Dict[str, Any]:
        """Extract additional properties based on entity type."""
        properties = {}
        
        if entity_type == "MONEY":
            # Extract numerical value
            numbers = re.findall(r'[\d,]+\.?\d*', text)
            if numbers:
                try:
                    amount = float(numbers[0].replace(',', ''))
                    properties['amount'] = amount
                    properties['currency'] = 'USD'  # Default assumption
                    properties['formatted_amount'] = text
                except ValueError:
                    pass
        
        elif entity_type == "PERCENTAGE":
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                try:
                    percentage = float(numbers[0])
                    properties['percentage'] = percentage
                    properties['decimal_value'] = percentage / 100.0
                except ValueError:
                    pass
        
        elif entity_type == "ACCOUNT_NUMBER":
            # Clean account number
            clean_number = re.sub(r'[^\d]', '', text)
            if clean_number:
                properties['clean_number'] = clean_number
                properties['length'] = len(clean_number)
                properties['masked_number'] = self._mask_account_number(clean_number)
        
        elif entity_type == "ROUTING_NUMBER":
            clean_number = re.sub(r'[^\d]', '', text)
            if len(clean_number) == 9:
                properties['clean_number'] = clean_number
                properties['is_valid_length'] = True
        
        return properties
    
    def _mask_account_number(self, account_number: str) -> str:
        """Mask account number for security."""
        if len(account_number) <= 4:
            return '*' * len(account_number)
        return '*' * (len(account_number) - 4) + account_number[-4:]
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping the one with higher confidence."""
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        final_entities = []
        
        for entity in sorted_entities:
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for accepted in final_entities:
                if self._entities_overlap(entity, accepted):
                    overlaps = True
                    break
            
            if not overlaps:
                final_entities.append(entity)
        
        return final_entities
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap in position."""
        return (entity1.position[0] < entity2.position[1] and 
                entity1.position[1] > entity2.position[0])
    
    def extract_from_table(self, table_data: Dict[str, Any], doc_id: str) -> List[Entity]:
        """
        Extract entities from structured table data.
        
        Args:
            table_data: Dictionary containing table information
            doc_id: Source document ID
            
        Returns:
            List of entities extracted from the table
        """
        entities = []
        
        # Extract from headers
        if 'headers' in table_data:
            for i, header in enumerate(table_data['headers']):
                header_entities = self._extract_pattern_entities(str(header), doc_id)
                for entity in header_entities:
                    entity.properties['source'] = 'table_header'
                    entity.properties['column_index'] = i
                entities.extend(header_entities)
        
        # Extract from structured data
        if 'data' in table_data and isinstance(table_data['data'], list):
            for row_idx, row in enumerate(table_data['data']):
                if isinstance(row, dict):
                    for col_name, value in row.items():
                        if value is not None:
                            value_entities = self._extract_pattern_entities(str(value), doc_id)
                            for entity in value_entities:
                                entity.properties['source'] = 'table_cell'
                                entity.properties['row_index'] = row_idx
                                entity.properties['column_name'] = col_name
                            entities.extend(value_entities)
        
        return self._deduplicate_entities(entities)