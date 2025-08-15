
"""Entity extraction for financial documents."""

import re
import uuid
from typing import List, Dict, Any
from ..models.data_models import Entity
from ..models.model_manager import OpenSourceModelManager
from ..config import get_config


class FinancialEntityExtractor:
    """
        Extract all financial entities from text using NER + patterns.
        
        Args:
            text: Text to extract entities from
            doc_id: Source document ID
            
        Returns:
            List of extracted entities
    """
    def __init__(self, model_manager: OpenSourceModelManager):
        self.model_manager = model_manager
        self.config = get_config

    def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        entities = []

        try:
            ner_entities = self._extract_ner_entities(text, doc_id)
            entities.extend(ner_entities)
        except Exception as e:
            print(f"Error in NER extraction: {e}")

        pattern_entities = self._extract_pattern_entities(text, doc_id)
        entities.extend(pattern_entities)

        entities = self._deduplicate_entities(entities)

        return entities
    
    def _extract_ner_entities(self, text: str, doc_id: str):
        entities = []

        ner_pipeline = self.model_manager.get_ner_pipeline()
        ner_result = ner_pipeline(text)

        for entity_data in ner_result:
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
        entities = []

        for entity_type, patterns in self.config.FINANCIAL_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    properties = self._extract_properties(match.group(), entity_type)

                    entity = Entity.create(
                        entity_type=entity_type,
                        text=match.group().strip(),
                        confidence=0.9,
                        source_doc=doc_id,
                        position=(match.start(), match.end()),
                        properties=properties
                    )
                    entities.append(entity)

        return entities
    
    def _map_ner_label(self, ner_label: str) -> str:
        return self.config.NER_LABEL_MAPPING.get(ner_label.upper(), "OTHER")
    
    def _extract_properties(self, text: str, entity_type: str) -> Dict[str, Any]:
        properties = {}

        if entity_type == "Money":
            numbers = re.findall(r'[\d,]+\.?\d*', text)
            if numbers:
                try:
                    amount = float(numbers[0].replace(',', ''))
                    properties["amount"] = amount
                    properties["currency"] = 'USD'
                    properties['formatted_amount'] = text

                except Exception as e:
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
        if len(account_number) <= 4:
            return '*' * len(account_number)
        return '*' * (len(account_number) - 4) + account_number[-4:]

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        sorted_entities = sorted(entities, key = lambda e: e.confidence, reverse=True)
        final_entities = []    
        
        for entity in sorted_entities:
            overlaps = False
            for accepted in final_entities:
                if self._entities_overlap(entity, accepted):
                    overlaps = True
                    break

            if not overlaps:
                final_entities.append(entity)

        return final_entities
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        return (entity1.position[0]< entity2.position[1] and 
                entity1.position[1]>entity2.position[0])