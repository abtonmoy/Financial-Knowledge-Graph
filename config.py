import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Application configuration"""

    MODELS = {
        "llm": {
            "name": "microsoft/DialoGPT-medium",
            "alternative": "microsoft/DialoGPT-small"
        },
        "ner": {
            "name": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "alternative": "dslim/bert-base-NER"
        },
        "embeddings": {
            "name": "all-MiniLM-L6-v2",
            "alternative": "all-distilroberta-v1"
        }
    }

    DEVICE = os.getenv("DEVICE", "auto")
    DATABASE_PATH = os.getenv("DB_PATH", "financial_kg.db")

    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./chroma_db")
    COLLECTION_NAME = "financial_docs"

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE",  "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    MAX_GENERATION_LENGTH = int(os.getenv("MAX_GEN_LENGTH", "512"))
    GENERATION_TEMPERATURE = int(os.getenv('GEN_TEMPERATURE', '0.1'))

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = os.getenv("API_PORT", "8000")

    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    FINANCIAL_PATHERNS = {
        "MONEY": [
            r'\$[\d,]+\.?\d*',  # $1,000.00
            r'\d+\.?\d*\s*(?:dollars?|USD|usd)',  # 1000 dollars
            r'(?:USD|usd|\$)\s*[\d,]+\.?\d*'  # USD 1000
        ],
        "PERCENTAGE": [
            r'\d+\.?\d*\s*%',  # 25.5%
            r'\d+\.?\d*\s*percent'  # 25 percent
        ],
        "ACCOUNT_NUMBER": [
            r'\b\d{8,20}\b',  # 8-20 digit sequences
            r'Account\s*#?\s*:?\s*(\d{8,20})',  # Account: 12345678
        ],
        "ROUTING_NUMBER": [
            r'\b\d{9}\b'  # 9-digit routing numbers
        ]
    }

    NER_LEVEL_MAPPING = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "COMPANY",
        "ORGANIZATION": "COMPANY",
        "MISC": "MISC",
        "LOC": "LOCATION"
    }