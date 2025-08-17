import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Application configuration"""

    MODELS = {
        "llm": {
            "name": 'HuggingFaceH4/zephyr-7b-beta',#"microsoft/DialoGPT-medium",
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

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    MAX_GENERATION_LENGTH = int(os.getenv("MAX_GEN_LENGTH", "512"))
    GENERATION_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.1"))

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    FINANCIAL_PATTERNS = {
        "MONEY": [
            r'\$[\d,]+\.?\d*',
            r'\d+\.?\d*\s*(?:dollars?|USD|usd)',
            r'(?:USD|usd|\$)\s*[\d,]+\.?\d*'
        ],
        "PERCENTAGE": [
            r'\d+\.?\d*\s*%',
            r'\d+\.?\d*\s*percent'
        ],
        "ACCOUNT_NUMBER": [
            r'\b\d{8,20}\b',
            r'Account\s*#?\s*:?\s*(\d{8,20})',
        ],
        "ROUTING_NUMBER": [
            r'\b\d{9}\b'
        ]
    }

    NER_LABEL_MAPPING = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "COMPANY",
        "ORGANIZATION": "COMPANY",
        "MISC": "MISC",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION"
    }

    # Ensure the vector store directory exists
    Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)

def get_config() -> Config:
    return Config
