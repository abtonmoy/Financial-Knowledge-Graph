# Financial Knowledge Graph

An open financial document processing system that extracts entities and relationships from financial documents (PDFs, Excel files) and enables intelligent querying through a hybrid RAG (Retrieval-Augmented Generation) system using Knowledge Graphs.

## Features

- **Multi-format Document Processing**: Support for PDF and Excel files
- **Financial Entity Extraction**: Automated extraction of financial entities (amounts, account numbers, companies, etc.)
- **Relationship Discovery**: Hybrid approach combining rule-based, table-based, LLM-powered, and proximity-based relationship extraction
- **Knowledge Graph Storage**: SQLite-based knowledge graph with comprehensive relationship support
- **RAG-Powered Querying**: Natural language querying with specialized handlers for financial queries
- **REST API**: Complete FastAPI-based REST API for all operations
- **Command Line Interface**: Full CLI for document processing and querying
- **Audit System**: Built-in data quality checks and validation
- **Vector Store**: ChromaDB integration for semantic search

## Requirements

### System Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended for GPU usage)
- CUDA-compatible GPU (optional, but recommended)

### Dependencies

The system uses open-source models and libraries:

- **Language Models**: Transformers-based models (configurable)
- **NER Models**: Token classification models for entity extraction
- **Embeddings**: Sentence transformers for semantic search
- **Vector Store**: ChromaDB for document retrieval
- **Database**: SQLite for knowledge graph storage

## Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd financial-knowledge-graph
```

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Initialize the system**:

```bash
python -m financial_kg.main test
```

## Quick Start

### 1. Process a Document

```bash
# Process a PDF or Excel file
python -m financial_kg.main process /path/to/financial_document.pdf
```

### 2. Query the Knowledge Base

```bash
# Ask questions about your documents
python -m financial_kg.main query "What is the highest amount in the statement?"
python -m financial_kg.main query "Who are the account holders?"
```

### 3. Start the Web API

```bash
# Start the FastAPI server
python -m financial_kg.main server
```

Access the API documentation at `http://localhost:8000/docs`

## API Usage

### Upload and Process Documents

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the total amount of all transactions?"}'
```

### Get Entities

```bash
curl -X GET "http://localhost:8000/api/v1/entities?entity_type=MONEY&limit=10"
```

### Get Relationships

```bash
curl -X GET "http://localhost:8000/api/v1/relationships?limit=10"
```

## Architecture

### Core Components

1. **Processing Engine** (`processing_engine.py`)

   - Orchestrates the entire processing pipeline
   - Manages model loading and inference
   - Coordinates between all system components

2. **Document Parsers** (`parsers/`)

   - **PDF Parser**: Extracts text and tables from PDF files
   - **Excel Parser**: Processes Excel spreadsheets with financial data detection

3. **Entity Extraction** (`extractors/entity_extractor.py`)

   - Combines NER models with regex patterns
   - Extracts financial entities: amounts, account numbers, companies, etc.
   - Provides confidence scoring and property extraction

4. **Relationship Extraction** (`extractors/relationship_extractor.py`)

   - **Rule-based**: Pattern matching for common financial relationships
   - **Table-based**: Infers relationships from table structures
   - **LLM-powered**: Uses language models for complex relationship understanding
   - **Proximity-based**: Finds relationships based on entity proximity in text

5. **Knowledge Graph** (`storage/knowledge_graph.py`)

   - SQLite-based storage with full relationship support
   - Entity and relationship querying with multiple filters
   - Statistical analysis and data validation

6. **RAG System** (`rag/generator.py`)
   - Hybrid retrieval combining vector search and knowledge graph facts
   - Specialized query handlers for financial questions
   - Local LLM integration for answer generation

### Data Models

- **Entity**: Represents financial entities with type, text, confidence, and properties
- **Relationship**: Connects entities with type, confidence, and metadata
- **Document**: Stores processed documents with extracted content
- **QueryResult**: Encapsulates query responses with sources and metadata

## Configuration

The system is configured through `config.py`. Key settings include:

```python
# Model Configuration
MODELS = {
    "llm": {"name": "microsoft/DialoGPT-medium"},
    "ner": {"name": "dbmdz/bert-large-cased-finetuned-conll03-english"},
    "embeddings": {"name": "all-MiniLM-L6-v2"}
}

# Processing Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_GENERATION_LENGTH = 200

# Storage Paths
DATABASE_PATH = "./data/knowledge_graph.db"
VECTOR_STORE_PATH = "./data/vector_store"
```

## Specialized Query Features

The system includes specialized handlers for different types of financial queries:

### Comparative Amount Queries

- **"What is the highest amount?"**
- **"Find the maximum payment"**
- Automatically analyzes all monetary entities and finds extremes

### Specific Entity Queries

- **"What account numbers are mentioned?"**
- **"Who are the companies involved?"**
- Targeted entity retrieval with type-specific filtering

### General Financial Queries

- Uses hybrid RAG approach for complex questions
- Combines document search with knowledge graph facts

## System Statistics and Monitoring

### View Statistics

```bash
python -m financial_kg.main stats
```

### Run Audit Checks

```bash
python -m financial_kg.main audit
```

### Health Check

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

## Advanced Features

### Entity Properties

The system extracts rich properties for different entity types:

- **Money Entities**: Amount, currency, category (small/medium/large), readable format
- **Account Numbers**: Validation, masking, length checks
- **Percentages**: Decimal values, categorization
- **Routing Numbers**: Format validation

### Relationship Types

- **PAYMENT**: Payment relationships between entities
- **OWNERSHIP**: Account ownership relationships
- **TRANSACTION**: Financial transactions
- **EMPLOYMENT**: Employment relationships
- **ASSOCIATION**: General associations

### Data Export

```bash
# Export entities and relationships
curl -X POST "http://localhost:8000/api/v1/export/graph" \
     -H "Content-Type: application/json" \
     -d '{"format": "json"}'
```

## Security Features

- Account number masking for sensitive data
- Input validation and sanitization
- Error handling and graceful degradation
- No external API dependencies for core functionality

## Testing

Run the built-in test with sample data:

```bash
python -m financial_kg.main test
```

## CLI Commands

| Command              | Description                                |
| -------------------- | ------------------------------------------ |
| `server`             | Start the FastAPI web server               |
| `process <file>`     | Process a single document                  |
| `query '<question>'` | Query the knowledge base                   |
| `entities [type]`    | List entities, optionally filtered by type |
| `audit`              | Run data quality audit checks              |
| `stats`              | Display system statistics                  |
| `test`               | Run system test with sample data           |
| `clear`              | Clear all data from the system             |

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch sizes or use CPU-only mode
2. **Model Loading Errors**: Check internet connection for initial model downloads
3. **CUDA Issues**: Ensure CUDA drivers are properly installed

### Performance Optimization

- Use GPU acceleration when available
- Adjust chunk sizes based on document complexity
- Configure model quantization for memory efficiency

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Hugging Face for transformer models
- ChromaDB for vector storage
- FastAPI for the web framework
- The open-source ML community
