"""RAG answer generation and retrieval system."""

import re
from typing import Dict, List, Any, Optional
import chromadb
from ..models.data_models import Document, QueryResult
from ..models.model_manager import OpenSourceModelManager
from ..storage.knowledge_graph import SimpleKnowledgeGraph
from ..config import get_config
from ..utils.text_utils import TextProcessor

class OpenSourceRAG:

    def __init__(self, model_manager: OpenSourceModelManager, knowledge_graph: SimpleKnowledgeGraph):
        self.model_manager = model_manager
        self.knowledge_graph = knowledge_graph
        self.config = get_config()
        self.text_processor = TextProcessor()

        # init chromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.config.VECTOR_STORE_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("RAG intialized")

    def add_document_to_vector_store(self, document: Document):
        """
        Add document chunks to the vector store.
        
        Args:
            document: Document to add to vector store
        """
        # split docs into chunks
        chunks = self.text_processor.chunk_test(
            document.text_content,
            chuck_size = self.config.CHUNK_SIZE,
            overlap=self.config.CHUNK_OVERLAP
        )

        if not chunks:
            print(f"No chunks created for document {document.id}")
            return
        
        embedder = self.model_manager.get_embeddings_model()
        embeddings = embedder.encode(chunks).tolist()

        # add to Chromadb
        chunk_ids = [f"{document.id}_chunk_{i}" for i in range(len(chunks))]

        try:
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunk_index": i,
                    "file_type": document.file_type
                } for i in range(len(chunks))],
                ids=chunk_ids
            )
            print(f"Added {len(chunks)} chunks to vector store for {document.filename}")
        except Exception as e:
            print(f"Error adding document to vector store: {e}")
            