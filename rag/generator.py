"""RAG answer generation and retrieval system."""

import re
from typing import Dict, List, Any
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
        print("RAG initialized")

    # ---------- Public API ----------

    def add_document_to_vector_store(self, document: Document):
        """
        Add document chunks to the vector store.

        Args:
            document: Document to add to vector store
        """
        # split docs into chunks (fixed: chunk_text + chunk_size spelling)
        chunks = self.text_processor.chunk_text(
            document.text_content,
            chunk_size=self.config.CHUNK_SIZE,
            overlap=self.config.CHUNK_OVERLAP
        )

        if not chunks:
            print(f"No chunks created for document {document.id}")
            return

        embedder = self.model_manager.get_embeddings_model()
        embeddings = embedder.encode(chunks).tolist()

        # add to ChromaDB
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

    def query(self, question: str, top_k: int = 3) -> QueryResult:
        """
        Answer a question using hybrid RAG.

        Args:
            question: User question
            top_k: Number of top documents to retrieve

        Returns:
            QueryResult with answer and sources
        """
        print(f"Answering: {question}")

        try:
            # s1 -> Vector Search for relevant document chunks
            vector_results = self._retrieve_documents(question, top_k)

            # s2 -> Extract entities from the question
            question_entities = self._extract_question_entities(question)

            # s3 -> Search KG for relevant facts
            graph_facts = self._retrieve_graph_facts(question_entities)

            # s4 -> Generate answer using local LLM
            context = self._prepare_context(vector_results, graph_facts)
            answer = self._generate_answer_local(question, context)

            docs_first = self._safe_first(vector_results.get("documents"))
            return QueryResult.create(
                answer=answer,
                documents=docs_first,
                graph_facts=graph_facts,
                metadata={
                    "question_entities": question_entities,
                    "vector_results_count": len(docs_first),
                    "graph_facts_count": len(graph_facts)
                }
            )

        except Exception as e:
            print(f"Error in RAG query: {e}")
            return QueryResult.create(
                answer=f"I encountered an error processing your question: {str(e)}",
                metadata={"error": str(e)}
            )

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with vector store statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.config.COLLECTION_NAME,
                "embedding_model": self.config.MODELS["embeddings"]["name"]
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_vector_store(self) -> bool:
        """
        Clear all data from the vector store.

        Returns:
            True if successful
        """
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.config.COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            return False

    # ---------- Internal helpers ----------

    def _retrieve_documents(self, query: str, top_k: int) -> Dict[str, Any]:
        """
        Retrieve relevant documents using vector search.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with retrieved documents and metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            # Ensure all expected keys exist with sane defaults
            return {
                "documents": results.get("documents", [[]]),
                "metadatas": results.get("metadatas", [[]]),
                "distances": results.get("distances", [[]]),
            }
        except Exception as e:
            print(f"Error in vector search: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _retrieve_graph_facts(self, question_entities: List[str]) -> List[Dict]:
        """
        Retrieve relevant facts from the knowledge graph.

        Args:
            question_entities: Entities extracted from the question

        Returns:
            List of relevant graph facts
        """
        graph_facts: List[Dict[str, Any]] = []

        for entity_text in question_entities:
            # search for entities containing this text
            entities = self.knowledge_graph.query_entities(
                text_contains=entity_text,
                limit=3
            )

            for entity in entities:
                # get related entities
                related = self.knowledge_graph.query_related_entities(entity["id"])
                graph_facts.extend(related[:5])  # limit to avoid context overflow

        # deduplicate and limit total facts
        seen_ids = set()
        unique_facts = []
        for fact in graph_facts:
            fid = fact.get("id")
            if fid and fid not in seen_ids:
                seen_ids.add(fid)
                unique_facts.append(fact)
                if len(unique_facts) >= 10:  # hard limit as using local models
                    break
        return unique_facts

    def _extract_question_entities(self, question: str) -> List[str]:
        """
        Extract potential entities from the question.

        Args:
            question

        Returns:
            List of potential entity strings
        """
        entities: List[str] = []

        # Capture capitalized tokens (robust to hyphens/apostrophes)
        # e.g., "New York", "O'Neill", "Model-3"
        for token in question.split():
            token_stripped = token.strip(",.;:!?()[]{}\"'")
            if len(token_stripped) > 2 and token_stripped[0].isupper() and re.match(r"^[A-Za-z0-9'’-]+$", token_stripped):
                entities.append(token_stripped)

        # Financial patterns
        financial_patterns = [
            r'\$[\d,]+\.?\d*',   # Dollar amounts
            r'\d+\.?\d*%',       # Percentages
            r'\b\d{8,20}\b'      # Account-like numbers
        ]
        for pattern in financial_patterns:
            entities.extend(re.findall(pattern, question))

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                deduped.append(e)
        return deduped

    def _prepare_context(self, vector_results: Dict[str, Any], graph_facts: List[Dict]) -> str:
        """
        Prepare context for the LLM from retrieved information.

        Args:
            vector_results: Results from vector search
            graph_facts: Facts from knowledge graph

        Returns:
            Formatted context string
        """
        context_parts: List[str] = []

        # add document excerpts
        docs_first = self._safe_first(vector_results.get("documents"))
        if docs_first:
            context_parts.append("Document Information:")
            for doc in docs_first[:3]:  # Limiting context
                doc_excerpt = doc[:200] + "..." if len(doc) > 200 else doc
                context_parts.append(f"- {doc_excerpt}")
            context_parts.append("")

        # add graph facts
        if graph_facts:
            context_parts.append("Related Financial Data:")
            for fact in graph_facts[:5]:  # Limit to avoid graph context overflow
                fact_text = f"{fact.get('type', 'Entity')}: {fact.get('text', 'Unknown')}"
                props = fact.get("properties")
                if isinstance(props, dict):
                    key_props = []
                    for key in ["amount", "percentage", "masked_number"]:
                        if key in props:
                            key_props.append(f"{key}: {props[key]}")
                    if key_props:
                        fact_text += f" ({', '.join(key_props)})"
                context_parts.append(f"- {fact_text}")
            context_parts.append("")
        return "\n".join(context_parts)

    def _generate_answer_local(self, question: str, context: str) -> str:
        """
        Generate answer using local LLM.

        Args:
            question: Original question
            context: Prepared context information

        Returns:
            Generated answer
        """
        # Create a focused prompt for financial Q&A
        prompt_template = (
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Based on the provided context, please answer the question about financial information. "
            "Be specific and cite relevant details from the context. If the context doesn't contain "
            "enough information, say so clearly.\n\n"
            "Answer:\n"
        )

        # FIX: use named placeholders
        prompt = prompt_template.format(context=context, question=question)

        # keep prompt reasonable length for local models
        max_total_len = 1000
        if len(prompt) > max_total_len:
            # Truncate context while keeping question and instruction
            # Allow at least 200 chars for question/instructions
            reserved_for_non_context = min(400, len(prompt) - len(context))  # conservative reserve
            max_context_len = max(0, max_total_len - reserved_for_non_context)
            truncated_context = (context[:max_context_len] + "\n[context truncated...]") if len(context) > max_context_len else context
            prompt = prompt_template.format(context=truncated_context, question=question)

        try:
            answer = self.model_manager.generate_text(
                prompt=prompt,
                max_length=self.config.MAX_GENERATION_LENGTH,
                temperature=self.config.GENERATION_TEMPERATURE
            )

            # clean up the answer
            answer = (answer or "").strip()
            if not answer or len(answer) < 10:
                return (
                    "Based on the available information, I cannot provide a specific answer to your question. "
                    "Please try rephrasing or providing more context."
                )

            # remove any repetitive text
            answer = self._clean_generated_text(answer)
            return answer
        except Exception as e:
            return f"I encountered an error generating the answer: {str(e)}"

    def _clean_generated_text(self, text: str) -> str:
        """
        Clean up generated text by removing repetitions and formatting issues.

        Args:
            text: Raw generated text

        Returns:
            Cleaned text
        """
        # Remove excessive repetition (simple unique-sentence pass)
        sentences = [s.strip() for s in text.split(".")]
        unique_sentences = []
        seen_sentences = set()

        for sentence in sentences:
            if sentence and sentence not in seen_sentences:
                seen_sentences.add(sentence)
                unique_sentences.append(sentence)

        cleaned = ". ".join(unique_sentences)
        if cleaned and not cleaned.endswith("."):
            cleaned += "."
        return cleaned

    @staticmethod
    def _safe_first(values: Any) -> List[str]:
        """
        Safely return the first inner list from a Chroma result (or an empty list).
        """
        if not values:
            return []
        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], list):
            return values[0]
        # In case Chroma returns a flat list (rare), still return a list
        if isinstance(values, list):
            return values
        return []
