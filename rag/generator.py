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
            # CHANGE: Add specialized query detection and handling
            query_type = self._detect_query_type(question)
            
            if query_type == "comparative_amount":
                return self._handle_comparative_amount_query(question)
            elif query_type == "specific_entity":
                return self._handle_specific_entity_query(question)
            else:
                return self._handle_general_query(question, top_k)

        except Exception as e:
            print(f"Error in RAG query: {e}")
            return QueryResult.create(
                answer=f"I encountered an error processing your question: {str(e)}",
                metadata={"error": str(e)}
            )

    # CHANGE: Add new method to detect different query types
    def _detect_query_type(self, question: str) -> str:
        """
        Detect the type of query to determine the best approach.
        
        Args:
            question: User question
            
        Returns:
            Query type string
        """
        question_lower = question.lower()
        
        # Comparative queries about amounts
        comparative_keywords = ["highest", "maximum", "largest", "biggest", "most", "greatest", "top"]
        amount_keywords = ["amount", "total", "sum", "value", "cost", "price", "money", "payment"]
        
        if any(comp in question_lower for comp in comparative_keywords) and any(amt in question_lower for amt in amount_keywords):
            return "comparative_amount"
        
        # Specific entity lookups
        entity_keywords = ["account", "company", "person", "name", "number"]
        if any(entity in question_lower for entity in entity_keywords):
            return "specific_entity"
        
        return "general"

    # CHANGE: Add specialized handler for comparative amount queries
    def _handle_comparative_amount_query(self, question: str) -> QueryResult:
        """
        Handle queries asking for highest/maximum amounts by examining all MONEY entities.
        
        Args:
            question: User question
            
        Returns:
            QueryResult with the highest amount found
        """
        try:
            # Get all MONEY entities from knowledge graph
            money_entities = self.knowledge_graph.query_entities(entity_type="MONEY", limit=1000)
            
            if not money_entities:
                return QueryResult.create(
                    answer="I couldn't find any monetary amounts in the processed documents.",
                    metadata={"query_type": "comparative_amount", "entities_found": 0}
                )
            
            # Extract amounts and find the highest
            amounts_with_context = []
            for entity in money_entities:
                try:
                    properties = entity.get('properties', {})
                    if isinstance(properties, str):
                        import json
                        properties = json.loads(properties)
                    
                    amount = properties.get('amount')
                    if amount is not None:
                        amounts_with_context.append({
                            'amount': float(amount),
                            'text': entity.get('text', ''),
                            'formatted': properties.get('formatted_amount', entity.get('text', '')),
                            'entity_id': entity.get('id', '')
                        })
                except (ValueError, TypeError, KeyError):
                    continue
            
            if not amounts_with_context:
                return QueryResult.create(
                    answer="I found monetary amounts but couldn't parse their numerical values for comparison.",
                    metadata={"query_type": "comparative_amount", "unparseable_entities": len(money_entities)}
                )
            
            # Find the highest amount
            highest_amount_info = max(amounts_with_context, key=lambda x: x['amount'])
            
            # Create detailed answer
            answer = f"The highest amount in the statement is {highest_amount_info['formatted']} (${highest_amount_info['amount']:,.2f})."
            
            # Add context about other significant amounts if there are multiple
            if len(amounts_with_context) > 1:
                sorted_amounts = sorted(amounts_with_context, key=lambda x: x['amount'], reverse=True)
                if len(sorted_amounts) >= 3:
                    second_highest = sorted_amounts[1]
                    third_highest = sorted_amounts[2]
                    answer += f" Other significant amounts include {second_highest['formatted']} and {third_highest['formatted']}."
            
            return QueryResult.create(
                answer=answer,
                documents=[f"Found {len(amounts_with_context)} monetary amounts"],
                metadata={
                    "query_type": "comparative_amount",
                    "highest_amount": highest_amount_info['amount'],
                    "total_amounts_found": len(amounts_with_context),
                    "all_amounts": [amt['amount'] for amt in sorted(amounts_with_context, key=lambda x: x['amount'], reverse=True)[:5]]
                }
            )
            
        except Exception as e:
            return QueryResult.create(
                answer=f"Error analyzing amounts: {str(e)}",
                metadata={"error": str(e), "query_type": "comparative_amount"}
            )

    # CHANGE: Add specialized handler for specific entity queries
    def _handle_specific_entity_query(self, question: str) -> QueryResult:
        """
        Handle queries looking for specific entities like account numbers, company names.
        
        Args:
            question: User question
            
        Returns:
            QueryResult with specific entity information
        """
        try:
            # Extract potential entity types from question
            question_lower = question.lower()
            target_types = []
            
            if "account" in question_lower:
                target_types.append("ACCOUNT_NUMBER")
            if "company" in question_lower or "organization" in question_lower:
                target_types.append("COMPANY")
            if "person" in question_lower or "name" in question_lower:
                target_types.append("PERSON")
            
            if not target_types:
                target_types = ["PERSON", "COMPANY", "ACCOUNT_NUMBER"]  # Default fallback
            
            # Get entities of the target types
            all_entities = []
            for entity_type in target_types:
                entities = self.knowledge_graph.query_entities(entity_type=entity_type, limit=100)
                all_entities.extend(entities)
            
            if not all_entities:
                return QueryResult.create(
                    answer="I couldn't find the specific information you're looking for in the processed documents.",
                    metadata={"query_type": "specific_entity", "searched_types": target_types}
                )
            
            # Format the response
            entity_info = []
            for entity in all_entities[:10]:  # Limit to prevent overwhelming response
                entity_text = entity.get('text', '')
                entity_type = entity.get('type', '')
                entity_info.append(f"{entity_type}: {entity_text}")
            
            answer = f"I found the following relevant information: {', '.join(entity_info[:5])}"
            if len(entity_info) > 5:
                answer += f" and {len(entity_info) - 5} more entries."
            
            return QueryResult.create(
                answer=answer,
                documents=[f"Found {len(all_entities)} relevant entities"],
                metadata={
                    "query_type": "specific_entity",
                    "entity_types_found": target_types,
                    "total_entities": len(all_entities)
                }
            )
            
        except Exception as e:
            return QueryResult.create(
                answer=f"Error searching for specific entities: {str(e)}",
                metadata={"error": str(e), "query_type": "specific_entity"}
            )

    # CHANGE: Rename and improve the general query handler
    def _handle_general_query(self, question: str, top_k: int) -> QueryResult:
        """
        Handle general queries using the original RAG approach but with improvements.
        
        Args:
            question: User question
            top_k: Number of top documents to retrieve
            
        Returns:
            QueryResult with answer and sources
        """
        # s1 -> Vector Search for relevant document chunks
        vector_results = self._retrieve_documents(question, top_k)

        # s2 -> Extract entities from the question (improved)
        question_entities = self._extract_question_entities_improved(question)

        # s3 -> Search KG for relevant facts
        graph_facts = self._retrieve_graph_facts(question_entities)

        # s4 -> Generate answer using local LLM (improved context)
        context = self._prepare_context_improved(question, vector_results, graph_facts)
        answer = self._generate_answer_local(question, context)

        docs_first = self._safe_first(vector_results.get("documents"))
        return QueryResult.create(
            answer=answer,
            documents=docs_first,
            graph_facts=graph_facts,
            metadata={
                "query_type": "general",
                "question_entities": question_entities,
                "vector_results_count": len(docs_first),
                "graph_facts_count": len(graph_facts)
            }
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

    # CHANGE: Improved question entity extraction with financial keywords
    def _extract_question_entities_improved(self, question: str) -> List[str]:
        """
        Extract potential entities from the question with improved financial keyword detection.

        Args:
            question: User question

        Returns:
            List of potential entity strings
        """
        entities: List[str] = []

        # CHANGE: Add financial keyword detection first
        financial_keywords = [
            "amount", "total", "sum", "balance", "payment", "cost", "price", "value",
            "money", "cash", "funds", "account", "transaction", "fee", "charge",
            "deposit", "withdrawal", "transfer", "statement", "bill", "invoice"
        ]
        
        question_lower = question.lower()
        for keyword in financial_keywords:
            if keyword in question_lower:
                entities.append(keyword)

        # Capture capitalized tokens (robust to hyphens/apostrophes)
        for token in question.split():
            token_stripped = token.strip(",.;:!?()[]{}\"'")
            if len(token_stripped) > 2 and token_stripped[0].isupper() and re.match(r"^[A-Za-z0-9'â€™-]+$", token_stripped):
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

    # CHANGE: Improved context preparation with question-aware content selection
    def _prepare_context_improved(self, question: str, vector_results: Dict[str, Any], graph_facts: List[Dict]) -> str:
        """
        Prepare context for the LLM with question-aware improvements.

        Args:
            question: Original question for context
            vector_results: Results from vector search
            graph_facts: Facts from knowledge graph

        Returns:
            Formatted context string
        """
        context_parts: List[str] = []

        # CHANGE: Add question context at the beginning
        question_lower = question.lower()
        
        # add document excerpts with smarter selection
        docs_first = self._safe_first(vector_results.get("documents"))
        if docs_first:
            context_parts.append("Document Information:")
            
            # CHANGE: For amount-related questions, prioritize content with dollar signs
            if any(word in question_lower for word in ["amount", "total", "money", "cost", "price", "highest", "maximum"]):
                # Sort documents by relevance to monetary content
                monetary_docs = []
                other_docs = []
                
                for doc in docs_first:
                    if '$' in doc or any(money_word in doc.lower() for money_word in ["amount", "total", "payment", "balance"]):
                        monetary_docs.append(doc)
                    else:
                        other_docs.append(doc)
                
                # Prioritize monetary content
                relevant_docs = (monetary_docs + other_docs)[:3]
            else:
                relevant_docs = docs_first[:3]
            
            for doc in relevant_docs:
                # CHANGE: Increase context length for financial queries
                doc_excerpt = doc[:400] + "..." if len(doc) > 400 else doc
                context_parts.append(f"- {doc_excerpt}")
            context_parts.append("")

        # add graph facts with better formatting
        if graph_facts:
            context_parts.append("Related Financial Data:")
            for fact in graph_facts[:5]:
                fact_text = f"{fact.get('type', 'Entity')}: {fact.get('text', 'Unknown')}"
                props = fact.get("properties")
                if isinstance(props, dict):
                    key_props = []
                    for key in ["amount", "percentage", "masked_number", "formatted_amount"]:
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
        # CHANGE: Create question-type-aware prompts
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["highest", "maximum", "largest", "most", "greatest"]):
            # Comparative prompt
            prompt_template = (
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "You are analyzing financial data. The question asks for a comparison to find the highest/maximum value. "
                "Look through the context for all monetary amounts (marked with $ or mentioned as amounts) and identify "
                "which one is the largest. Provide the specific amount and any relevant details.\n\n"
                "Answer:\n"
            )
        elif any(word in question_lower for word in ["total", "sum", "add"]):
            # Calculation prompt
            prompt_template = (
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "You are analyzing financial data. The question asks for a calculation or total. "
                "Look through the context for relevant amounts and perform the requested calculation. "
                "Show your work if possible.\n\n"
                "Answer:\n"
            )
        else:
            # General prompt
            prompt_template = (
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Based on the provided context, please answer the question about financial information. "
                "Be specific and cite relevant details from the context. If the context doesn't contain "
                "enough information, say so clearly.\n\n"
                "Answer:\n"
            )

        prompt = prompt_template.format(context=context, question=question)

        # keep prompt reasonable length for local models
        max_total_len = 1200  # CHANGE: Increased from 1000 for better context
        if len(prompt) > max_total_len:
            reserved_for_non_context = min(500, len(prompt) - len(context))
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