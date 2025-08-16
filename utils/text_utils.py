"""Text processing utilities."""

import re
from typing import List, Dict, Any

class TextProcessor:
    """Utility class for text processing operations."""
    
    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in words
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk = ' '.join(chunk_words)
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start += chunk_size - overlap
            
            if start >= len(words):
                break
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)]', ' ', text)
        
        # Normalize currency formatting
        text = re.sub(r'\$\s+', '$', text)
        text = re.sub(r'\s+\%', '%', text)
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (could be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def find_financial_keywords(self, text: str) -> Dict[str, List[str]]:
        """
        Find financial keywords in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of keyword categories and found terms
        """
        keywords = {
            "money_terms": ["payment", "deposit", "withdrawal", "balance", "amount", "total", "cost", "price"],
            "account_terms": ["account", "checking", "savings", "credit", "debit", "statement"],
            "transaction_terms": ["transfer", "transaction", "purchase", "sale", "fee", "charge"],
            "time_terms": ["monthly", "annual", "quarterly", "daily", "period", "date"]
        }
        
        found_keywords = {}
        text_lower = text.lower()
        
        for category, terms in keywords.items():
            found_terms = [term for term in terms if term in text_lower]
            if found_terms:
                found_keywords[category] = found_terms
        
        return found_keywords
    
    def mask_sensitive_data(self, text: str) -> str:
        """
        Mask sensitive information in text.
        
        Args:
            text: Text containing potentially sensitive data
            
        Returns:
            Text with sensitive data masked
        """
        # Mask account numbers (8+ digits)
        text = re.sub(r'\b\d{8,}\b', lambda m: '*' * (len(m.group()) - 4) + m.group()[-4:], text)
        
        # Mask SSNs (XXX-XX-XXXX format)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', text)
        
        # Mask credit card numbers (XXXX XXXX XXXX XXXX format)
        text = re.sub(r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b', 'XXXX XXXX XXXX XXXX', text)
        
        return text
    
    def extract_amounts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract monetary amounts from text.
        
        Args:
            text: Text to extract amounts from
            
        Returns:
            List of amount dictionaries with value and position
        """
        amounts = []
        
        # Pattern for currency amounts
        pattern = r'\$[\d,]+\.?\d*'
        
        for match in re.finditer(pattern, text):
            amount_text = match.group()
            try:
                # Extract numeric value
                numeric_value = float(re.sub(r'[\$,]', '', amount_text))
                amounts.append({
                    'text': amount_text,
                    'value': numeric_value,
                    'start': match.start(),
                    'end': match.end()
                })
            except ValueError:
                continue
        
        return amounts
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Create a simple summary of text by taking key sentences.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Text summary
        """
        if len(text) <= max_length:
            return text
        
        sentences = self.extract_sentences(text)
        if not sentences:
            return text[:max_length] + "..."
        
        # Simple strategy: take first and last sentences, plus any with financial terms
        summary_sentences = []
        
        # Always include first sentence if it exists
        if sentences:
            summary_sentences.append(sentences[0])
        
        # Look for sentences with financial keywords
        financial_keywords = ["$", "payment", "account", "balance", "total", "amount"]
        current_length = len(summary_sentences[0]) if summary_sentences else 0
        
        for sentence in sentences[1:]:
            if any(keyword in sentence.lower() for keyword in financial_keywords):
                if current_length + len(sentence) < max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
        
        # Add last sentence if there's room
        if len(sentences) > 1 and sentences[-1] not in summary_sentences:
            last_sentence = sentences[-1]
            if current_length + len(last_sentence) < max_length:
                summary_sentences.append(last_sentence)
        
        summary = '. '.join(summary_sentences)
        
        # Truncate if still too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary