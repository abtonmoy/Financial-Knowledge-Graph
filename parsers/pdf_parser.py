"""PDF document parser"""
"""PDF document parser."""

from typing import Dict, List, Any
import PyPDF2
from .base_parser import BaseParser

class PDFParser(BaseParser):
    """Parser for PDF documents."""
    
    def supports_file_type(self, file_path: str) -> bool:
        """Check if this is a PDF file."""
        return self.get_file_extension(file_path) == '.pdf'
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and tables from PDF files.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text, tables, and metadata
        """
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid or unreadable file: {file_path}")
        
        text = ""
        tables = []
        metadata = {"pages": 0}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Simple table detection (look for tabular patterns)
                    page_tables = self._extract_simple_tables(page_text, page_num)
                    tables.extend(page_tables)
                
                # Add document metadata if available
                if pdf_reader.metadata:
                    metadata.update({
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "subject": pdf_reader.metadata.get('/Subject', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                        "producer": pdf_reader.metadata.get('/Producer', '')
                    })
        
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF: {e}")
            
        return {
            "text": text.strip(),
            "tables": tables,
            "metadata": metadata
        }
    
    def _extract_simple_tables(self, text: str, page_num: int) -> List[Dict]:
        """
        Extract simple tabular data from text using pattern matching.
        
        Args:
            text: Text content from the page
            page_num: Page number
            
        Returns:
            List of detected tables
        """
        tables = []
        lines = text.split('\n')
        
        # Look for lines with multiple whitespace-separated values
        potential_table_lines = []
        for line_num, line in enumerate(lines):
            # Skip empty lines and lines with only 1-2 words
            if len(line.strip()) == 0:
                continue
                
            # Split by multiple whitespaces to detect columns
            columns = [col.strip() for col in line.split() if col.strip()]
            
            # Consider it a potential table row if it has 3+ columns
            if len(columns) >= 3:
                potential_table_lines.append({
                    "line_number": line_num,
                    "content": line.strip(),
                    "columns": columns
                })
        
        # Group consecutive potential table lines
        if len(potential_table_lines) >= 2:  # At least header + 1 row
            current_table = []
            
            for i, line_data in enumerate(potential_table_lines):
                current_table.append(line_data)
                
                # Check if this should end the current table
                # (gap in line numbers or significant change in column count)
                if (i + 1 < len(potential_table_lines) and 
                    potential_table_lines[i + 1]["line_number"] - line_data["line_number"] > 2):
                    
                    if len(current_table) >= 2:
                        tables.append(self._format_table(current_table, page_num))
                    current_table = []
            
            # Add the final table if it exists
            if len(current_table) >= 2:
                tables.append(self._format_table(current_table, page_num))
        
        return tables
    
    def _format_table(self, table_lines: List[Dict], page_num: int) -> Dict:
        """
        Format detected table lines into a structured table.
        
        Args:
            table_lines: List of detected table lines
            page_num: Page number where table was found
            
        Returns:
            Formatted table dictionary
        """
        return {
            "page": page_num + 1,  # 1-indexed for user display
            "type": "extracted_table",
            "row_count": len(table_lines),
            "estimated_columns": len(table_lines[0]["columns"]) if table_lines else 0,
            "raw_lines": [line["content"] for line in table_lines],
            "structured_data": [line["columns"] for line in table_lines],
            "header": table_lines[0]["columns"] if table_lines else []
        }