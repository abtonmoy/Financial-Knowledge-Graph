"""Excel document parser."""

from typing import Dict, List, Any
import pandas as pd
from .base_parser import BaseParser

class ExcelParser(BaseParser):
    """Parser for Excel documents (.xlsx, .xls)."""
    
    def supports_file_type(self, file_path: str) -> bool:
        """Check if this is an Excel file."""
        extension = self.get_file_extension(file_path)
        return extension in ['.xlsx', '.xls']
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Extract data from Excel files.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with extracted text, tables, and metadata
        """
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid or unreadable file: {file_path}")
        
        text = ""
        tables = []
        metadata = {"sheets": 0}
        
        try:
            excel_file = pd.ExcelFile(file_path)
            metadata["sheets"] = len(excel_file.sheet_names)
            metadata["sheet_names"] = excel_file.sheet_names
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Convert to text representation
                    sheet_text = f"\n=== Sheet: {sheet_name} ===\n"
                    if not df.empty:
                        sheet_text += df.to_string(index=False)
                    else:
                        sheet_text += "[Empty Sheet]"
                    text += sheet_text
                    
                    # Store as structured table
                    table_data = {
                        "sheet_name": sheet_name,
                        "type": "excel_sheet",
                        "data": df.to_dict('records') if not df.empty else [],
                        "headers": list(df.columns) if not df.empty else [],
                        "shape": df.shape,
                        "row_count": len(df),
                        "column_count": len(df.columns) if not df.empty else 0,
                        "has_data": not df.empty
                    }
                    
                    # Add summary statistics for numeric columns
                    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_columns:
                        table_data["numeric_summary"] = df[numeric_columns].describe().to_dict()
                    
                    tables.append(table_data)
                    
                except Exception as sheet_error:
                    print(f"Warning: Could not process sheet '{sheet_name}': {sheet_error}")
                    continue
        
        except Exception as e:
            raise RuntimeError(f"Error parsing Excel file: {e}")
            
        return {
            "text": text.strip(),
            "tables": tables,
            "metadata": metadata
        }
    
    def _detect_financial_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect common financial patterns in Excel data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            "has_currency_columns": False,
            "has_date_columns": False,
            "has_account_numbers": False,
            "potential_balance_sheet": False,
            "potential_income_statement": False
        }
        
        # Check column names for financial indicators
        column_names = [str(col).lower() for col in df.columns]
        
        # Currency indicators
        currency_keywords = ['amount', 'balance', 'total', 'cost', 'price', 'value', 'payment']
        patterns["has_currency_columns"] = any(keyword in ' '.join(column_names) for keyword in currency_keywords)
        
        # Date indicators
        date_keywords = ['date', 'time', 'period', 'month', 'year']
        patterns["has_date_columns"] = any(keyword in ' '.join(column_names) for keyword in date_keywords)
        
        # Account indicators
        account_keywords = ['account', 'id', 'number', 'reference']
        patterns["has_account_numbers"] = any(keyword in ' '.join(column_names) for keyword in account_keywords)
        
        # Financial statement indicators
        balance_keywords = ['assets', 'liabilities', 'equity', 'balance']
        income_keywords = ['revenue', 'income', 'expense', 'profit', 'loss']
        
        patterns["potential_balance_sheet"] = any(keyword in ' '.join(column_names) for keyword in balance_keywords)
        patterns["potential_income_statement"] = any(keyword in ' '.join(column_names) for keyword in income_keywords)
        
        return patterns