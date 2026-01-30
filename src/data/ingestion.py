"""Data ingestion module for loading and validating data."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.utils.logger import log
from src.utils.config import config


class DataIngestion:
    """Handle data loading and validation."""
    
    def __init__(self):
        """Initialize data ingestion."""
        self.validation_config = config.data_validation
        
    def load_data(self, file_path: str, file_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.
        
        Args:
            file_path: Path to the data file
            file_type: File type ('csv' or 'excel'). Auto-detected if None.
            
        Returns:
            Loaded DataFrame
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Auto-detect file type if not provided
            if file_type is None:
                file_type = self._detect_file_type(path)
            
            log.info(f"Loading data from {file_path} (type: {file_type})")
            
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type in ['excel', 'xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            log.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            log.error(f"Error loading data: {str(e)}")
            raise
    
    def _detect_file_type(self, path: Path) -> str:
        """Detect file type from extension."""
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return 'csv'
        elif suffix in ['.xlsx', '.xls']:
            return 'excel'
        else:
            raise ValueError(f"Cannot detect file type from extension: {suffix}")
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema against required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        required_columns = self.validation_config.get('required_columns', [])
        
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
        
        # Check missing value rate
        max_missing_rate = self.validation_config.get('max_missing_rate', 0.3)
        for col in df.columns:
            missing_rate = df[col].isna().sum() / len(df)
            if missing_rate > max_missing_rate:
                errors.append(
                    f"Column '{col}' has {missing_rate:.2%} missing values "
                    f"(threshold: {max_missing_rate:.2%})"
                )
        
        is_valid = len(errors) == 0
        
        if is_valid:
            log.info("Schema validation passed")
        else:
            log.warning(f"Schema validation failed: {'; '.join(errors)}")
        
        return is_valid, errors
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of the dataset.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_values': df.isna().sum().to_dict(),
            'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        return summary
    
    def upload_handler(self, uploaded_file) -> pd.DataFrame:
        """
        Handle file upload from Streamlit.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Loaded DataFrame
        """
        try:
            file_name = uploaded_file.name
            file_extension = Path(file_name).suffix.lower()
            
            log.info(f"Processing uploaded file: {file_name}")
            
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            log.info(f"Successfully loaded {len(df)} rows from uploaded file")
            return df
            
        except Exception as e:
            log.error(f"Error processing uploaded file: {str(e)}")
            raise
