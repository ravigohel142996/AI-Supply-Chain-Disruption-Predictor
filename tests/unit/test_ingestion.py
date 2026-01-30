"""Unit tests for data ingestion module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.ingestion import DataIngestion


class TestDataIngestion:
    """Tests for DataIngestion class."""
    
    @pytest.fixture
    def ingestion(self):
        """Create DataIngestion instance."""
        return DataIngestion()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'order_id': ['ORD001', 'ORD002', 'ORD003'],
            'order_value': [5000, 7500, 3000],
            'shipping_distance': [500, 800, 300],
            'lead_time': [5, 7, 3],
            'is_delayed': [0, 1, 0]
        })
    
    def test_load_data_csv(self, ingestion, tmp_path):
        """Test loading CSV file."""
        # Create temporary CSV
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        df.to_csv(csv_file, index=False)
        
        # Load data
        loaded_df = ingestion.load_data(str(csv_file))
        
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ['col1', 'col2']
    
    def test_detect_file_type(self, ingestion):
        """Test file type detection."""
        csv_path = Path("test.csv")
        assert ingestion._detect_file_type(csv_path) == 'csv'
        
        xlsx_path = Path("test.xlsx")
        assert ingestion._detect_file_type(xlsx_path) == 'excel'
    
    def test_validate_schema_success(self, ingestion, sample_data):
        """Test successful schema validation."""
        is_valid, errors = ingestion.validate_schema(sample_data)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_schema_missing_columns(self, ingestion):
        """Test schema validation with missing columns."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        is_valid, errors = ingestion.validate_schema(df)
        assert not is_valid
        assert len(errors) > 0
    
    def test_validate_schema_empty_df(self, ingestion):
        """Test schema validation with empty DataFrame."""
        df = pd.DataFrame()
        is_valid, errors = ingestion.validate_schema(df)
        assert not is_valid
        assert 'empty' in str(errors).lower()
    
    def test_get_data_summary(self, ingestion, sample_data):
        """Test data summary generation."""
        summary = ingestion.get_data_summary(sample_data)
        
        assert summary['n_rows'] == 3
        assert summary['n_columns'] == 5
        assert 'columns' in summary
        assert 'missing_values' in summary
        assert 'dtypes' in summary
