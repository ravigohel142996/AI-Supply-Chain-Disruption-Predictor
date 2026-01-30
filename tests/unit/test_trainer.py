"""Unit tests for model trainer."""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from src.models.trainer import ModelTrainer


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y)
        return X_df, y_series
    
    def test_create_model_xgboost(self):
        """Test XGBoost model creation."""
        trainer = ModelTrainer(model_type='xgboost')
        model = trainer.create_model()
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_model_random_forest(self):
        """Test Random Forest model creation."""
        trainer = ModelTrainer(model_type='random_forest')
        model = trainer.create_model()
        assert model is not None
        assert hasattr(model, 'fit')
    
    def test_train_with_validation(self, sample_data):
        """Test model training with validation split."""
        X, y = sample_data
        trainer = ModelTrainer(model_type='xgboost')
        metrics = trainer.train(X, y, validation_split=True)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert trainer.model is not None
    
    def test_train_without_validation(self, sample_data):
        """Test model training without validation split."""
        X, y = sample_data
        trainer = ModelTrainer(model_type='random_forest')
        metrics = trainer.train(X, y, validation_split=False)
        
        assert 'accuracy' in metrics
        assert trainer.model is not None
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        trainer = ModelTrainer(model_type='xgboost')
        trainer.train(X, y, validation_split=False)
        
        importance_df = trainer.get_feature_importance()
        assert len(importance_df) == len(X.columns)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
