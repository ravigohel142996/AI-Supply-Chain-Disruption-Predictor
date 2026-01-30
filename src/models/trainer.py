"""ML model trainer for supply chain disruption prediction."""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

from src.utils.logger import log
from src.utils.config import config


class ModelTrainer:
    """Train and evaluate ML models for disruption prediction."""
    
    def __init__(self, model_type: Optional[str] = None):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model ('xgboost', 'random_forest', 'gradient_boosting')
        """
        self.model_type = model_type or config.model.type
        self.model = None
        self.feature_names = None
        self.metrics = {}
        
    def create_model(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a model instance.
        
        Args:
            hyperparameters: Model hyperparameters
            
        Returns:
            Model instance
        """
        if hyperparameters is None:
            hyperparameters = config.model.hyperparameters
        
        log.info(f"Creating {self.model_type} model")
        
        if self.model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=hyperparameters.get('n_estimators', 200),
                max_depth=hyperparameters.get('max_depth', 10),
                learning_rate=hyperparameters.get('learning_rate', 0.1),
                subsample=hyperparameters.get('subsample', 0.8),
                colsample_bytree=hyperparameters.get('colsample_bytree', 0.8),
                random_state=hyperparameters.get('random_state', 42),
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=hyperparameters.get('n_estimators', 200),
                max_depth=hyperparameters.get('max_depth', 10),
                random_state=hyperparameters.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=hyperparameters.get('n_estimators', 200),
                max_depth=hyperparameters.get('max_depth', 10),
                learning_rate=hyperparameters.get('learning_rate', 0.1),
                subsample=hyperparameters.get('subsample', 0.8),
                random_state=hyperparameters.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: bool = True) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Whether to split data for validation
            
        Returns:
            Training metrics
        """
        log.info(f"Training model with {len(X)} samples and {len(X.columns)} features")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Create model
        self.model = self.create_model()
        
        if validation_split:
            # Split data
            test_size = config.model.test_size
            random_state = config.model.random_state
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            log.info(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            self.metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
            self.metrics['train_size'] = len(X_train)
            self.metrics['val_size'] = len(X_val)
            
            log.info(f"Training complete. Validation Accuracy: {self.metrics['accuracy']:.4f}, "
                    f"AUC: {self.metrics['roc_auc']:.4f}")
        else:
            # Train on full dataset
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            
            self.metrics = self._calculate_metrics(y, y_pred, y_pred_proba)
            self.metrics['train_size'] = len(X)
            
            log.info(f"Training complete. Training Accuracy: {self.metrics['accuracy']:.4f}")
        
        return self.metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv_folds: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation metrics
        """
        if cv_folds is None:
            cv_folds = config.model.cv_folds
        
        log.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Create model
        model = self.create_model()
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
        
        cv_metrics = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'cv_folds': cv_folds
        }
        
        log.info(f"Cross-validation complete. Mean AUC: {cv_metrics['mean_score']:.4f} "
                f"(+/- {cv_metrics['std_score']:.4f})")
        
        return cv_metrics
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series,
                             param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for tuning
            
        Returns:
            Best parameters and scores
        """
        if param_grid is None:
            # Default parameter grid based on model type
            if self.model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'learning_rate': [0.01, 0.1]
                }
            else:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10]
                }
        
        log.info("Starting hyperparameter tuning")
        
        # Create base model
        base_model = self.create_model()
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        log.info(f"Hyperparameter tuning complete. Best score: {tuning_results['best_score']:.4f}")
        log.info(f"Best parameters: {tuning_results['best_params']}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return tuning_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        }
        
        return metrics
    
    def save_model(self, model_path: Optional[str] = None,
                  version: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            model_path: Path to save the model
            version: Model version
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Determine save path
        if model_path is None:
            models_dir = config.paths['models']
            models_dir.mkdir(parents=True, exist_ok=True)
            
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_filename = f"{self.model_type}_v{version}.joblib"
            model_path = models_dir / model_filename
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'version': version,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        log.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to the model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        log.info(f"Loading model from {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data.get('metrics', {})
        
        log.info(f"Model loaded successfully (type: {self.model_type})")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            log.warning("Model does not support feature importance")
            return pd.DataFrame()
