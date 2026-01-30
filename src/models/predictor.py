"""Predictor module for making predictions and risk scoring."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import log
from src.utils.config import config


class RiskPredictor:
    """Make predictions and calculate risk scores."""
    
    def __init__(self, model):
        """
        Initialize predictor.
        
        Args:
            model: Trained model instance
        """
        self.model = model
        self.risk_config = config.risk
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions
        """
        if self.model is None:
            raise ValueError("No model available for prediction")
        
        predictions = self.model.predict(X)
        log.info(f"Made predictions for {len(X)} samples")
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of delay.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("No model available for prediction")
        
        probabilities = self.model.predict_proba(X)[:, 1]
        log.info(f"Predicted probabilities for {len(X)} samples")
        
        return probabilities
    
    def categorize_risk(self, probabilities: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Categorize risk levels based on probabilities.
        
        Args:
            probabilities: Delay probabilities
            
        Returns:
            Tuple of (risk levels as integers, risk labels)
        """
        low_threshold = self.risk_config.low_threshold
        medium_threshold = self.risk_config.medium_threshold
        high_threshold = self.risk_config.high_threshold
        
        # Initialize risk levels
        risk_levels = np.zeros(len(probabilities), dtype=int)
        risk_labels = []
        
        for i, prob in enumerate(probabilities):
            if prob < low_threshold:
                risk_levels[i] = 0
                risk_labels.append('Low')
            elif prob < medium_threshold:
                risk_levels[i] = 1
                risk_labels.append('Medium')
            elif prob < high_threshold:
                risk_levels[i] = 2
                risk_labels.append('High')
            else:
                risk_levels[i] = 3
                risk_labels.append('Critical')
        
        return risk_levels, risk_labels
    
    def predict_with_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with risk categorization.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with predictions, probabilities, and risk categories
        """
        # Get predictions and probabilities
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Categorize risk
        risk_levels, risk_labels = self.categorize_risk(probabilities)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'delay_probability': probabilities,
            'risk_level': risk_levels,
            'risk_category': risk_labels
        })
        
        log.info(f"Risk distribution: {pd.Series(risk_labels).value_counts().to_dict()}")
        
        return results
    
    def batch_predict(self, X: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        """
        Make predictions in batches for large datasets.
        
        Args:
            X: Feature matrix
            batch_size: Size of each batch
            
        Returns:
            DataFrame with all predictions
        """
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        log.info(f"Processing {n_samples} samples in {n_batches} batches")
        
        all_results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            X_batch = X.iloc[start_idx:end_idx]
            batch_results = self.predict_with_risk(X_batch)
            all_results.append(batch_results)
        
        results = pd.concat(all_results, ignore_index=True)
        log.info("Batch prediction complete")
        
        return results
    
    def calibrate_predictions(self, probabilities: np.ndarray,
                             calibration_method: str = 'platt') -> np.ndarray:
        """
        Calibrate prediction probabilities.
        
        Args:
            probabilities: Raw probabilities
            calibration_method: Calibration method ('platt' or 'isotonic')
            
        Returns:
            Calibrated probabilities
        """
        # Placeholder for calibration - would need calibration set
        # For now, return original probabilities
        log.info(f"Calibration method '{calibration_method}' requested but not implemented")
        return probabilities
    
    def get_confidence_intervals(self, X: pd.DataFrame,
                                 confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Calculate confidence intervals for predictions (if supported by model).
        
        Args:
            X: Feature matrix
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            DataFrame with confidence intervals
        """
        # Placeholder - would need bootstrap or model-specific implementation
        probabilities = self.predict_proba(X)
        
        # Simple heuristic based on prediction uncertainty
        uncertainty = probabilities * (1 - probabilities)
        margin = uncertainty * 1.96  # For 95% CI
        
        ci_df = pd.DataFrame({
            'prediction': probabilities,
            'lower_bound': np.clip(probabilities - margin, 0, 1),
            'upper_bound': np.clip(probabilities + margin, 0, 1),
            'uncertainty': uncertainty
        })
        
        return ci_df
