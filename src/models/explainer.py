"""SHAP-based explainability module."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go

from src.utils.logger import log


class ModelExplainer:
    """Explain model predictions using SHAP."""
    
    def __init__(self, model, X_train: Optional[pd.DataFrame] = None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model
            X_train: Training data for background (used for SHAP)
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
        
        if X_train is not None:
            self.initialize_explainer(X_train)
    
    def initialize_explainer(self, X_background: pd.DataFrame,
                           sample_size: int = 100) -> None:
        """
        Initialize SHAP explainer.
        
        Args:
            X_background: Background data for SHAP
            sample_size: Number of samples to use for background
        """
        try:
            log.info("Initializing SHAP explainer")
            
            # Sample background data if too large
            if len(X_background) > sample_size:
                X_background = X_background.sample(n=sample_size, random_state=42)
            
            # Create TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            
            log.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            log.warning(f"Could not initialize TreeExplainer: {str(e)}")
            # Fallback to KernelExplainer
            try:
                log.info("Falling back to KernelExplainer")
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(X_background, min(50, len(X_background)))
                )
                log.info("KernelExplainer initialized successfully")
            except Exception as e2:
                log.error(f"Could not initialize explainer: {str(e2)}")
                self.explainer = None
    
    def explain_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate SHAP values for predictions.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        log.info(f"Generating SHAP values for {len(X)} samples")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, take values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        self.shap_values = shap_values
        
        return shap_values
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X: Features
            
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_instance(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            X: Features
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary with explanation
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        # Get SHAP values for the instance
        instance_shap = self.shap_values[instance_idx]
        
        # Create explanation dictionary
        explanation = {
            'feature_values': X.iloc[instance_idx].to_dict(),
            'shap_values': dict(zip(X.columns, instance_shap)),
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
        }
        
        return explanation
    
    def get_top_features(self, X: pd.DataFrame, instance_idx: int = 0,
                        top_n: int = 5) -> pd.DataFrame:
        """
        Get top contributing features for a prediction.
        
        Args:
            X: Features
            instance_idx: Index of instance
            top_n: Number of top features to return
            
        Returns:
            DataFrame with top features
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        # Get SHAP values for the instance
        instance_shap = self.shap_values[instance_idx]
        
        # Create DataFrame
        top_features = pd.DataFrame({
            'feature': X.columns,
            'value': X.iloc[instance_idx].values,
            'shap_value': instance_shap,
            'abs_shap_value': np.abs(instance_shap)
        }).sort_values('abs_shap_value', ascending=False).head(top_n)
        
        return top_features
    
    def create_waterfall_plot(self, X: pd.DataFrame, instance_idx: int = 0) -> go.Figure:
        """
        Create waterfall plot for instance explanation.
        
        Args:
            X: Features
            instance_idx: Index of instance
            
        Returns:
            Plotly figure
        """
        top_features = self.get_top_features(X, instance_idx, top_n=10)
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP Values",
            orientation="h",
            measure=["relative"] * len(top_features),
            x=top_features['shap_value'].values,
            y=top_features['feature'].values,
            text=[f"{val:.3f}" for val in top_features['shap_value'].values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=f"SHAP Feature Contribution (Instance {instance_idx})",
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Feature",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_summary_plot_data(self, X: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
        """
        Create data for SHAP summary plot.
        
        Args:
            X: Features
            max_display: Maximum number of features to display
            
        Returns:
            DataFrame with plot data
        """
        if self.shap_values is None:
            self.explain_predictions(X)
        
        # Calculate mean absolute SHAP values
        importance = self.get_feature_importance(X)
        top_features = importance.head(max_display)['feature'].tolist()
        
        # Create summary data
        summary_data = []
        for feature in top_features:
            feature_idx = X.columns.get_loc(feature)
            for i in range(len(X)):
                summary_data.append({
                    'feature': feature,
                    'shap_value': self.shap_values[i, feature_idx],
                    'feature_value': X.iloc[i, feature_idx]
                })
        
        return pd.DataFrame(summary_data)
    
    def get_risk_drivers(self, X: pd.DataFrame, instance_idx: int = 0,
                        threshold: float = 0.1) -> List[Dict]:
        """
        Identify main risk drivers for a prediction.
        
        Args:
            X: Features
            instance_idx: Index of instance
            threshold: Minimum SHAP value threshold
            
        Returns:
            List of risk drivers
        """
        top_features = self.get_top_features(X, instance_idx, top_n=10)
        
        # Filter by threshold and positive contribution
        risk_drivers = top_features[
            (top_features['abs_shap_value'] > threshold) & 
            (top_features['shap_value'] > 0)
        ]
        
        drivers = []
        for _, row in risk_drivers.iterrows():
            drivers.append({
                'feature': row['feature'],
                'value': row['value'],
                'contribution': row['shap_value'],
                'importance': row['abs_shap_value']
            })
        
        return drivers
