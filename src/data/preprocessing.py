"""Data preprocessing and feature engineering module."""
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src.utils.logger import log
from src.utils.config import config


class DataPreprocessor:
    """Handle data cleaning and preprocessing."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        initial_rows = len(df_cleaned)
        
        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()
        log.info(f"Removed {initial_rows - len(df_cleaned)} duplicate rows")
        
        # Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # Handle outliers
        df_cleaned = self._handle_outliers(df_cleaned)
        
        log.info(f"Data cleaning complete. Final shape: {df_cleaned.shape}")
        return df_cleaned
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_filled = df.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        categorical_cols = df_filled.select_dtypes(include=['object', 'category']).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            if df_filled[col].isna().any():
                median_value = df_filled[col].median()
                df_filled[col].fillna(median_value, inplace=True)
                log.debug(f"Filled missing values in '{col}' with median: {median_value}")
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            if df_filled[col].isna().any():
                mode_value = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 'Unknown'
                df_filled[col].fillna(mode_value, inplace=True)
                log.debug(f"Filled missing values in '{col}' with mode: {mode_value}")
        
        return df_filled
    
    def _handle_outliers(self, df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """
        Handle outliers using IQR method.
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        if threshold is None:
            threshold = config.data_validation.get('outlier_threshold', 3.0)
        
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate z-scores
            z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
            
            # Cap outliers at threshold
            outliers = z_scores > threshold
            if outliers.any():
                upper_limit = df_cleaned[col].mean() + threshold * df_cleaned[col].std()
                lower_limit = df_cleaned[col].mean() - threshold * df_cleaned[col].std()
                df_cleaned.loc[outliers, col] = df_cleaned.loc[outliers, col].clip(lower_limit, upper_limit)
                log.debug(f"Handled {outliers.sum()} outliers in column '{col}'")
        
        return df_cleaned
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: Optional[List[str]] = None,
                          fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns to encode
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    # Fit and transform
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                    log.debug(f"Encoded column '{col}' with {len(self.encoders[col].classes_)} unique values")
                else:
                    # Transform only
                    if col in self.encoders:
                        # Handle unseen categories
                        df_encoded[col] = df_encoded[col].astype(str)
                        known_values = set(self.encoders[col].classes_)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_values else self.encoders[col].classes_[0]
                        )
                        df_encoded[col] = self.encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            numeric_cols: List of numeric columns to scale
            fit: Whether to fit scalers (True for training, False for prediction)
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        if numeric_cols is None:
            numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col in df_scaled.columns:
                if fit:
                    # Fit and transform
                    self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]])
                    log.debug(f"Scaled column '{col}'")
                else:
                    # Transform only
                    if col in self.scalers:
                        df_scaled[col] = self.scalers[col].transform(df_scaled[[col]])
        
        return df_scaled


class FeatureEngineering:
    """Feature engineering for supply chain data."""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for supply chain prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        df_features = df.copy()
        
        # Time-based features
        if 'order_date' in df_features.columns:
            df_features['order_date'] = pd.to_datetime(df_features['order_date'], errors='coerce')
            df_features['day_of_week'] = df_features['order_date'].dt.dayofweek
            df_features['month'] = df_features['order_date'].dt.month
            df_features['quarter'] = df_features['order_date'].dt.quarter
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Ratio features
        if 'order_value' in df_features.columns and 'shipping_distance' in df_features.columns:
            df_features['value_per_km'] = df_features['order_value'] / (df_features['shipping_distance'] + 1)
        
        if 'lead_time' in df_features.columns and 'shipping_distance' in df_features.columns:
            df_features['speed'] = df_features['shipping_distance'] / (df_features['lead_time'] + 1)
        
        # Risk score combinations
        if 'supplier_reliability_score' in df_features.columns and 'weather_risk_index' in df_features.columns:
            df_features['combined_risk'] = (
                (1 - df_features['supplier_reliability_score']) * df_features['weather_risk_index']
            )
        
        # Inventory pressure
        if 'inventory_level' in df_features.columns and 'demand_forecast' in df_features.columns:
            df_features['inventory_pressure'] = (
                df_features['demand_forecast'] / (df_features['inventory_level'] + 1)
            )
        
        # High value order flag
        if 'order_value' in df_features.columns:
            median_value = df_features['order_value'].median()
            df_features['is_high_value'] = (df_features['order_value'] > median_value).astype(int)
        
        log.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    @staticmethod
    def select_features(df: pd.DataFrame, feature_config: Optional[Dict] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features for modeling.
        
        Args:
            df: Input DataFrame
            feature_config: Feature configuration
            
        Returns:
            Tuple of (DataFrame with selected features, list of feature names)
        """
        if feature_config is None:
            feature_config = config.features
        
        numerical_features = feature_config.get('numerical', [])
        categorical_features = feature_config.get('categorical', [])
        
        # Combine all features
        all_features = numerical_features + categorical_features
        
        # Filter to only available features
        available_features = [f for f in all_features if f in df.columns]
        
        log.info(f"Selected {len(available_features)} features for modeling")
        return df[available_features], available_features
