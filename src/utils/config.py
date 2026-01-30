"""Configuration loader for the AI Supply Chain Disruption Predictor."""
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class AppConfig(BaseModel):
    """Application configuration."""
    name: str
    version: str
    environment: str
    debug: bool


class ModelConfig(BaseModel):
    """Model configuration."""
    type: str
    version: str
    hyperparameters: Dict[str, Any]
    cv_folds: int
    test_size: float
    random_state: int


class RiskConfig(BaseModel):
    """Risk threshold configuration."""
    low_threshold: float
    medium_threshold: float
    high_threshold: float


class BusinessConfig(BaseModel):
    """Business impact configuration."""
    avg_order_value: float
    sla_penalty_rate: float
    churn_probability_multiplier: float
    customer_lifetime_value: float


class APIConfig(BaseModel):
    """API configuration."""
    host: str
    port: int
    reload: bool
    workers: int
    cors_origins: list


class Settings(BaseSettings):
    """Environment variables settings."""
    app_env: str = "production"
    debug: bool = False
    api_secret_key: str = "default-secret-key-change-me"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration."""
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_path = self.base_dir / config_path
        self._config = self._load_config()
        self.env_settings = Settings()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def app(self) -> AppConfig:
        """Get application configuration."""
        return AppConfig(**self._config['app'])
    
    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(**self._config['model'])
    
    @property
    def risk(self) -> RiskConfig:
        """Get risk configuration."""
        return RiskConfig(**self._config['risk'])
    
    @property
    def business(self) -> BusinessConfig:
        """Get business configuration."""
        return BusinessConfig(**self._config['business'])
    
    @property
    def api(self) -> APIConfig:
        """Get API configuration."""
        return APIConfig(**self._config['api'])
    
    @property
    def features(self) -> Dict[str, Any]:
        """Get feature configuration."""
        return self._config.get('features', {})
    
    @property
    def alerts(self) -> Dict[str, Any]:
        """Get alert configuration."""
        return self._config.get('alerts', {})
    
    @property
    def data_validation(self) -> Dict[str, Any]:
        """Get data validation configuration."""
        return self._config.get('data_validation', {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    @property
    def paths(self) -> Dict[str, Path]:
        """Get path configuration."""
        paths_config = self._config.get('paths', {})
        return {key: self.base_dir / value for key, value in paths_config.items()}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)


# Global configuration instance
config = Config()
