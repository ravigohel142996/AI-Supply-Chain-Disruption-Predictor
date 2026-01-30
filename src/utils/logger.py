"""Logging configuration for the AI Supply Chain Disruption Predictor."""
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.utils.config import config


class Logger:
    """Centralized logger for the application."""
    
    def __init__(self):
        """Initialize logger."""
        self._configured = False
        
    def setup(self, log_file: Optional[str] = None) -> None:
        """Setup logger configuration."""
        if self._configured:
            return
            
        # Remove default logger
        logger.remove()
        
        # Get logging configuration
        log_config = config.logging
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', 
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>")
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=True
        )
        
        # Add file handler
        if log_file is None:
            logs_dir = config.paths.get('logs', Path('logs'))
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / "app.log"
        
        logger.add(
            str(log_file),
            format=log_format,
            level=log_level,
            rotation=log_config.get('rotation', '500 MB'),
            retention=log_config.get('retention', '10 days'),
            compression="zip"
        )
        
        self._configured = True
        logger.info("Logger initialized successfully")
    
    def get_logger(self):
        """Get logger instance."""
        if not self._configured:
            self.setup()
        return logger


# Global logger instance
app_logger = Logger()
log = app_logger.get_logger()
