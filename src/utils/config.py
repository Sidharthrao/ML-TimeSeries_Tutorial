"""
Configuration parameters for the time series forecasting project
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for project parameters"""
    
    # Data paths
    DATA_PATH: str = "Data/household_power_consumption.txt"
    MODELS_DIR: str = "models"
    OUTPUTS_DIR: str = "outputs"
    
    # Data preprocessing
    MISSING_VALUE_INDICATOR: str = "?"
    RESAMPLE_FREQ: str = "H"  # Hourly
    TRAIN_SPLIT: float = 0.85  # 85% for training
    
    # Feature engineering
    LAG_FEATURES: list = None  # Will be set to [1, 2, 3, 6, 12, 24] + [24*7, 24*7*2] for weekly
    ROLLING_WINDOWS: list = None  # Will be set to [3, 6, 12, 24, 168] (hours)
    
    # Model parameters
    FORECAST_HORIZON: int = 24  # Forecast 24 hours ahead
    FORECAST_HORIZON_EXTENDED: int = 48  # Extended forecast
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "outputs/pipeline.log"
    
    def __post_init__(self):
        """Set default values for lists"""
        if self.LAG_FEATURES is None:
            self.LAG_FEATURES = [1, 2, 3, 6, 12, 24, 24*7, 24*7*2]  # hourly and weekly lags
        if self.ROLLING_WINDOWS is None:
            self.ROLLING_WINDOWS = [3, 6, 12, 24, 168]  # 3h, 6h, 12h, 24h, 7d
        
        # Create directories if they don't exist
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.OUTPUTS_DIR, exist_ok=True)
        if self.LOG_FILE:
            os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)

