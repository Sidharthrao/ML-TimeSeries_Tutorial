"""
Feature engineering for time series forecasting
Creates temporal, lag, rolling, and derived features
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from datetime index
    
    Features created:
    - Hour of day (0-23)
    - Day of week (0=Monday, 6=Sunday)
    - Day of month (1-31)
    - Month (1-12)
    - Year
    - Is weekend (boolean)
    - Cyclical encoding (sine/cosine) for hour, day of week, month
    
    Args:
        df: DataFrame with DatetimeIndex
    
    Returns:
        DataFrame with temporal features added
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    logger.info("Creating temporal features")
    
    df_features = df.copy()
    
    # Basic temporal features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    df_features['quarter'] = df_features.index.quarter
    df_features['week_of_year'] = df_features.index.isocalendar().week
    
    # Derived features
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    df_features['is_weekday'] = (df_features['day_of_week'] < 5).astype(int)
    
    # Cyclical encoding for hour (24-hour cycle)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    # Cyclical encoding for month (12-month cycle)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    logger.info(f"Created {len([c for c in df_features.columns if c not in df.columns])} temporal features")
    
    return df_features


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Create lag features (previous time step values)
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column to create lags for
        lags: List of lag periods (e.g., [1, 2, 3, 24, 168] for 1h, 2h, 3h, 1d, 1w)
    
    Returns:
        DataFrame with lag features added
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 24*7, 24*7*2]  # Default: hourly and weekly lags
    
    logger.info(f"Creating lag features for lags: {lags}")
    
    df_features = df.copy()
    
    for lag in lags:
        df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Special lags: same hour previous day, same hour previous week
    if isinstance(df_features.index, pd.DatetimeIndex):
        # Previous day same hour (approximately 24 hours ago)
        df_features[f'{target_col}_lag_prev_day'] = df_features[target_col].shift(24)
        # Previous week same hour (approximately 168 hours ago)
        df_features[f'{target_col}_lag_prev_week'] = df_features[target_col].shift(168)
    
    logger.info(f"Created {len(lags) + 2} lag features")
    
    return df_features


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    windows: List[int] = None,
    functions: List[str] = None
) -> pd.DataFrame:
    """
    Create rolling window statistics
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column
        windows: List of window sizes (in hours)
        functions: List of functions to apply ('mean', 'std', 'min', 'max', 'median')
    
    Returns:
        DataFrame with rolling features added
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    if windows is None:
        windows = [3, 6, 12, 24, 168]  # 3h, 6h, 12h, 24h, 7d
    
    if functions is None:
        functions = ['mean', 'std', 'min', 'max']
    
    logger.info(f"Creating rolling features with windows: {windows}, functions: {functions}")
    
    df_features = df.copy()
    
    for window in windows:
        rolling = df_features[target_col].rolling(window=window, min_periods=1)
        
        if 'mean' in functions:
            df_features[f'{target_col}_rolling_mean_{window}'] = rolling.mean()
        if 'std' in functions:
            df_features[f'{target_col}_rolling_std_{window}'] = rolling.std()
        if 'min' in functions:
            df_features[f'{target_col}_rolling_min_{window}'] = rolling.min()
        if 'max' in functions:
            df_features[f'{target_col}_rolling_max_{window}'] = rolling.max()
        if 'median' in functions:
            df_features[f'{target_col}_rolling_median_{window}'] = rolling.median()
    
    # Exponential weighted moving averages
    for alpha in [0.1, 0.3, 0.5]:
        df_features[f'{target_col}_ewm_{alpha}'] = df_features[target_col].ewm(alpha=alpha, adjust=False).mean()
    
    logger.info(f"Created {len(windows) * len(functions) + 3} rolling features")
    
    return df_features


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns
    
    Features:
    - Power factor (from active/reactive power)
    - Energy consumption rates
    - Differences from rolling means
    
    Args:
        df: DataFrame with power consumption columns
    
    Returns:
        DataFrame with derived features added
    """
    logger.info("Creating derived features")
    
    df_features = df.copy()
    
    # Power factor (if reactive power available)
    if 'Global_active_power' in df.columns and 'Global_reactive_power' in df.columns:
        # Apparent power
        df_features['apparent_power'] = np.sqrt(
            df_features['Global_active_power']**2 + df_features['Global_reactive_power']**2
        )
        # Power factor
        df_features['power_factor'] = np.where(
            df_features['apparent_power'] > 0,
            df_features['Global_active_power'] / df_features['apparent_power'],
            0
        )
    
    # Differences from rolling means (if target column exists)
    if 'Global_active_power' in df.columns:
        for window in [24, 168]:  # Daily and weekly
            rolling_mean = df_features['Global_active_power'].rolling(window=window, min_periods=1).mean()
            df_features[f'Global_active_power_diff_from_mean_{window}'] = (
                df_features['Global_active_power'] - rolling_mean
            )
    
    # Hourly change (if target exists)
    if 'Global_active_power' in df.columns:
        df_features['Global_active_power_change'] = df_features['Global_active_power'].diff()
        df_features['Global_active_power_pct_change'] = df_features['Global_active_power'].pct_change()
    
    logger.info("Created derived features")
    
    return df_features


def create_all_features(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    lags: List[int] = None,
    rolling_windows: List[int] = None
) -> pd.DataFrame:
    """
    Create all feature engineering steps in sequence
    
    Args:
        df: Input DataFrame
        target_col: Target variable column
        lags: List of lag periods
        rolling_windows: List of rolling window sizes
    
    Returns:
        DataFrame with all features added
    """
    logger.info("Starting complete feature engineering pipeline")
    
    # Step 1: Temporal features
    df = create_temporal_features(df)
    
    # Step 2: Lag features
    df = create_lag_features(df, target_col=target_col, lags=lags)
    
    # Step 3: Rolling features
    df = create_rolling_features(df, target_col=target_col, windows=rolling_windows)
    
    # Step 4: Derived features
    df = create_derived_features(df)
    
    logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
    
    return df

