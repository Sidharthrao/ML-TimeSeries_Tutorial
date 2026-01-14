"""
Feature selection utilities for time series models
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def select_features(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    method: str = 'correlation',
    threshold: float = 0.1,
    max_features: Optional[int] = None
) -> List[str]:
    """
    Select relevant features for modeling
    
    Methods:
    - 'correlation': Select features with correlation above threshold
    - 'all': Return all feature columns (excluding target)
    - 'manual': Return specific feature list
    
    Args:
        df: DataFrame with features and target
        target_col: Target variable column name
        method: Selection method
        threshold: Correlation threshold (for 'correlation' method)
        max_features: Maximum number of features to select
    
    Returns:
        List of selected feature column names
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Exclude target and datetime index from features
    feature_cols = [col for col in df.columns if col != target_col]
    
    if method == 'all':
        selected = feature_cols
    elif method == 'correlation':
        # Calculate correlation with target
        correlations = df[feature_cols].corrwith(df[target_col]).abs()
        selected = correlations[correlations >= threshold].index.tolist()
        logger.info(f"Selected {len(selected)} features with correlation >= {threshold}")
    elif method == 'manual':
        # Return specific important features
        selected = [
            col for col in feature_cols
            if any(x in col for x in ['hour', 'day_of_week', 'lag', 'rolling', 'month'])
        ]
    elif method == 'rfe':
        # Recursive Feature Elimination
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found for RFE")
            
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor
            
            # Prepare data
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            # Handle missing values for RFE
            X = X.fillna(method='ffill').fillna(0)
            y = y.fillna(method='ffill').fillna(0)
            
            # Initialize estimator
            estimator = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
            
            # Select number of features
            n_features = max_features if max_features else min(len(feature_cols), 20)
            
            logger.info(f"Running RFE to select {n_features} features")
            selector = RFE(estimator, n_features_to_select=n_features, step=1)
            selector = selector.fit(X, y)
            
            selected = pd.Series(feature_cols)[selector.support_].tolist()
            
        except ImportError:
            logger.warning("scikit-learn not available. Fallback to correlation.")
            return select_features(df, target_col, method='correlation', max_features=max_features)
            
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Limit number of features if specified (and not already done by RFE)
    if method != 'rfe' and max_features and len(selected) > max_features:
        if method == 'correlation':
            # Keep top correlated features
            correlations = df[selected].corrwith(df[target_col]).abs()
            selected = correlations.nlargest(max_features).index.tolist()
        else:
            selected = selected[:max_features]
    
    logger.info(f"Selected {len(selected)} features using {method} for modeling")
    
    return selected

