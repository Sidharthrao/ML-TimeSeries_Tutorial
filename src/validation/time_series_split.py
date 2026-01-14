"""
Time series data splitting and cross-validation utilities
"""

import pandas as pd
import numpy as np
from typing import Tuple, Generator, Optional
import logging

logger = logging.getLogger(__name__)


def time_series_split(
    df: pd.DataFrame,
    train_size: float = 0.85,
    test_size: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data chronologically into train and test sets
    
    Important: This maintains temporal order, unlike random splits.
    All training data comes before test data chronologically.
    
    Args:
        df: DataFrame with DatetimeIndex (sorted chronologically)
        train_size: Proportion of data for training (0.0 to 1.0)
        test_size: Proportion of data for testing (if None, uses 1 - train_size)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if not df.index.is_monotonic_increasing:
        logger.warning("DataFrame index is not sorted. Sorting now...")
        df = df.sort_index()
    
    n_total = len(df)
    n_train = int(n_total * train_size)
    
    if test_size is None:
        test_size = 1.0 - train_size
    
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    
    logger.info(f"Time series split: Train={len(train_df)} ({len(train_df)/n_total*100:.1f}%), "
                f"Test={len(test_df)} ({len(test_df)/n_total*100:.1f}%)")
    logger.info(f"Train period: {train_df.index.min()} to {train_df.index.max()}")
    logger.info(f"Test period: {test_df.index.min()} to {test_df.index.max()}")
    
    return train_df, test_df


def time_series_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_size: float = 0.7,
    test_size: float = 0.1,
    gap: int = 0
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Time series cross-validation generator (walk-forward validation)
    
    This implements a walk-forward validation strategy where:
    - Training window grows or slides forward
    - Test window moves forward chronologically
    - No data leakage (future data never used in training)
    
    Args:
        df: DataFrame with DatetimeIndex
        n_splits: Number of cross-validation splits
        train_size: Initial training set proportion
        test_size: Test set proportion for each split
        gap: Gap between training and test sets (in number of periods)
    
    Yields:
        Tuple of (train_df, test_df) for each fold
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    
    n_total = len(df)
    n_train_initial = int(n_total * train_size)
    n_test = int(n_total * test_size)
    
    # Calculate step size for moving the window
    remaining = n_total - n_train_initial - n_test
    step_size = max(1, remaining // n_splits) if n_splits > 1 else 0
    
    logger.info(f"Time series CV: {n_splits} folds, train_size={train_size}, test_size={test_size}")
    
    for i in range(n_splits):
        # Calculate split indices
        train_start = 0
        train_end = n_train_initial + (i * step_size)
        test_start = train_end + gap
        test_end = min(test_start + n_test, n_total)
        
        # Ensure we have enough data
        if test_end <= train_end:
            logger.warning(f"Fold {i+1}: Not enough data, skipping")
            continue
        
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        logger.info(f"Fold {i+1}: Train={len(train_df)}, Test={len(test_df)}, "
                   f"Train period: {train_df.index.min()} to {train_df.index.max()}, "
                   f"Test period: {test_df.index.min()} to {test_df.index.max()}")
        
        yield train_df, test_df

