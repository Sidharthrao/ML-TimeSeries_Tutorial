"""
Concept drift detection utilities for time series monitoring
Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Union, Optional
import logging
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Class to detect concept drift between reference (train) and current (test/production) data
    """
    
    def __init__(self, psi_threshold: float = 0.1, p_value_threshold: float = 0.05):
        """
        Initialize DriftDetector
        
        Args:
            psi_threshold: Threshold for PSI warning (default 0.1)
                           < 0.1: No change
                           0.1 - 0.2: Minor drift
                           > 0.2: Significant drift
            p_value_threshold: p-value threshold for KS test (default 0.05)
        """
        self.psi_threshold = psi_threshold
        self.p_value_threshold = p_value_threshold

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            expected: Reference data (e.g., training set)
            actual: Current data (e.g., test set)
            buckets: Number of buckets for quantization
            
        Returns:
            PSI value
        """
        def scale_range(input, min_val, max_val):
            input += -(np.min(input))
            input /= np.max(input) / (max_val - min_val)
            input += min_val
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        
        try:
            expected_percents = np.percentile(expected, breakpoints)
            
            # Handle unique values being less than buckets
            if len(np.unique(expected_percents)) < len(breakpoints):
                # Fallback to linear spacing if percentiles collapse
                min_val, max_val = np.min(expected), np.max(expected)
                expected_percents = np.linspace(min_val, max_val, buckets + 1)

            actual_percents = expected_percents # Use same breakpoints for actual
            
            expected_cnt, _ = np.histogram(expected, expected_percents)
            actual_cnt, _ = np.histogram(actual, actual_percents)
            
            # Avoid division by zero
            expected_cnt = np.where(expected_cnt == 0, 0.0001, expected_cnt)
            actual_cnt = np.where(actual_cnt == 0, 0.0001, actual_cnt)
            
            expected_dist = expected_cnt / len(expected)
            actual_dist = actual_cnt / len(actual)
            
            psi_values = (actual_dist - expected_dist) * np.log(actual_dist / expected_dist)
            psi = np.sum(psi_values)
            
            return psi
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return np.nan

    def detect_drift_psi(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect drift using PSI for all common columns
        
        Args:
            train_df: Training data (reference)
            test_df: Test data (current)
            
        Returns:
            Dictionary of {column: psi_score}
        """
        logger.info("Checking for drift using PSI")
        
        drift_scores = {}
        common_cols = [c for c in train_df.columns if c in test_df.columns]
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                psi = self.calculate_psi(
                    train_df[col].dropna().values,
                    test_df[col].dropna().values
                )
                drift_scores[col] = psi
                
                if psi > self.psi_threshold:
                    logger.warning(f"Drift detected in {col} (PSI={psi:.4f})")
        
        return drift_scores

    def detect_drift_ks(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect drift using Kolmogorov-Smirnov test for all common columns
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            Dictionary of results per column
        """
        logger.info("Checking for drift using KS Test")
        
        results = {}
        common_cols = [c for c in train_df.columns if c in test_df.columns]
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                stat, p_value = ks_2samp(
                    train_df[col].dropna(),
                    test_df[col].dropna()
                )
                
                is_drift = p_value < self.p_value_threshold
                
                results[col] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_drift': is_drift
                }
                
                if is_drift:
                    logger.warning(f"Drift detected in {col} (KS p-value={p_value:.6f})")
        
        return results
