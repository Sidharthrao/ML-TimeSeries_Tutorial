"""
Stationarity testing module for time series data
Implements ADF and KPSS tests to check for stationarity and suggest differencing
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Union, Optional
import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import statsmodels
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Stationarity tests will not work.")


class StationarityTester:
    """
    Class to perform stationarity tests on time series data
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize StationarityTester
        
        Args:
            significance_level: p-value threshold for significance (default 0.05)
        """
        self.significance_level = significance_level
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for stationarity testing")

    def check_stationarity_adf(self, series: pd.Series) -> Dict[str, Union[float, bool, str]]:
        """
        Perform Augmented Dickey-Fuller (ADF) test
        Null Hypothesis (H0): The series has a unit root (non-stationary)
        Alternate Hypothesis (H1): The series has no unit root (stationary)
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary containing test results
        """
        logger.info("Performing Augmented Dickey-Fuller (ADF) test")
        
        # Drop NaNs
        clean_series = series.dropna()
        if len(clean_series) < 10:
            logger.warning("Series too short for reliable ADF test")
        
        # Perform test
        try:
            result = adfuller(clean_series, autolag='AIC')
            
            p_value = result[1]
            test_stat = result[0]
            is_stationary = p_value < self.significance_level
            
            metrics = {
                'test': 'ADF',
                'test_statistic': test_stat,
                'p_value': p_value,
                'is_stationary': is_stationary,
                'n_lags': result[2],
                'critical_values': result[4],
                'conclusion': 'Stationary' if is_stationary else 'Non-Stationary'
            }
            
            logger.info(f"ADF Result: p-value={p_value:.4f}, Stationary={is_stationary}")
            return metrics
            
        except Exception as e:
            logger.error(f"ADF test failed: {e}")
            raise

    def check_stationarity_kpss(self, series: pd.Series, regression: str = 'c') -> Dict[str, Union[float, bool, str]]:
        """
        Perform KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
        Null Hypothesis (H0): The series is stationary directly (or trend-stationary)
        Alternate Hypothesis (H1): The series has a unit root (non-stationary)
        
        Note: Hypothesis is opposite of ADF!
        
        Args:
            series: Time series data
            regression: 'c' (constant, level stationarity) or 'ct' (constant+trend, trend stationarity)
            
        Returns:
            Dictionary containing test results
        """
        logger.info(f"Performing KPSS test (regression='{regression}')")
        
        # Drop NaNs
        clean_series = series.dropna()
        
        # Perform test
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Interpolation dependent on the number of observations")
                result = kpss(clean_series, regression=regression, nlags='auto')
            
            p_value = result[1]
            test_stat = result[0]
            # H0 is stationarity, so if p < alpha, we reject H0 -> Non-Stationary
            is_stationary = p_value >= self.significance_level
            
            metrics = {
                'test': 'KPSS',
                'test_statistic': test_stat,
                'p_value': p_value,
                'is_stationary': is_stationary,
                'n_lags': result[2],
                'critical_values': result[3],
                'conclusion': 'Stationary' if is_stationary else 'Non-Stationary'
            }
            
            logger.info(f"KPSS Result: p-value={p_value:.4f}, Stationary={is_stationary}")
            return metrics
            
        except Exception as e:
            logger.error(f"KPSS test failed: {e}")
            raise

    def suggest_differencing(self, series: pd.Series, max_d: int = 2) -> int:
        """
        Iteratively check stationarity to suggest differencing order (d)
        
        Args:
            series: Time series data
            max_d: Maximum differencing order to check
            
        Returns:
            Suggested order of differencing (0 to max_d)
        """
        current_series = series.copy()
        
        for d in range(max_d + 1):
            logger.info(f"Checking stationarity for d={d}")
            
            # Use ADF as primary check
            adf_result = self.check_stationarity_adf(current_series)
            
            # Use KPSS as confirmation (optional but good practice)
            # kpss_result = self.check_stationarity_kpss(current_series)
            
            if adf_result['is_stationary']:
                logger.info(f"Series is stationary at d={d}")
                return d
            
            # If not stationary and we haven't hit max_d, difference the series
            if d < max_d:
                current_series = current_series.diff().dropna()
        
        logger.warning(f"Series remained non-stationary even after d={max_d} differencing")
        return max_d
