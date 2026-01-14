"""
Classical time series forecasting models
ARIMA, SARIMA, ETS, STL decomposition
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import STL
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available. Classical models will not work.")

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    logger.warning("pmdarima not available. Auto-ARIMA will not work.")


class ARIMAModel:
    """ARIMA (AutoRegressive Integrated Moving Average) model"""
    
    def __init__(self, order: Optional[Tuple[int, int, int]] = None, use_auto: bool = True):
        """
        Initialize ARIMA model
        
        Args:
            order: (p, d, q) order tuple. If None and use_auto=True, will auto-select
            use_auto: Whether to use auto-ARIMA for parameter selection
        """
        self.order = order
        self.use_auto = use_auto
        self.model = None
        self.fitted_model = None
    
    def fit(self, y: pd.Series, **kwargs) -> 'ARIMAModel':
        """Fit ARIMA model to data"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA models")
        
        logger.info("Fitting ARIMA model")
        
        if self.use_auto and PMDARIMA_AVAILABLE and self.order is None:
            logger.info("Using auto-ARIMA for parameter selection")
            auto_model = pm.auto_arima(
                y,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                d=kwargs.get('d', None),
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            self.order = auto_model.order
            logger.info(f"Auto-ARIMA selected order: {self.order}")
            self.fitted_model = auto_model
        else:
            if self.order is None:
                # Default order
                self.order = (1, 1, 1)
                logger.info(f"Using default ARIMA order: {self.order}")
            
            self.model = ARIMA(y, order=self.order)
            self.fitted_model = self.model.fit(**kwargs)
        
        logger.info("ARIMA model fitted successfully")
        return self
    
    def predict(self, n_periods: int, **kwargs) -> np.ndarray:
        """Forecast n periods ahead"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if PMDARIMA_AVAILABLE and hasattr(self.fitted_model, 'predict'):
            forecasts = self.fitted_model.predict(n_periods=n_periods, **kwargs)
        else:
            forecasts = self.fitted_model.forecast(steps=n_periods, **kwargs)
        
        return np.array(forecasts)
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values from training"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.fitted_model, 'fittedvalues'):
            return self.fitted_model.fittedvalues.values
        elif hasattr(self.fitted_model, 'predict_in_sample'):
            return self.fitted_model.predict_in_sample()
        else:
            return np.array([])


class SARIMAModel:
    """SARIMA (Seasonal ARIMA) model"""
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
        use_auto: bool = True
    ):
        """
        Initialize SARIMA model
        
        Args:
            order: (p, d, q) non-seasonal order
            seasonal_order: (P, D, Q, s) seasonal order (s is period)
            use_auto: Whether to use auto-SARIMA
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_auto = use_auto
        self.fitted_model = None
    
    def fit(self, y: pd.Series, **kwargs) -> 'SARIMAModel':
        """Fit SARIMA model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for SARIMA models")
        
        logger.info(f"Fitting SARIMA model with order={self.order}, seasonal_order={self.seasonal_order}")
        
        if self.use_auto and PMDARIMA_AVAILABLE:
            logger.info("Using auto-SARIMA for parameter selection")
            auto_model = pm.auto_arima(
                y,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=True,
                m=self.seasonal_order[3],  # Period
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            self.order = auto_model.order
            self.seasonal_order = auto_model.seasonal_order
            logger.info(f"Auto-SARIMA selected order: {self.order}, seasonal: {self.seasonal_order}")
            self.fitted_model = auto_model
        else:
            self.fitted_model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                **kwargs
            ).fit(disp=False)
        
        logger.info("SARIMA model fitted successfully")
        return self
    
    def predict(self, n_periods: int, **kwargs) -> np.ndarray:
        """Forecast n periods ahead"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if PMDARIMA_AVAILABLE and hasattr(self.fitted_model, 'predict'):
            forecasts = self.fitted_model.predict(n_periods=n_periods, **kwargs)
        else:
            forecasts = self.fitted_model.forecast(steps=n_periods, **kwargs)
        
        return np.array(forecasts)
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.fitted_model, 'fittedvalues'):
            return self.fitted_model.fittedvalues.values
        elif hasattr(self.fitted_model, 'predict_in_sample'):
            return self.fitted_model.predict_in_sample()
        else:
            return np.array([])


class ETSModel:
    """Exponential Smoothing (ETS) model"""
    
    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = 24
    ):
        """
        Initialize ETS model
        
        Args:
            trend: 'add', 'mul', or None
            seasonal: 'add', 'mul', or None
            seasonal_periods: Number of periods in a season
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None
    
    def fit(self, y: pd.Series, **kwargs) -> 'ETSModel':
        """Fit ETS model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ETS models")
        
        logger.info(f"Fitting ETS model with trend={self.trend}, seasonal={self.seasonal}")
        
        try:
            self.fitted_model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                **kwargs
            ).fit()
            logger.info("ETS model fitted successfully")
        except Exception as e:
            logger.warning(f"ETS model fitting failed: {e}. Trying simpler configuration.")
            # Try without seasonal component
            try:
                self.fitted_model = ExponentialSmoothing(
                    y,
                    trend=self.trend,
                    seasonal=None,
                    **kwargs
                ).fit()
                logger.info("ETS model fitted with simplified configuration")
            except Exception as e2:
                logger.error(f"ETS model fitting failed completely: {e2}")
                raise
        
        return self
    
    def predict(self, n_periods: int, **kwargs) -> np.ndarray:
        """Forecast n periods ahead"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts = self.fitted_model.forecast(steps=n_periods, **kwargs)
        return np.array(forecasts)
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        return self.fitted_model.fittedvalues.values


class STLModel:
    """STL (Seasonal and Trend decomposition using Loess) forecasting model"""
    
    def __init__(self, period: int = 24, robust: bool = True):
        """
        Initialize STL model
        
        Args:
            period: Seasonal period
            robust: Whether to use robust decomposition
        """
        self.period = period
        self.robust = robust
        self.decomposition = None
        self.trend_model = None
        self.seasonal_component = None
    
    def fit(self, y: pd.Series, **kwargs) -> 'STLModel':
        """Fit STL decomposition and forecast models"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for STL models")
        
        logger.info(f"Fitting STL decomposition with period={self.period}")
        
        # Perform STL decomposition
        self.decomposition = STL(y, period=self.period, robust=self.robust).fit()
        
        # Extract components
        trend = self.decomposition.trend
        seasonal = self.decomposition.seasonal
        residual = self.decomposition.resid
        
        # Store seasonal component for forecasting
        self.seasonal_component = seasonal[-self.period:].values
        
        # Fit simple model on trend (using ARIMA)
        trend_clean = trend.dropna()
        if len(trend_clean) > 0:
            try:
                self.trend_model = ARIMA(trend_clean, order=(1, 1, 1)).fit()
            except:
                # Fallback to simple moving average
                self.trend_model = None
        
        logger.info("STL model fitted successfully")
        return self
    
    def predict(self, n_periods: int, **kwargs) -> np.ndarray:
        """Forecast n periods ahead using STL decomposition"""
        if self.decomposition is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Forecast trend
        if self.trend_model is not None:
            trend_forecast = self.trend_model.forecast(steps=n_periods)
        else:
            # Use last trend value
            last_trend = self.decomposition.trend.iloc[-1]
            trend_forecast = np.full(n_periods, last_trend)
        
        # Extend seasonal component
        seasonal_forecast = np.tile(self.seasonal_component, (n_periods // self.period + 1))[:n_periods]
        
        # Combine trend and seasonal
        forecasts = trend_forecast + seasonal_forecast
        
        return np.array(forecasts)
    
    def get_fitted_values(self) -> np.ndarray:
        """Get fitted values from decomposition"""
        if self.decomposition is None:
            raise ValueError("Model must be fitted first")
        
        return (self.decomposition.trend + self.decomposition.seasonal).values

