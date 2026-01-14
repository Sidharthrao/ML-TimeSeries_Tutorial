"""
Machine Learning models for time series forecasting
XGBoost, Random Forest, Linear Regression
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML models will not work.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. XGBoost model will not work.")


class XGBoostModel:
    """XGBoost model for time series forecasting"""
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost model
        
        Args:
            **kwargs: XGBoost parameters
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for XGBoostModel")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.feature_names = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False,
        cv_folds: int = 3
    ) -> 'XGBoostModel':
        """Fit XGBoost model"""
        logger.info("Fitting XGBoost model")
        
        self.feature_names = X.columns.tolist()
        
        if tune_hyperparameters:
            logger.info("Tuning XGBoost hyperparameters")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X, y)
        
        logger.info("XGBoost model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target values"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        if self.feature_names is None:
            return pd.Series()
        
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        return importance


class RandomForestModel:
    """Random Forest model for time series forecasting"""
    
    def __init__(self, **kwargs):
        """
        Initialize Random Forest model
        
        Args:
            **kwargs: Random Forest parameters
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RandomForestModel")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        self.model = RandomForestRegressor(**default_params)
        self.feature_names = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False,
        cv_folds: int = 3
    ) -> 'RandomForestModel':
        """Fit Random Forest model"""
        logger.info("Fitting Random Forest model")
        
        self.feature_names = X.columns.tolist()
        
        if tune_hyperparameters:
            logger.info("Tuning Random Forest hyperparameters")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X, y)
        
        logger.info("Random Forest model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target values"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        if self.feature_names is None:
            return pd.Series()
        
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        return importance


class LinearRegressionModel:
    """Linear Regression with regularization for time series forecasting"""
    
    def __init__(self, model_type: str = 'ridge', alpha: float = 1.0):
        """
        Initialize Linear Regression model
        
        Args:
            model_type: 'ridge', 'lasso', or 'elastic_net'
            alpha: Regularization strength
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for LinearRegressionModel")
        
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        elif model_type == 'elastic_net':
            self.model = ElasticNet(alpha=alpha, random_state=42, max_iter=2000)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'ridge', 'lasso', or 'elastic_net'")
        
        self.model_type = model_type
        self.feature_names = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = False,
        cv_folds: int = 3
    ) -> 'LinearRegressionModel':
        """Fit Linear Regression model"""
        logger.info(f"Fitting {self.model_type} Linear Regression model")
        
        self.feature_names = X.columns.tolist()
        
        if tune_hyperparameters:
            logger.info(f"Tuning {self.model_type} hyperparameters")
            if self.model_type == 'elastic_net':
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            else:
                param_grid = {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model.fit(X, y)
        
        logger.info(f"{self.model_type} Linear Regression model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target values"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_coefficients(self) -> pd.Series:
        """Get model coefficients"""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        if self.feature_names is None:
            return pd.Series()
        
        coef = pd.Series(
            self.model.coef_,
            index=self.feature_names
        ).sort_values(key=abs, ascending=False)
        
        return coef

