"""
Evaluation metrics for time series forecasting
"""

import numpy as np
import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE score
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE score
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE score (as percentage)
    """
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    return 1 - (ss_res / ss_tot)


def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Error (bias)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Mean error (positive = overestimation, negative = underestimation)
    """
    return np.mean(y_pred - y_true)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (percentage of correct direction predictions)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Directional accuracy (0-100%)
    """
    if len(y_true) < 2:
        return np.nan
    
    # Calculate direction of change
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    accuracy = np.mean(true_direction == pred_direction) * 100
    
    return accuracy


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'Mean Error': mean_error(y_true, y_pred),
        'Directional Accuracy': directional_accuracy(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{model_name} Performance Metrics:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        if pd.isna(value):
            print(f"{metric_name:20s}: N/A")
        elif 'MAPE' in metric_name or 'Accuracy' in metric_name:
            print(f"{metric_name:20s}: {value:.2f}%")
        else:
            print(f"{metric_name:20s}: {value:.4f}")
    print("-" * 50)

