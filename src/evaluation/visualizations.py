"""
Visualization utilities for time series forecasting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_forecast(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Forecast vs Actual",
    save_path: Optional[str] = None
) -> None:
    """
    Plot forecasted vs actual values
    
    Args:
        y_true: True time series values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(14, 6))
    
    # Create index for predictions if not provided
    if isinstance(y_pred, pd.Series):
        pred_index = y_pred.index
        y_pred = y_pred.values
    else:
        # Assume predictions start after training data
        if len(y_pred) <= len(y_true):
            pred_index = y_true.index[-len(y_pred):]
        else:
            # Extend index for future predictions
            last_date = y_true.index[-1]
            pred_index = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=len(y_pred),
                freq='H'
            )
    
    plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2, alpha=0.7)
    plt.plot(pred_index, y_pred, label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Global Active Power (kW)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Plot residual analysis (residuals over time and distribution)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    axes[0, 0].plot(residuals, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residual Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[1, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Residuals vs Predicted', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_forecast_intervals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    title: str = "Forecast with Confidence Intervals",
    save_path: Optional[str] = None
) -> None:
    """
    Plot forecast with confidence intervals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        lower: Lower confidence interval
        upper: Upper confidence interval
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(14, 6))
    
    # Plot actual
    plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2, color='blue', alpha=0.7)
    
    # Plot predictions
    if isinstance(y_pred, pd.Series):
        pred_index = y_pred.index
        y_pred = y_pred.values
    else:
        if len(y_pred) <= len(y_true):
            pred_index = y_true.index[-len(y_pred):]
        else:
            last_date = y_true.index[-1]
            pred_index = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=len(y_pred),
                freq='H'
            )
    
    plt.plot(pred_index, y_pred, label='Forecast', linewidth=2, color='red', linestyle='--')
    
    # Plot confidence intervals if provided
    if lower is not None and upper is not None:
        plt.fill_between(
            pred_index,
            lower,
            upper,
            alpha=0.3,
            color='red',
            label='95% Confidence Interval'
        )
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Global Active Power (kW)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str = 'RMSE',
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple models using a bar chart
    
    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
        metric_name: Metric to compare
        save_path: Path to save figure (optional)
    """
    models = list(metrics_dict.keys())
    values = [metrics_dict[model].get(metric_name, np.nan) for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, values, alpha=0.7, edgecolor='black')
    
    # Color bars based on value (lower is better for error metrics)
    if 'Error' in metric_name or 'RMSE' in metric_name or 'MAE' in metric_name or 'MAPE' in metric_name:
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(models)))
    else:
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(models)))
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Model Comparison: {metric_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (model, value) in enumerate(zip(models, values)):
        if not pd.isna(value):
            plt.text(i, value, f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple models across multiple metrics
    
    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
        metrics_to_plot: List of metrics to plot (if None, plots all)
        save_path: Path to save figure (optional)
    """
    if metrics_to_plot is None:
        # Get all metrics from first model
        metrics_to_plot = list(list(metrics_dict.values())[0].keys())
    
    models = list(metrics_dict.keys())
    n_metrics = len(metrics_to_plot)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [metrics_dict[model].get(metric, np.nan) for model in models]
        bars = axes[idx].bar(models, values, alpha=0.7, edgecolor='black')
        
        # Color bars
        if 'Error' in metric or 'RMSE' in metric or 'MAE' in metric or 'MAPE' in metric:
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(models)))
        else:
            colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(models)))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        axes[idx].set_title(metric, fontweight='bold')
        axes[idx].set_xticklabels(models, rotation=45, ha='right')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, value in enumerate(values):
            if not pd.isna(value):
                axes[idx].text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Model Comparison Across Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

