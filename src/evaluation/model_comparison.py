"""
Model comparison and selection utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compare_models(
    metrics_dict: Dict[str, Dict[str, float]],
    primary_metric: str = 'RMSE',
    lower_is_better: bool = True
) -> pd.DataFrame:
    """
    Compare multiple models and create comparison DataFrame
    
    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
        primary_metric: Primary metric for ranking
        lower_is_better: Whether lower values are better for primary metric
    
    Returns:
        DataFrame with model comparison
    """
    # Create DataFrame
    comparison_df = pd.DataFrame(metrics_dict).T
    
    # Rank models by primary metric
    if primary_metric in comparison_df.columns:
        if lower_is_better:
            comparison_df['Rank'] = comparison_df[primary_metric].rank(ascending=True)
        else:
            comparison_df['Rank'] = comparison_df[primary_metric].rank(ascending=False)
        comparison_df = comparison_df.sort_values('Rank')
    else:
        logger.warning(f"Primary metric '{primary_metric}' not found in metrics")
    
    return comparison_df


def select_best_model(
    metrics_dict: Dict[str, Dict[str, float]],
    primary_metric: str = 'RMSE',
    lower_is_better: bool = True
) -> Tuple[str, Dict[str, float]]:
    """
    Select best model based on primary metric
    
    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
        primary_metric: Primary metric for selection
        lower_is_better: Whether lower values are better
    
    Returns:
        Tuple of (best_model_name, best_model_metrics)
    """
    if not metrics_dict:
        raise ValueError("metrics_dict is empty")
    
    best_model = None
    best_value = None
    
    for model_name, metrics in metrics_dict.items():
        if primary_metric not in metrics:
            logger.warning(f"Metric '{primary_metric}' not found for model '{model_name}'")
            continue
        
        value = metrics[primary_metric]
        
        if pd.isna(value):
            continue
        
        if best_value is None:
            best_value = value
            best_model = model_name
        elif lower_is_better and value < best_value:
            best_value = value
            best_model = model_name
        elif not lower_is_better and value > best_value:
            best_value = value
            best_model = model_name
    
    if best_model is None:
        raise ValueError(f"Could not find best model for metric '{primary_metric}'")
    
    logger.info(f"Best model selected: {best_model} (based on {primary_metric}={best_value:.4f})")
    
    return best_model, metrics_dict[best_model]


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy
    
    Tests the null hypothesis that two forecasts have equal accuracy.
    
    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        h: Forecast horizon (for adjusting autocorrelation)
    
    Returns:
        Tuple of (test_statistic, p_value)
    """
    try:
        from scipy import stats
    except ImportError:
        logger.warning("scipy not available for Diebold-Mariano test")
        return np.nan, np.nan
    
    # Loss differential
    d = errors1**2 - errors2**2
    
    # Mean and variance
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    if d_var == 0:
        return np.nan, np.nan
    
    # Test statistic
    n = len(d)
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # P-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def generate_evaluation_report(
    metrics_dict: Dict[str, Dict[str, float]],
    primary_metric: str = 'RMSE',
    save_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive evaluation report
    
    Args:
        metrics_dict: Dictionary of model metrics
        primary_metric: Primary metric for ranking
        save_path: Path to save report (optional)
    
    Returns:
        Report as string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TIME SERIES FORECASTING MODEL EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Model comparison table
    comparison_df = compare_models(metrics_dict, primary_metric)
    report_lines.append("MODEL COMPARISON")
    report_lines.append("-" * 80)
    report_lines.append(comparison_df.to_string())
    report_lines.append("")
    
    # Best model
    best_model, best_metrics = select_best_model(metrics_dict, primary_metric)
    report_lines.append(f"BEST MODEL: {best_model}")
    report_lines.append("-" * 80)
    for metric, value in best_metrics.items():
        if pd.isna(value):
            report_lines.append(f"{metric:20s}: N/A")
        elif 'MAPE' in metric or 'Accuracy' in metric:
            report_lines.append(f"{metric:20s}: {value:.2f}%")
        else:
            report_lines.append(f"{metric:20s}: {value:.4f}")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 80)
    for metric in ['MAE', 'RMSE', 'MAPE', 'R²']:
        if metric in comparison_df.columns:
            values = comparison_df[metric].dropna()
            if len(values) > 0:
                report_lines.append(f"{metric}:")
                report_lines.append(f"  Min:  {values.min():.4f} ({values.idxmin()})")
                report_lines.append(f"  Max:  {values.max():.4f} ({values.idxmax()})")
                report_lines.append(f"  Mean: {values.mean():.4f}")
                report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to {save_path}")
    
    return report

