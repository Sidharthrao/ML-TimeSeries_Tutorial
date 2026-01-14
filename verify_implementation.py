"""
Verification script for new time series features:
- Stationarity Testing
- Drift Detection
- Feature Selection (RFE)
- Multivariate DL Models
"""

import pandas as pd
import numpy as np
import logging
from src.validation.stationarity import StationarityTester
from src.validation.drift_detection import DriftDetector
from src.features.feature_selection import select_features
from src.models.dl_models import LSTMModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_stationarity():
    logger.info("--- Verifying Stationarity Tests ---")
    tester = StationarityTester()
    
    # Generate stationary (white noise)
    np.random.seed(42)
    stationary_series = pd.Series(np.random.normal(0, 1, 500))
    
    # Generate non-stationary (random walk)
    non_stationary_series = pd.Series(np.random.normal(0, 1, 500).cumsum())
    
    # Test Stationary
    res_adf = tester.check_stationarity_adf(stationary_series)
    res_kpss = tester.check_stationarity_kpss(stationary_series)
    logger.info(f"Stationary Series - ADF: {res_adf['conclusion']}, KPSS: {res_kpss['conclusion']}")
    assert res_adf['is_stationary'], "ADF failed for white noise"
    
    # Test Non-Stationary
    res_adf_ns = tester.check_stationarity_adf(non_stationary_series)
    logger.info(f"Non-Stationary Series - ADF: {res_adf_ns['conclusion']}")
    assert not res_adf_ns['is_stationary'], "ADF failed for random walk"
    
    # Suggest Differencing
    d = tester.suggest_differencing(non_stationary_series)
    logger.info(f"Suggested differencing order: {d}")
    assert d >= 1, "Failed to suggest differencing for random walk"
    
    logger.info("Stationarity verification passed!")

def verify_drift_detection():
    logger.info("\n--- Verifying Drift Detection ---")
    detector = DriftDetector()
    
    # No Drift
    train_normal = pd.DataFrame({'f1': np.random.normal(0, 1, 1000)})
    test_normal = pd.DataFrame({'f1': np.random.normal(0, 1, 1000)})
    
    # Drift (Mean Shift)
    test_drift = pd.DataFrame({'f1': np.random.normal(2, 1, 1000)})
    
    # Test PSI
    psi_no_drift = detector.detect_drift_psi(train_normal, test_normal)
    psi_drift = detector.detect_drift_psi(train_normal, test_drift)
    
    logger.info(f"PSI (No Drift): {psi_no_drift['f1']:.4f}")
    logger.info(f"PSI (Drift): {psi_drift['f1']:.4f}")
    
    assert psi_no_drift['f1'] < 0.2, "False positive drift (PSI)"
    assert psi_drift['f1'] > 0.2, "False negative drift (PSI)"
    
    logger.info("Drift detection verification passed!")

def verify_feature_selection():
    logger.info("\n--- Verifying Feature Selection (RFE) ---")
    
    # Synthetic data: y = 2*x1 + 3*x2 + noise. x3 is pure noise.
    df = pd.DataFrame({
        'target': np.random.normal(0, 1, 100),
        'feat_important_1': np.random.normal(0, 1, 100),
        'feat_important_2': np.random.normal(0, 1, 100),
        'feat_noise': np.random.normal(0, 1, 100)
    })
    # Make target dependent
    df['target'] = 2 * df['feat_important_1'] + 3 * df['feat_important_2'] + np.random.normal(0, 0.1, 100)
    
    selected = select_features(df, target_col='target', method='rfe', max_features=2)
    logger.info(f"Selected features: {selected}")
    
    assert 'feat_important_1' in selected, "Missed important feature 1"
    assert 'feat_important_2' in selected, "Missed important feature 2"
    assert 'feat_noise' not in selected, "Selected noise feature"
    
    logger.info("Feature selection verification passed!")

def verify_multivariate_dl():
    logger.info("\n--- Verifying Multivariate DL Models ---")
    
    try:
        # Synthetic Multivariate Data
        # Target depends on Lag-1 of target AND Lag-1 of Exogenous
        data = pd.DataFrame({
            'target': np.zeros(200),
            'exog': np.random.normal(0, 1, 200)
        })
        for i in range(1, 200):
            data.loc[i, 'target'] = 0.5 * data.loc[i-1, 'target'] + 0.5 * data.loc[i-1, 'exog'] + np.random.normal(0, 0.1)
            
        lstm = LSTMModel(sequence_length=10, units=20)
        
        # Fit with DataFrame (Multivariate)
        logger.info("Fitting Multivariate LSTM...")
        lstm.fit(data, target_col='target', epochs=5)
        
        # Predict
        last_seq = data.values[-10:] # Shape (10, 2)
        preds = lstm.predict(n_periods=5, last_sequence=last_seq)
        logger.info(f"Predictions: {preds}")
        
        assert len(preds) == 5, "Prediction length mismatch"
        logger.info("Multivariate DL verification passed!")
        
    except ImportError:
        logger.warning("Skipping DL verification (TensorFlow/Keras not installed)")

if __name__ == "__main__":
    verify_stationarity()
    verify_drift_detection()
    verify_feature_selection()
    verify_multivariate_dl()
