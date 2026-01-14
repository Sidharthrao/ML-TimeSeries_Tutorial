# Electricity Consumption Time Series Forecasting

## Project Overview

This project implements a comprehensive time series forecasting pipeline for predicting household electricity consumption. The system forecasts the next hour's average Global_active_power consumption using minute-level electricity consumption data.

## Features

- **Data Preprocessing**: Handles missing values, resampling to hourly frequency, outlier detection
- **Exploratory Data Analysis**: Time series decomposition, stationarity tests, autocorrelation analysis
- **Feature Engineering**: Temporal features, lag features, rolling statistics, derived features
- **Multiple Model Types**:
  - Classical: ARIMA, SARIMA, ETS, STL
  - Machine Learning: XGBoost, Random Forest, Linear Regression
  - Advanced: LSTM, GRU, Prophet
- **Comprehensive Evaluation**: Multiple metrics (MAE, RMSE, MAPE, R²), visualizations, model comparison
- **Production Ready**: Modular architecture, logging, configuration management

## Project Structure

```
Capstone_Project - Time Series/
├── Data/
│   └── household_power_consumption.txt
├── notebooks/
│   └── electricity_forecasting_pipeline.ipynb  # Main comprehensive notebook
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Forecasting models
│   ├── evaluation/        # Metrics and visualizations
│   ├── validation/        # Time series splitting
│   └── utils/             # Configuration and logging
├── models/                # Saved model artifacts
├── outputs/               # Plots and reports
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository or navigate to the project directory

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

The main workflow is in the Jupyter notebook:

```bash
jupyter notebook notebooks/electricity_forecasting_pipeline.ipynb
```

The notebook contains:
- Complete data loading and preprocessing
- Comprehensive EDA with visualizations
- Feature engineering pipeline
- Model training (all model types)
- Model evaluation and comparison
- Final forecast generation

### Using Individual Modules

You can also use the modular code directly:

```python
from src.data.data_loader import load_data
from src.data.data_preprocessor import preprocess_data
from src.models.classical_models import ARIMAModel
from src.evaluation.metrics import calculate_metrics

# Load and preprocess data
df = load_data("Data/household_power_consumption.txt")
df_processed = preprocess_data(df)

# Train model
model = ARIMAModel(use_auto=True)
model.fit(df_processed['Global_active_power'])

# Forecast
forecast = model.predict(n_periods=24)
```

## Configuration

Configuration parameters can be modified in `src/utils/config.py`:

- `TRAIN_SPLIT`: Proportion of data for training (default: 0.85)
- `FORECAST_HORIZON`: Number of hours to forecast ahead (default: 24)
- `LAG_FEATURES`: Lag periods for feature engineering
- `ROLLING_WINDOWS`: Rolling window sizes for feature engineering

## Model Performance

The pipeline trains and evaluates multiple models:

1. **Classical Models**: ARIMA, SARIMA, ETS, STL
2. **Machine Learning**: XGBoost, Random Forest, Linear Regression
3. **Advanced**: Prophet, LSTM, GRU

Models are evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Directional Accuracy

## Outputs

The pipeline generates:
- Preprocessed data
- EDA visualizations (time series plots, decomposition, ACF/PACF)
- Model comparison charts
- Forecast plots
- Residual analysis
- Evaluation report

All outputs are saved in the `outputs/` directory.

## Key Findings

Based on the analysis:
- **Seasonality**: Strong daily (24-hour) and weekly (168-hour) patterns
- **Stationarity**: Series requires differencing for ARIMA models
- **Best Model**: Determined by lowest RMSE on test set

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Author

Time Series Forecasting Project - Industry Ready Implementation

## License

This project is for educational and research purposes.

## Acknowledgments

- Dataset: Household Power Consumption
- Libraries: statsmodels, pmdarima, scikit-learn, xgboost, prophet, tensorflow

