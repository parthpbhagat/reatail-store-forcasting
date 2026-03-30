# Inventory Demand Forecasting App

This is a Streamlit-based sales forecasting project that predicts future product demand using multiple machine learning and time-series methods.

## 🔧 What it does

- Loads a CSV dataset (`inventory_forecast_dataset_v2.csv` or user upload).
- Expects columns: `Product_Name`, `Category`, `Date`, `Units_Sold`.
- Creates features:
  - Lag values (1, 3, 7, 14 days)
  - Day of week, month, weekend flag, trend index
  - Encoded product and category
- Trains models:
  - Linear Regression
  - Gradient Boosting
  - Random Forest
  - XGBoost (if installed)
  - ARIMA (if installed)
  - Prophet (if installed)
- Displays model metrics (MAE, RMSE, MAPE)
- Lets user select models for forecast
- Forecasts future demand (days or months, including 3-month option)
- Generates interactive charts and color-coded result table
- Produces downloadable English summary report

## ✅ Features

- Multi-model training and comparison
- Ensemble prediction from selected models
- Forecast horizon control: days or months
- Color-coded demand position table:
  - `High Demand`
  - `Medium-High Demand`
  - `Medium Demand`
  - `Low Demand`
- 3-month prediction support (`Months` slider 1–6)

## 📁 Project files

- `app.py`: main Streamlit app with full pipeline
- `inventory_forecast_dataset_v2.csv`: default dataset
- `sale_forcasting.ipynb`: Jupyter notebook for exploration
- `models/model_metadata.json` (metadata store)
- `outputs/30_day_forecast.csv` (sample output)
- `README.md`: this file

## 🛠️ Setup

1. Install Python 3.8+.
2. In terminal:

```bash
pip install streamlit pandas numpy scikit-learn plotly
# optional:
pip install xgboost statsmodels prophet
```

3. Run app:

```bash
cd "c:\Users\BAPS\OneDrive\Desktop\sale forcasting"
streamlit run app.py
```

## 📊 Use-case examples

- Sales demand prediction for next 7, 30, or 90 days
- Forecast for next 1–6 months (3-month mode for your request)
- Compare model quality and choose best model for inventory planning
- Create a standard report for managers

## 🧾 How to use

1. Open URL shown in terminal.
2. Upload your CSV or use default.
3. Verify dataset preview and correct columns.
4. Choose product and forecast period.
5. Choose one or multiple models.
6. Click `Generate Forecast`.
7. View graphs, table, and download markdown report.

## 📌 Notes

- ARIMA and Prophet use sample product this app auto-selects.
- If data has `Unit_Price ($)` or `Revenue ($)`, app ignores them in model features.
- Ensure `Date` is parseable; missing values are forward-filled.
