# Sales Forecasting Project - Complete Code Explanation

## 📋 Project Overview
This is a comprehensive sales forecasting application built with Streamlit that predicts inventory demand using multiple machine learning models. The app handles data preprocessing, feature engineering, model training, and interactive forecasting with visualizations.

## 🔧 File Structure
- `app.py`: Main Streamlit application
- `inventory_forecast_dataset_v2.csv`: Sample dataset
- `sale_forcasting.ipynb`: Jupyter notebook for analysis
- `README.md`: Project documentation

---

## 📝 LINE-BY-LINE CODE EXPLANATION

### 1. IMPORTS SECTION (Lines 1-27)

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
```

**Explanation:**
- `streamlit as st`: Web app framework for creating interactive dashboards
- `pandas as pd`: Data manipulation and analysis library
- `numpy as np`: Numerical computing library
- `sklearn` modules: Machine learning algorithms and evaluation metrics
- `datetime.timedelta`: For date calculations in forecasting
- `plotly.express as px`: Interactive charting library
- `warnings.filterwarnings('ignore')`: Suppress warning messages for cleaner output

### 2. OPTIONAL MODEL IMPORTS (Lines 29-42)

```python
# Try to import additional models
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
```

**Explanation:**
- Uses try-except blocks to handle optional dependencies
- Sets boolean flags to track which advanced models are available
- App will work even if XGBoost, ARIMA, or Prophet are not installed

### 3. APP CONFIGURATION (Lines 44-47)

```python
st.set_page_config(page_title="Inventory Forecasting", layout="wide")
st.title("🏪 Inventory Demand Forecasting App")
st.markdown("Upload your own dataset (CSV) or use the default sales data. Required columns: **Product_Name, Category, Date, Units_Sold**.")
```

**Explanation:**
- `st.set_page_config()`: Sets browser tab title and layout
- `st.title()`: Main heading with emoji
- `st.markdown()`: Instructions for users about required data format

### 4. DATASET UPLOAD SECTION (Lines 49-65)

```python
# Dataset upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])

# Load default dataset
if uploaded_file is None:
    st.info("Loading default dataset...")
    try:
        df = pd.read_csv('inventory_forecast_dataset_v2.csv', parse_dates=['Date'])
        st.success("Default dataset loaded successfully!")
        use_default = True
    except FileNotFoundError:
        st.error("Default dataset file not found. Please upload a CSV file.")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded_file)
        use_default = False
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
```

**Explanation:**
- `st.file_uploader()`: Creates file upload widget accepting CSV files
- Conditional logic: if no file uploaded, load default dataset
- `parse_dates=['Date']`: Automatically converts Date column to datetime
- Error handling with user-friendly messages
- `st.stop()`: Halts execution if critical errors occur

### 5. FEATURE ENGINEERING FUNCTION (Lines 67-85)

```python
def create_features(group):
    """Create lag and calendar features for time series forecasting"""
    g = group.copy().sort_values('Date')
    g = g.set_index('Date').asfreq('D')
    g['Units_Sold'] = g['Units_Sold'].ffill().fillna(0)

    # Lag features
    for lag in [1, 3, 7, 14]:
        g[f'lag_{lag}'] = g['Units_Sold'].shift(lag)

    # Calendar features
    g['day_of_week'] = g.index.dayofweek
    g['month'] = g.index.month
    g['is_weekend'] = (g.index.dayofweek >= 5).astype(int)
    g['trend'] = np.arange(len(g))

    return g.reset_index()
```

**Explanation:**
- Function processes each product group separately
- `asfreq('D')`: Converts to daily frequency, fills missing dates
- `ffill()`: Forward fills missing sales data
- **Lag features**: Previous days' sales (1, 3, 7, 14 days ago)
- **Calendar features**: Day of week, month, weekend flag, trend counter
- Returns processed dataframe for each product

### 6. OUTLIER ANALYSIS SECTION (Lines 87-143)

```python
# Outlier Analysis Section
st.write("### 🔍 Outlier Analysis")

# Calculate outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Before outlier removal
outliers_before, lower_bound, upper_bound = detect_outliers_iqr(df, 'Units_Sold')
```

**Explanation:**
- **IQR Method**: Interquartile Range for outlier detection
- Q1 (25th percentile) and Q3 (75th percentile)
- IQR = Q3 - Q1
- Bounds: Q1 - 1.5*IQR and Q3 + 1.5*IQR
- Any points outside these bounds are outliers

### 7. VISUALIZATION - BEFORE/AFTER OUTLIERS (Lines 144-175)

```python
col1, col2 = st.columns(2)

with col1:
    st.write("#### 📈 Before Outlier Removal")
    fig_before = px.box(df, y='Units_Sold', title='Units Sold Distribution (Before)',
                       labels={'Units_Sold': 'Units Sold'})
    fig_before.add_hline(y=lower_bound, line_dash="dash", line_color="red",
                        annotation_text=f"Lower Bound: {lower_bound:.1f}")
    fig_before.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                        annotation_text=f"Upper Bound: {upper_bound:.1f}")
    st.plotly_chart(fig_before, use_container_width=True)
```

**Explanation:**
- Creates two-column layout for side-by-side comparison
- **Box plots**: Show data distribution and quartiles
- **Reference lines**: Red dashed lines show outlier boundaries
- Interactive Plotly charts with hover information

### 8. DATA CLEANING (Lines 176-185)

```python
# Use clean data for further processing
df = df_clean
st.success("✅ Outlier analysis completed. Using cleaned dataset for modeling.")
```

**Explanation:**
- Replaces original dataframe with cleaned version
- Success message confirms outlier removal completion

### 9. DATA VALIDATION (Lines 187-197)

```python
expected_cols = ['Product_Name', 'Category', 'Date', 'Units_Sold']
if not all(col in df.columns for col in expected_cols):
    st.error(f"Dataset must contain these columns: {expected_cols}")
    st.stop()
else:
    # Parse dates after checking columns
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    if not use_default:
        st.success("Dataset loaded successfully!")
```

**Explanation:**
- Validates presence of all required columns
- Ensures data integrity before processing
- Date parsing happens after column validation

### 10. MODEL TRAINING SECTION (Lines 199-280)

```python
with st.spinner("Processing data and training models..."):
    # Feature Engineering
    feat_df = df.groupby('Product_Name', group_keys=False).apply(create_features)
    feat_df = feat_df.dropna().reset_index(drop=True)

    le_product = LabelEncoder()
    le_category = LabelEncoder()
    feat_df['product_enc'] = le_product.fit_transform(feat_df['Product_Name'])
    feat_df['category_enc'] = le_category.fit_transform(feat_df['Category'])

    # Train/Test Split
    FEATURE_COLS = ['lag_1', 'lag_3', 'lag_7', 'lag_14', 'day_of_week', 'month', 'is_weekend', 'trend', 'product_enc', 'category_enc']
    TARGET = 'Units_Sold'

    X = feat_df[FEATURE_COLS]
    y = feat_df[TARGET]

    train_size = int(len(feat_df) * 0.85)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
```

**Explanation:**
- **Feature Engineering**: Applies create_features to each product group
- **Label Encoding**: Converts categorical product/category names to numbers
- **Train/Test Split**: 85% training, 15% testing data
- Explicit feature column list ensures consistency

### 11. INDIVIDUAL MODEL TRAINING (Lines 282-350)

```python
# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mape = np.mean(np.abs((y_test - lr_preds) / y_test)) * 100

models['Linear Regression'] = lr_model
model_metrics['Linear Regression'] = {'MAE': lr_mae, 'RMSE': lr_rmse, 'MAPE': lr_mape}
```

**Explanation:**
- **Linear Regression**: Basic ML algorithm
- **Metrics**:
  - MAE: Mean Absolute Error (average prediction error)
  - RMSE: Root Mean Square Error (penalizes large errors more)
  - MAPE: Mean Absolute Percentage Error (relative error)
- Stores model and metrics in dictionaries

### 12. ADVANCED MODELS (Lines 352-420)

```python
# 4. XGBoost (if available)
if XGB_AVAILABLE:
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mape = np.mean(np.abs((y_test - xgb_preds) / y_test)) * 100

    models['XGBoost'] = xgb_model
    model_metrics['XGBoost'] = {'MAE': xgb_mae, 'RMSE': xgb_rmse, 'MAPE': xgb_mape}
```

**Explanation:**
- **XGBoost**: Gradient boosting algorithm, often high accuracy
- Conditional training based on availability
- Same evaluation metrics as other models

### 13. TIME SERIES MODELS (Lines 422-470)

```python
# 5. ARIMA (if available) - Simple implementation for demonstration
if ARIMA_AVAILABLE:
    # Use a sample product for ARIMA demonstration
    sample_product = df['Product_Name'].value_counts().index[0]
    product_data = df[df['Product_Name'] == sample_product].set_index('Date')['Units_Sold']
    product_data = product_data.resample('D').sum().fillna(0)

    if len(product_data) > 30:
        try:
            arima_model = ARIMA(product_data, order=(5,1,0))
            arima_fit = arima_model.fit()
            arima_pred = arima_fit.forecast(steps=10)

            # Calculate metrics (simplified)
            arima_mae = np.mean(np.abs(arima_pred - product_data[-10:].values)) if len(product_data) >= 10 else 0
            arima_rmse = np.sqrt(np.mean((arima_pred - product_data[-10:].values)**2)) if len(product_data) >= 10 else 0
            arima_mape = np.mean(np.abs((arima_pred - product_data[-10:].values) / product_data[-10:].values)) * 100 if len(product_data) >= 10 else 0

            models['ARIMA'] = {'model': arima_fit, 'sample_product': sample_product}
            model_metrics['ARIMA'] = {'MAE': arima_mae, 'RMSE': arima_rmse, 'MAPE': arima_mape}
        except:
            pass
```

**Explanation:**
- **ARIMA**: AutoRegressive Integrated Moving Average for time series
- Uses only one sample product (most frequent) for demonstration
- **Order (5,1,0)**: 5 autoregressive terms, 1 differencing, 0 moving average terms
- Stored as dictionary since it works differently from ML models

### 14. MODEL COMPARISON DISPLAY (Lines 472-476)

```python
# Model Comparison Table
st.write("### 🤖 Model Comparison")
metrics_df = pd.DataFrame(model_metrics).T.round(2)
st.dataframe(metrics_df)
```

**Explanation:**
- Creates comparison table from model_metrics dictionary
- `.T` transposes for better readability (models as rows)
- `.round(2)` formats numbers to 2 decimal places

### 15. MODEL SELECTION (Lines 478-488)

```python
# Model Selection
st.write("### 🎯 Model Selection")
available_models = list(models.keys())
selected_models = st.multiselect(
    "Select models for prediction:",
    available_models,
    default=['Linear Regression', 'Gradient Boosting'] if len(available_models) >= 2 else available_models[:1],
    help="Select multiple models to combine their predictions"
)
```

**Explanation:**
- `st.multiselect()`: Allows users to choose multiple models
- Default selection: Linear Regression and Gradient Boosting if available
- Help text guides users on ensemble prediction

### 16. FORECASTING INTERFACE (Lines 490-500)

```python
selected_product = st.selectbox("Select product for forecasting:", product_list)
forecast_unit = st.radio("Forecast horizon unit:", ['Days', 'Months'], index=0)

if forecast_unit == 'Days':
    forecast_period = st.slider("Number of days to forecast:", 7, 90, 30)
    forecast_days = forecast_period
else:
    forecast_period = st.slider("Number of months to forecast:", 1, 6, 3)
    forecast_days = forecast_period * 30
```

**Explanation:**
- **Product Selection**: Dropdown with all available products
- **Time Unit Selection**: Radio buttons for Days or Months
- **Dynamic Sliders**: Different ranges based on unit selection
- **Month Conversion**: Approximates months as 30 days

### 17. FORECAST GENERATION (Lines 502-520)

```python
if st.button("Generate Forecast"):
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
```

**Explanation:**
- Button triggers forecast calculation
- `last_date + timedelta(days=1)`: Starts from day after last data point
- `pd.date_range()`: Creates sequence of future dates

### 18. PREDICTION LOGIC (Lines 522-580)

```python
prod_history = feat_df[feat_df['Product_Name'] == selected_product].sort_values('Date').tail(30)

if len(prod_history) > 0:
    last_row = prod_history.iloc[-1]
    recent_sales = prod_history['Units_Sold'].values

    # Generate predictions for each selected model
    model_predictions = {}
    daily_preds = []

    for i, fdate in enumerate(future_dates):
        # ... prediction logic for each day
```

**Explanation:**
- Gets last 30 days of historical data for selected product
- `last_row`: Most recent feature values for prediction
- `recent_sales`: Array of recent sales for lag calculations
- Loops through each future date to make predictions

### 19. INDIVIDUAL PREDICTIONS (Lines 582-620)

```python
row = {
    'lag_1': history_ext[-1] if len(history_ext) >= 1 else 0,
    'lag_3': history_ext[-3] if len(history_ext) >= 3 else 0,
    'lag_7': history_ext[-7] if len(history_ext) >= 7 else 0,
    'lag_14': history_ext[-14] if len(history_ext) >= 14 else 0,
    'day_of_week': fdate.dayofweek,
    'month': fdate.month,
    'is_weekend': int(fdate.dayofweek >= 5),
    'trend': last_row['trend'] + i + 1,
    'product_enc': last_row['product_enc'],
    'category_enc': last_row['category_enc']
}

X_row = pd.DataFrame([row])[FEATURE_COLS]

# Get predictions from all selected models
preds = [models[model].predict(X_row)[0] for model in selected_models]
ensemble_pred = np.mean(preds)  # Average of selected models
```

**Explanation:**
- **Feature Construction**: Builds feature row for each future date
- **Lag Values**: Uses recent sales history for lag features
- **Calendar Features**: Day of week, month, weekend flag for future date
- **Trend**: Continues trend from historical data
- **Ensemble**: Averages predictions from all selected models

### 20. RESULTS VISUALIZATION (Lines 622-650)

```python
# Display Results
col1, col2 = st.columns(2)

with col1:
    st.write("### 📈 Prediction Graph")
    fig = px.line(forecast_df, x='Date', y='Predicted_Demand',
                title=f"{selected_product} - {forecast_days} Day Forecast",
                markers=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write("### 📊 Prediction Data")
    st.dataframe(forecast_df)

    # Summary statistics
    st.write("**Summary Statistics:**")
    st.write(f"Minimum Demand: {min(daily_preds):.1f}")
    st.write(f"Maximum Demand: {max(daily_preds):.1f}")
    st.write(f"Average Demand: {np.mean(daily_preds):.1f}")
    st.write(f"Total Demand: {sum(daily_preds):.1f}")
```

**Explanation:**
- **Two-column layout**: Graph on left, data table on right
- **Line chart**: Shows predicted demand over time with markers
- **Summary stats**: Min, max, average, and total predicted demand

### 21. DEMAND POSITION TABLE (Lines 652-690)

```python
# Prediction Table with Color Coding
st.write("### 📋 Demand Position Table")

# Get product category
product_category = df[df['Product_Name'] == selected_product]['Category'].iloc[0] if len(df[df['Product_Name'] == selected_product]) > 0 else 'Unknown'

# Calculate demand positions
mean_demand = np.mean(daily_preds)
std_demand = np.std(daily_preds)

def get_demand_position(demand):
    if demand > mean_demand + std_demand:
        return "High Demand"
    elif demand > mean_demand:
        return "Medium-High Demand"
    elif demand > mean_demand - std_demand:
        return "Medium Demand"
    else:
        return "Low Demand"

def get_color(position):
    if position == "High Demand":
        return "background-color: #ffcccc"
    elif position == "Medium-High Demand":
        return "background-color: #ffffcc"
    elif position == "Medium Demand":
        return "background-color: #ccffcc"
    else:
        return "background-color: #cccccc"

prediction_table = pd.DataFrame({
    'Date': future_dates.strftime('%Y-%m-%d'),
    'Product_Name': selected_product,
    'Category': product_category,
    'Predicted_Demand': daily_preds,
    'Demand_Position': [get_demand_position(d) for d in daily_preds]
})

# Apply color coding
def color_rows(row):
    return [get_color(row['Demand_Position'])] * len(row)

styled_table = prediction_table.style.apply(color_rows, axis=1)
st.dataframe(styled_table)
```

**Explanation:**
- **Demand Classification**: Based on standard deviations from mean
- **Color Coding**: Red (high), Yellow (medium-high), Green (medium), Gray (low)
- **Styling**: Uses pandas styling to apply background colors
- **Business Intelligence**: Helps identify peak demand periods

### 22. SUMMARY REPORT (Lines 692-750)

```python
# Text Summary Report
st.write("### 📝 Summary Report")

report = f"""
# 📊 Sales Forecasting Report

## 📅 Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏪 Product: {selected_product}
## 📆 Forecast Period: {forecast_days} days (from {future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')})

## 🤖 Models Used: {', '.join(selected_models)}
## 🎯 Prediction Method: Ensemble Average

## 📈 Forecast Summary:
- **Total Predicted Demand**: {sum(daily_preds):.1f} units
- **Average Daily Demand**: {np.mean(daily_preds):.1f} units
- **Peak Demand**: {max(daily_preds):.1f} units
- **Lowest Demand**: {min(daily_preds):.1f} units

## 📊 Daily Breakdown:
"""

for i, (date, pred) in enumerate(zip(future_dates, daily_preds)):
    report += f"- {date.strftime('%Y-%m-%d')} ({date.strftime('%A')}): {pred:.1f} units\n"

report += f"""

## 📋 Model Performance (Test Set):
"""

for model, metrics in model_metrics.items():
    if model in selected_models:
        report += f"- **{model}**: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%\n"

report += f"""
- **Ensemble**: MAE={ensemble_mae:.2f}, RMSE={ensemble_rmse:.2f}, MAPE={ensemble_mape:.2f}%

## 💡 Recommendations:
- Monitor inventory levels closely during peak demand periods
- Consider safety stock for high-demand days
- Review forecast accuracy weekly and retrain models as needed

---
*Report generated by Sales Forecasting App*
"""

st.code(report, language='markdown')

# Download button for report
st.download_button(
    label="📥 Download Report",
    data=report,
    file_name=f"forecast_report_{selected_product}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
    mime="text/markdown"
)
```

**Explanation:**
- **Markdown Report**: Professional summary with all key information
- **Dynamic Content**: Includes actual predictions, dates, and metrics
- **Business Recommendations**: Actionable insights for inventory management
- **Download Feature**: Saves report as .md file with timestamp

---

## 🎯 KEY CONCEPTS EXPLAINED

### Time Series Forecasting
- **Lag Features**: Use past values to predict future values
- **Calendar Features**: Capture seasonal/weekend patterns
- **Trend**: Account for long-term growth/decline

### Model Ensemble
- **Multiple Models**: Combine predictions from different algorithms
- **Weighted Average**: Reduce individual model biases
- **Improved Accuracy**: Ensemble often outperforms single models

### Outlier Handling
- **IQR Method**: Statistical approach to identify extreme values
- **Data Cleaning**: Remove or handle anomalous data points
- **Distribution Analysis**: Understand data spread and normality

### Business Intelligence
- **Demand Classification**: Categorize future demand levels
- **Color Coding**: Visual cues for quick decision making
- **Inventory Planning**: Optimize stock based on predictions

This comprehensive explanation covers every aspect of the sales forecasting application, from data loading to final business recommendations.