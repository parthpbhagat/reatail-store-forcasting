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



st.set_page_config(page_title="Inventory Forecasting", layout="wide")



st.title("🏪 Inventory Demand Forecasting App")

st.markdown("Upload your own dataset (CSV) or use the default sales data. Required columns: **Product_Name, Category, Date, Units_Sold**.")



# Dataset uploader

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])



def load_csv_with_fallback(source, parse_dates=None):
    """
    Try reading CSV as UTF-8; if decoding fails, fall back to latin-1.
    Works for both file paths and uploaded file-like objects.
    """
    try:
        return pd.read_csv(source, parse_dates=parse_dates)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, parse_dates=parse_dates, encoding="latin1")

# Load default dataset
if uploaded_file is None:
    st.info("Loading default dataset...")
    try:
        df = load_csv_with_fallback('inventory_forecast_dataset_v2.csv', parse_dates=['Date'])
        st.success("Default dataset loaded successfully!")
        use_default = True
    except FileNotFoundError:
        st.error("Default dataset file not found. Please upload a CSV file.")
        st.stop()
else:
    try:
        df = load_csv_with_fallback(uploaded_file, parse_dates=['Date'])
        use_default = False
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()


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



# Dataset preview

st.write("### 📊 Dataset Preview")

st.dataframe(df.head())



st.success("Dataset loaded. Proceeding to feature engineering and modeling.")



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

    
    # Quick summary + stocking hints
    st.write("### 🧾 Data Summary & Stock Suggestions")
    total_rows = len(df)
    unique_products = df['Product_Name'].nunique()
    unique_categories = df['Category'].nunique()
    min_date, max_date = df['Date'].min(), df['Date'].max()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Rows", total_rows)
    col_b.metric("Products", unique_products)
    col_c.metric("Categories", unique_categories)
    col_d.metric("Date Range", f"{min_date.date()} → {max_date.date()}")

    lookback_days = st.slider(
        "Lookback window (days) for demand trend",
        7, 120, 30,
        help="Uses the most recent N days to rank products to keep in stock"
    )
    cutoff_date = max_date - pd.Timedelta(days=lookback_days)
    recent_df = df[df['Date'] >= cutoff_date]

    if recent_df.empty:
        st.warning("No data in the selected lookback window. Try a larger window.")
    else:
        demand = (recent_df
                  .groupby(['Product_Name', 'Category'])
                  .agg(total_units=('Units_Sold', 'sum'),
                       avg_daily=('Units_Sold', 'mean'),
                       last_sale=('Date', 'max'))
                  .reset_index())
        overall_mean = demand['avg_daily'].mean()
        overall_std = demand['avg_daily'].std(ddof=0) if len(demand) > 1 else 0

        def recommend(val, mean, std):
            if std == 0:
                return "Maintain"
            if val > mean + std:
                return "Stock Up"
            if val < mean - std:
                return "Reduce"
            return "Maintain"

        demand['Recommendation'] = demand['avg_daily'].apply(lambda v: recommend(v, overall_mean, overall_std))
        demand = demand.sort_values('avg_daily', ascending=False)

        st.write("Top products to keep (ranked by recent average daily demand):")
        st.dataframe(
            demand[['Product_Name', 'Category', 'avg_daily', 'total_units', 'Recommendation']]
            .rename(columns={'avg_daily': 'Avg_Daily_Units'})
            .round({'Avg_Daily_Units': 2})
        )

        csv_data = demand.to_csv(index=False)
        st.download_button("Download stocking suggestions (CSV)", data=csv_data,
                           file_name="stock_suggestions.csv", mime="text/csv")

    

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

        

        # Multiple Models Training

        models = {}

        model_metrics = {}

        

        # 1. Linear Regression

        lr_model = LinearRegression()

        lr_model.fit(X_train, y_train)

        lr_preds = lr_model.predict(X_test)

        lr_mae = mean_absolute_error(y_test, lr_preds)

        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

        lr_mape = np.mean(np.abs((y_test - lr_preds) / y_test)) * 100

        

        models['Linear Regression'] = lr_model

        model_metrics['Linear Regression'] = {'MAE': lr_mae, 'RMSE': lr_rmse, 'MAPE': lr_mape}

        

        # 2. Gradient Boosting

        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        gb_model.fit(X_train, y_train)

        gb_preds = gb_model.predict(X_test)

        gb_mae = mean_absolute_error(y_test, gb_preds)

        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))

        gb_mape = np.mean(np.abs((y_test - gb_preds) / y_test)) * 100

        

        models['Gradient Boosting'] = gb_model

        model_metrics['Gradient Boosting'] = {'MAE': gb_mae, 'RMSE': gb_rmse, 'MAPE': gb_mape}

        

        # 3. Random Forest (additional model)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        rf_model.fit(X_train, y_train)

        rf_preds = rf_model.predict(X_test)

        rf_mae = mean_absolute_error(y_test, rf_preds)

        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

        rf_mape = np.mean(np.abs((y_test - rf_preds) / y_test)) * 100

        

        models['Random Forest'] = rf_model

        model_metrics['Random Forest'] = {'MAE': rf_mae, 'RMSE': rf_rmse, 'MAPE': rf_mape}

        

        # Model Comparison Table

        st.write("### 🤖 Model Comparison")

        metrics_df = pd.DataFrame(model_metrics).T.round(2)

        st.dataframe(metrics_df)

        

        # Model Selection

        st.write("### 🎯 Model Selection")

        available_models = list(models.keys())

        selected_models = st.multiselect(

            "Select models for prediction:",

            available_models,

            default=['Linear Regression', 'Gradient Boosting'] if len(available_models) >= 2 else available_models[:1],

            help="Select multiple models to combine their predictions"

        )

        

        if not selected_models:

            st.warning("Please select at least one model.")

            st.stop()

        

        # Selected Models Summary

        st.write(f"**Selected Models:** {', '.join(selected_models)}")

        

        ensemble_models = selected_models

        ensemble_preds = np.mean([models[model].predict(X_test) for model in ensemble_models], axis=0)

        ensemble_mae = mean_absolute_error(y_test, ensemble_preds)

        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))

        ensemble_mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

        

        col1, col2, col3 = st.columns(3)

        col1.metric("Ensemble MAE", round(ensemble_mae, 2))

        col2.metric("Ensemble RMSE", round(ensemble_rmse, 2))

        col3.metric("Ensemble MAPE", f"{round(ensemble_mape, 2)}%")

        

        st.markdown("---")

        st.write("### 🔮 Future Forecasting")

        

        product_list = df['Product_Name'].unique()

        st.write(f"Forecasting all products ({len(product_list)} total)")

        forecast_months = st.slider("Forecast horizon (months):", 1, 12, 3)

        forecast_days = forecast_months * 30



        if st.button("Generate Forecast"):

            last_date = df['Date'].max()

            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

            

            all_forecasts = []



            for selected_product in product_list:

                prod_history = feat_df[feat_df['Product_Name'] == selected_product].sort_values('Date').tail(30)

                if len(prod_history) == 0:

                    continue



                last_row = prod_history.iloc[-1]

                recent_sales = prod_history['Units_Sold'].values



                daily_preds = []

                for i, fdate in enumerate(future_dates):

                    history_ext = np.concatenate([recent_sales, daily_preds])

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

                    preds = [models[model].predict(X_row)[0] for model in selected_models]

                    ensemble_pred = np.mean(preds) if preds else 0

                    daily_preds.append(max(ensemble_pred, 0))



                product_category = prod_history['Category'].iloc[-1] if 'Category' in prod_history.columns else 'Unknown'

                forecast_df = pd.DataFrame({

                    'Date': future_dates,

                    'Product_Name': selected_product,

                    'Category': product_category,

                    'Predicted_Demand': daily_preds

                })

                all_forecasts.append(forecast_df)



            if not all_forecasts:

                st.warning("Insufficient data available for forecasting.")

            else:

                all_forecasts_df = pd.concat(all_forecasts).reset_index(drop=True)
                st.write("###  Prediction Graph (All Products)")
                fig = px.line(all_forecasts_df, x='Date', y='Predicted_Demand', color='Product_Name',
                             title=f"{len(product_list)} Products - {forecast_months} Month Forecast",
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)
                st.write("###  Prediction Data")
                st.dataframe(all_forecasts_df)
                # Short summary (3-4 lines)
                total_pred = all_forecasts_df['Predicted_Demand'].sum()
                avg_daily = all_forecasts_df.groupby('Date')['Predicted_Demand'].sum().mean()
                top_product = (all_forecasts_df.groupby('Product_Name')['Predicted_Demand'].sum().sort_values(ascending=False).index[0])
                st.write("### Summary")
                st.write(f"- Horizon: {forecast_months} months (~{forecast_days} days)")
                st.write(f"- Total predicted demand: {total_pred:.1f} units")
                st.write(f"- Avg daily demand (all products): {avg_daily:.1f} units")
                st.write(f"- Top projected product: {top_product}")
                # Category-wise demand position
                cat_stats = (all_forecasts_df.groupby('Category')['Predicted_Demand'].mean().reset_index().rename(columns={'Predicted_Demand': 'Avg_Predicted_Demand'}))
                overall_mean = cat_stats['Avg_Predicted_Demand'].mean()
                overall_std = cat_stats['Avg_Predicted_Demand'].std(ddof=0) if len(cat_stats) > 1 else 0
                def classify(val, mean, std):
                    if std == 0:
                        return 'Medium Demand'
                    if val > mean + std:
                        return 'High Demand'
                    if val < mean - std:
                        return 'Low Demand'
                    return 'Medium Demand'
                cat_stats['Demand_Position'] = cat_stats['Avg_Predicted_Demand'].apply(lambda v: classify(v, overall_mean, overall_std))
                st.write("### Category-wise Demand Position")
                st.dataframe(cat_stats)

