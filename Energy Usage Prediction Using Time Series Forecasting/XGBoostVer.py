from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import seaborn as sns
from multiprocessing import Pool
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fit_model(df_chunk):
    X = df_chunk.drop(columns=["y", "ds"])
    y = df_chunk["y"]
    
    # Filter NaN, infinite, or extremely large values
    valid_indices = y.notna() & np.isfinite(y) & (y < np.finfo(np.float64).max)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    model = XGBRegressor()
    model.fit(X, y)
    return model

def main():
    start_time = time.time()
    logging.info("Starting the main function")

    # Load dataset
    dataset = fetch_ucirepo(id=235)
    X = dataset.data.features
    y = dataset.data.targets

    # Combine data
    df = pd.concat([X, y], axis=1)

    # Create date column
    df["date"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)

    # Drop unnecessary columns
    df.drop(columns=["Date", "Time"], inplace=True)
    df.set_index("date", inplace=True)

    # Fill missing values
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Convert 'Global_active_power' to numeric
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

    # Prepare dataset for model
    df_xgboost = df[["Global_active_power"]].reset_index()
    df_xgboost.columns = ["ds", "y"]

    # Add date features
    df_xgboost["hour"] = df_xgboost["ds"].dt.hour
    df_xgboost["day"] = df_xgboost["ds"].dt.day
    df_xgboost["month"] = df_xgboost["ds"].dt.month
    df_xgboost["year"] = df_xgboost["ds"].dt.year

    # Clean NaN and infinite values
    df_xgboost.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_xgboost.dropna(inplace=True)

    # Split into train and test sets
    train_size = int(len(df_xgboost) * 0.8)
    train_df = df_xgboost[:train_size]
    test_df = df_xgboost[train_size:]

    # Split train set into chunks
    num_chunks = 4
    train_chunks = np.array_split(train_df, num_chunks)

    # Train model in parallel
    logging.info("Starting model training with multiprocessing")
    with Pool(num_chunks) as pool:
        models = pool.map(fit_model, train_chunks)

    # Make predictions
    logging.info("Making predictions")
    forecasts = [model.predict(test_df.drop(columns=["ds", "y"])) for model in models]
    forecast = pd.DataFrame({"ds": test_df["ds"], "yhat": np.mean(forecasts, axis=0)})

    # Align actual and forecasted values
    common_dates = test_df.set_index("ds").index.intersection(forecast.set_index("ds").index)
    if len(common_dates) == 0:
        logging.error("No common dates found between actual and forecasted values. Check the data.")
        return

    y_true = test_df.set_index("ds").loc[common_dates, "y"]
    y_pred = forecast.set_index("ds").loc[common_dates, "yhat"]

    logging.info(f"Before align - y_true: {len(y_true)}, y_pred: {len(y_pred)}")

    y_true, y_pred = y_true.align(y_pred, join='inner')
    y_true.dropna(inplace=True)
    y_pred.dropna(inplace=True)

    if len(y_true) != len(y_pred):
        y_pred = y_pred.reindex(y_true.index, method='nearest')

    logging.info(f"After align - y_true: {len(y_true)}, y_pred: {len(y_pred)}")

    if len(y_true) != len(y_pred):
        logging.error(f"Final y_true length: {len(y_true)}, y_pred length: {len(y_pred)}")
        return
    
    # Visualize the forecast
    fig = px.line(forecast, x="ds", y="yhat", title="Electricity Consumption Forecast")
    fig.show()
    
    # Plot: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual Values", color="blue")
    plt.plot(y_pred, label="Predicted Values", color="red", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Energy Consumption")
    plt.title("Actual vs. Predicted Energy Consumption")
    plt.legend()
    plt.show()

    # Residuals Analysis
    residuals = y_true - y_pred
    logging.info(f"Residuals:\n{residuals}")

    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label="Residuals", color="purple")
    plt.axhline(y=0, color="black", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Residual Values")
    plt.title("Residual Analysis")
    plt.legend()
    plt.show()

    # Error Distribution Plot
    sns.histplot(residuals, bins=30, kde=True, color="orange")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.show()

    # Calculate Error Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Log Error Metrics
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")

    end_time = time.time()
    logging.info(f"Finished the main function in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
