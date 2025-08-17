# Utility to save processed data after signal generation
import os
def save_processed_data(processed_data, base_folder="data/processed"):
    """
    Save processed DataFrames to CSV files in data/processed/<symbol>/<timeframe>.csv
    :param processed_data: Nested dict {symbol: {timeframe: DataFrame}}
    :param base_folder: Base folder to save processed data
    """
    for symbol, tf_dict in processed_data.items():
        symbol_folder = os.path.join(base_folder, symbol)
        os.makedirs(symbol_folder, exist_ok=True)
        for timeframe, df in tf_dict.items():
            filename = f"{timeframe}.csv"
            filepath = os.path.join(symbol_folder, filename)
            df.to_csv(filepath)
            print(f"Saved processed data for {symbol} {timeframe} to {filepath}")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a DataFrame (from MT5) and cleans it.
    :param data: DataFrame with MT5 columns.
    :return: Cleaned DataFrame.
    """
    # Santiy check
    assert 'close' in data.columns, "Missing close price column"
    data = data[data['close'].notnull()] # Ensure 'close' column is not null
    data = data[data['close'] != 0]
    return data

def save_raw_data_as_csv(data_dict, base_folder="data/raw"):
    """
    Save each DataFrame in data_dict as a CSV file in the format:
    <base_folder>/<symbol>/<timeframe>.csv
    """
    for symbol, tf_dict in data_dict.items():
        symbol_folder = os.path.join(base_folder, symbol)
        os.makedirs(symbol_folder, exist_ok=True)
        for timeframe, df in tf_dict.items():
            filename = f"{timeframe}.csv"
            filepath = os.path.join(symbol_folder, filename)
            df.to_csv(filepath)
            print(f"Saved {symbol} {timeframe} to {filepath}")

def run_eda(data_dict: dict, max_lag: int=20) -> None:
    """
    Perform EDA for each symbol and timeframe in the nested dictionary.
    :param data_dict: {symbol: {timeframe: DataFrame}}
    """
    for symbol, tf_dict in data_dict.items():
        for timeframe, df in tf_dict.items():
            print(f"\n--- EDA for {symbol} at timeframe {timeframe} ---")
            data = load_data(df.copy())  # Load and clean data

            # Getting returns
            data['returns'] = data['close'].pct_change()
            data.dropna(inplace=True)

            # Basic returns statistics
            mean_return = data['returns'].mean()
            std_return = data['returns'].std()
            sharp_ratio = ((mean_return / std_return))* np.sqrt(252) if std_return != 0 else 0
            skew = data['returns'].skew()
            kurt = data['returns'].kurtosis()

            print (f"""
            Basic Returns Statistics:
                Mean Return: {mean_return:.6f}
                Standard Deviation: {std_return:.6f}
                Sharpe Ratio: {sharp_ratio:.6f}
                Skewness: {skew:.6f}
            """)

            # Volatility
            data['rolling_volatility'] = data['returns'].rolling(window=230).std() * np.sqrt(252)
            avg_range = (data['high'] - data['low']).mean()
            print(f"Average Daily Range: {avg_range:.6f}")
            print(f"Average Daily Volatility (30): {data['rolling_volatility'].iloc[-1]:.6f}")

            ## Spread Analysis
            avg_spread = data['spread'].mean()
            spread_to_range = avg_spread / avg_range if avg_range > 0 else 0
            print(f"Avg spread: {avg_spread:.2f}, Spread/Range: {spread_to_range:.4f}")

            # --- Volume diagnostics ---
            avg_vol = data['tick_volume'].mean()
            print(f"Avg tick volume: {avg_vol:.2f}")

            # --- Autocorrelation diagnostics ---
            acf_vals = [data['returns'].autocorr(lag=i) for i in range(1, max_lag+1)]
            print("Return autocorrelations (first 5 lags):", np.round(acf_vals[:5], 3))

            # --- Optional: distribution diagnostics ---
            q05, q95 = data['returns'].quantile([0.05, 0.95])
            print(f"5% quantile: {q05:.4f}, 95% quantile: {q95:.4f}")

            print("-" * 50)
