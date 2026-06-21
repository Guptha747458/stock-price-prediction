import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any, Optional

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators: SMA, EMA, RSI, Bollinger Bands, and Volume MA.
    
    Parameters:
    df (pd.DataFrame): Stock data with columns [Open, High, Low, Close, Adj Close, Volume]
    
    Returns:
    pd.DataFrame: Stock data with added technical indicators.
    """
    df = df.copy()
    
    # Moving Averages
    for window in [10, 50, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
    # Relative Strength Index (RSI - 14 days)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use exponential moving average for Wilder's smoothing
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10) # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20 days, 2 standard deviations)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-10)
    
    # Volume moving average (10 days)
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    
    return df

def create_lagged_features(df: pd.DataFrame, lag_days: int = 5) -> pd.DataFrame:
    """
    Creates lagging columns for Close price to frame prediction as tabular regression.
    
    Parameters:
    df (pd.DataFrame): Dataframe with stock price columns.
    lag_days (int): Number of previous days to include as individual features.
    
    Returns:
    pd.DataFrame: Dataframe with lagged features.
    """
    df = df.copy()
    for lag in range(1, lag_days + 1):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    return df

def prepare_regression_data(df: pd.DataFrame, lag_days: int = 5, train_split: float = 0.8) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, List[str]
]:
    """
    Prepares features and targets for standard machine learning regressors (XGBoost, Random Forest).
    
    Parameters:
    df (pd.DataFrame): Stock data with indicators.
    lag_days (int): Lagging days.
    train_split (float): Fraction of data to use for training (chronological split).
    
    Returns:
    Tuple of:
      - X_train, y_train: Training features and targets
      - X_test, y_test: Testing features and targets
      - feature_scaler: MinMaxScaler fitted on features
      - target_scaler: MinMaxScaler fitted on targets (for inverse transformation)
      - feature_names: List of column names used as features
    """
    # Create lagged features first
    df_lags = create_lagged_features(df, lag_days=lag_days)
    
    # Drop rows with NaN (which will be at the beginning due to indicators/lags)
    df_clean = df_lags.dropna().copy()  # .copy() avoids SettingWithCopyWarning on the next assignment
    
    # Target: Tomorrow's close price (Shift Close back by 1)
    df_clean['Target'] = df_clean['Close'].shift(-1)
    # Drop the last row since its Target is NaN (we don't know tomorrow's close yet in historical data)
    # NOTE: Tomorrow's prediction is computed separately in app.py using the latest row of df_lags.
    df_clean = df_clean.dropna()
    
    # Define features
    exclude_cols = ['Target']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    # Chronological Split
    split_idx = int(len(df_clean) * train_split)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    # Extract features & targets
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df['Target'].values.reshape(-1, 1)
    
    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df['Target'].values.reshape(-1, 1)
    
    # Scale features and target independently
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    X_train = feature_scaler.fit_transform(X_train_raw)
    X_test = feature_scaler.transform(X_test_raw)
    
    y_train = target_scaler.fit_transform(y_train_raw).ravel()
    y_test = target_scaler.transform(y_test_raw).ravel()
    
    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler, feature_cols

def prepare_lstm_data(df: pd.DataFrame, sequence_length: int = 10, train_split: float = 0.8) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, List[str]
]:
    """
    Prepares sequence data for LSTM model: [samples, time_steps, features].
    
    Parameters:
    df (pd.DataFrame): Stock data with indicators.
    sequence_length (int): Length of historical sequence to feed LSTM.
    train_split (float): Fraction of data for training.
    
    Returns:
    Tuple of:
      - X_train, y_train: Training sequence features and targets
      - X_test, y_test: Testing sequence features and targets
      - feature_scaler: Fitted MinMaxScaler for features
      - target_scaler: Fitted MinMaxScaler for target (Close price)
      - feature_cols: List of column names used as features
    """
    # Drop rows with NaN from technical indicators
    df_clean = df.dropna().copy()
    
    # Features & Targets
    # We predict tomorrow's Close, which is df_clean['Close'] shifted by -1
    df_clean['Target'] = df_clean['Close'].shift(-1)
    df_clean = df_clean.dropna()
    
    feature_cols = [col for col in df_clean.columns if col != 'Target']
    
    # Scale data
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(df_clean[feature_cols].values)
    scaled_targets = target_scaler.fit_transform(df_clean['Target'].values.reshape(-1, 1)).ravel()
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i : i + sequence_length])
        y.append(scaled_targets[i + sequence_length])
        
    X = np.array(X)
    y = np.array(y)
    
    # Chronological Split
    split_idx = int(len(X) * train_split)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler, feature_cols
