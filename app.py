import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def download_data(ticker: str, period: str = "5y") -> pd.DataFrame:

def download_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    ticker = ticker.strip().upper()

    data = yf.download(
        tickers=ticker,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # Flatten yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        ticker_level = None

        for level in range(data.columns.nlevels):
            if ticker in data.columns.get_level_values(level):
                ticker_level = level
                break

        if ticker_level is not None:
            data = data.xs(
                ticker,
                axis=1,
                level=ticker_level,
                drop_level=True
            )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

    data.columns = [str(col) for col in data.columns]

    return data



def build_features(df: pd.DataFrame, predict_days_ahead: int = 1) -> pd.DataFrame:
    df = df.copy()
    df["Close_shift_1"] = df["Close"].shift(1)
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    df["Target"] = df["Close"].shift(-predict_days_ahead)
    df = df.dropna()
    return df


def train_model(df: pd.DataFrame):
    feature_cols = ["Close_shift_1", "MA_5", "MA_10", "MA_20"]
    X = df[feature_cols]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, feature_cols, df, score


def predict_next_day(model, feature_cols, df: pd.DataFrame):
    latest_row = df.iloc[-1]
    X_latest = latest_row[feature_cols].values.reshape(1, -1)
    predicted_price = model.predict(X_latest)[0]
    return float(predicted_price), float(latest_row["Close"])


pred_price, last_close = predict_next_day(
    model, feature_cols, df_used
)

        # Prepare features using the latest available stock data
        latest_features = df_raw.copy()

        latest_features["Close_shift_1"] = (
            latest_features["Close"].shift(1)
        )
        latest_features["MA_5"] = (
            latest_features["Close"].rolling(window=5).mean()
        )
        latest_features["MA_10"] = (
            latest_features["Close"].rolling(window=10).mean()
        )
        latest_features["MA_20"] = (
            latest_features["Close"].rolling(window=20).mean()
        )

        latest_features = latest_features.dropna()

        if latest_features.empty:
            st.error("Not enough recent stock data to predict.")
            return

        pred_price, last_close = predict_next_day(
            model, feature_cols, latest_features
        )

