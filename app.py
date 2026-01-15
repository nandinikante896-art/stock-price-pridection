import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def download_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    data = yf.download(ticker, period=period)
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


def main():
    st.title("Stock Price Prediction App")

    st.sidebar.header("Settings")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    period = st.sidebar.selectbox("History Period", ["1y", "2y", "5y", "10y"], index=2)
    predict_days_ahead = st.sidebar.slider(
        "Predict days ahead", min_value=1, max_value=5, value=1
    )

    if st.sidebar.button("Run Prediction"):
        with st.spinner("Downloading data..."):
            df_raw = download_data(ticker, period)

        if df_raw.empty:
            st.error("No data found. Please check the ticker symbol.")
            return

        st.subheader(f"Raw data for {ticker}")
        st.write(df_raw.tail())

        st.line_chart(df_raw["Close"])

        with st.spinner("Building features and training model..."):
            df_feat = build_features(df_raw, predict_days_ahead=predict_days_ahead)
            if df_feat.empty:
                st.error("Not enough data after feature engineering.")
                return

            model, feature_cols, df_used, score = train_model(df_feat)

        st.subheader("Model Performance")
        st.write(f"R² on test set: **{score:.4f}**")

        pred_price, last_close = predict_next_day(model, feature_cols, df_used)

        st.subheader("Prediction")
        st.write(f"Last available close price: **{last_close:.2f}**")
        st.write(
            f"Predicted close price for the next {predict_days_ahead} day(s): **{pred_price:.2f}**"
        )


if __name__ == "__main__":
    main()

