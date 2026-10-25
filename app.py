
import streamlit as st
import yfinance as yf
import pandas as pd

from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

FEATURE_COLS = [
    "Close_shift_1",
    "MA_5",
    "MA_10",
    "MA_20"
]


# --------------------------------------------------
# Download and normalize stock data
# --------------------------------------------------
def download_data(
    ticker: str,
    period: str = "5y"
) -> pd.DataFrame:

    ticker = ticker.strip().upper()

    data = yf.download(
        tickers=ticker,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # Handle yfinance MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        ticker_level = None
        ticker_value = None

        for level in range(data.columns.nlevels):
            for value in data.columns.get_level_values(level).unique():
                if str(value).upper() == ticker:
                    ticker_level = level
                    ticker_value = value
                    break

            if ticker_level is not None:
                break

        if ticker_level is not None:
            data = data.xs(
                ticker_value,
                axis=1,
                level=ticker_level,
                drop_level=True
            )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

    data.columns = [str(col) for col in data.columns]

    if "Close" not in data.columns:
        return pd.DataFrame()

    data["Close"] = pd.to_numeric(
        data["Close"], errors="coerce"
    )

    data = data.dropna(subset=["Close"])

    return data


# --------------------------------------------------
# Create input features
# --------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["Close"] = pd.to_numeric(
        result["Close"], errors="coerce"
    )

    result["Close_shift_1"] = result["Close"].shift(1)
    result["MA_5"] = result["Close"].rolling(5).mean()
    result["MA_10"] = result["Close"].rolling(10).mean()
    result["MA_20"] = result["Close"].rolling(20).mean()

    return result


# --------------------------------------------------
# Prepare training data
# --------------------------------------------------
def build_features(
    df: pd.DataFrame,
    predict_days_ahead: int = 1
) -> pd.DataFrame:

    result = add_features(df)

    result["Target"] = result["Close"].shift(
        -predict_days_ahead
    )

    result = result.dropna(
        subset=FEATURE_COLS + ["Target"]
    )

    return result


# --------------------------------------------------
# Train the Linear Regression model
# --------------------------------------------------
def train_model(df: pd.DataFrame):

    if len(df) < 30:
        raise ValueError(
            "Not enough stock data. Please choose a longer period."
        )

    X = df[FEATURE_COLS]
    y = df["Target"]

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = (
        model.score(X_test, y_test)
        if len(X_test) > 0
        else float("nan")
    )

    return model, FEATURE_COLS, score


# --------------------------------------------------
# Predict next trading day's closing price
# --------------------------------------------------
def predict_next_day(
    model,
    feature_cols,
    df: pd.DataFrame
):

    latest_row = df.iloc[[-1]]

    X_latest = latest_row[feature_cols]

    predicted_price = model.predict(X_latest)[0]
    last_close = latest_row["Close"].iloc[0]

    return float(predicted_price), float(last_close)


# --------------------------------------------------
# Main Streamlit application
# --------------------------------------------------
def main():

    st.title("📈 Stock Price Prediction")
    st.write(
        "Analyze historical stock prices and estimate "
        "the next trading day's closing price."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        ticker = st.text_input(
            "Stock ticker symbol",
            value="AAPL"
        ).strip().upper()

    with col2:
        period = st.selectbox(
            "Historical data period",
            ["1y", "2y", "5y", "10y"],
            index=2
        )

    st.caption(
        "Examples: AAPL (Apple), MSFT (Microsoft), "
        "GOOGL (Alphabet), RELIANCE.NS (Reliance Industries)."
    )

    if st.button("🔍 Run Prediction", use_container_width=True):

        if not ticker:
            st.error("Please enter a stock ticker symbol.")
            return

        with st.spinner("Downloading stock data and training model..."):

            try:
                df_raw = download_data(ticker, period)

                if df_raw.empty:
                    st.error(
                        "Stock data could not be retrieved. "
                        "Check the ticker symbol and try again."
                    )
                    return

                # Show historical closing prices
                st.subheader(f"{ticker} — Historical Closing Price")

                st.line_chart(
                    df_raw["Close"].rename("Closing Price")
                )

                # Training dataset includes the next-day target
                training_data = build_features(
                    df_raw,
                    predict_days_ahead=1
                )

                if len(training_data) < 30:
                    st.error(
                        "Not enough usable data to train the model."
                    )
                    return

                model, feature_cols, score = train_model(
                    training_data
                )

                # Prepare features for the latest available date.
                # Do not drop the latest row using Target,
                # because the next day's actual target is unknown.
                latest_features = add_features(df_raw)

                latest_features = latest_features.dropna(
                    subset=feature_cols
                )

                if latest_features.empty:
                    st.error(
                        "Not enough recent data to create prediction features."
                    )
                    return

                pred_price, last_close = predict_next_day(
                    model,
                    feature_cols,
                    latest_features
                )

                st.subheader("Prediction Result")

                metric1, metric2, metric3 = st.columns(3)

                with metric1:
                    st.metric(
                        "Latest Available Closing Price",
                        f"{last_close:,.2f}"
                    )

                with metric2:
                    st.metric(
                        "Estimated Next Trading-Day Close",
                        f"{pred_price:,.2f}",
                        delta=f"{pred_price - last_close:,.2f}"
                    )

                with metric3:
                    st.metric(
                        "Test Set R²",
                        f"{score:.4f}"
                    )

                st.info(
                    "This is an estimate from a simple Linear "
                    "Regression model, not a guaranteed market price."
                )

                st.caption(
                    f"Latest historical observation: "
                    f"{df_raw.index[-1].strftime('%Y-%m-%d')}"
                )

            except Exception as error:
                st.error(
                    f"Prediction failed: {error}"
                )
                st.exception(error)


if __name__ == "__main__":
    main()
