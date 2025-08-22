import time
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ========================== APP CONFIG ==========================
st.set_page_config(page_title="Elder-Ray Power Dominance — Multi-Index", layout="wide")

st.title("📈 Elder-Ray Power Dominance — Multi-Index")
st.markdown(
    "Pick one or more indices in the sidebar. "
    "The overview aggregates latest Elder-Ray readings for each. "
    "Click into per-ticker tabs to see charts (Price+EMA, Bull/Bear Power, Dominance). "
    "Data by Yahoo Finance."
)

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.header("⚙️ Settings")

    tickers = st.multiselect(
        "Select indices / tickers",
        ["SPY", "QQQ", "IWM", "DIA", "ES=F", "NQ=F", "YM=F", "RTY=F"],
        default=["SPY", "QQQ"],
    )
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)
    ema_period = st.number_input("EMA period", min_value=1, value=13, step=1)
    bars = st.number_input("Bars to display", min_value=20, value=100, step=10)
    refresh = st.number_input("Auto-refresh (seconds, 0=off)", min_value=0, value=0, step=5)

    st.caption("Tip: For intraday intervals (1–15m) use liquid tickers like SPY/QQQ/ES=F/NQ=F for more reliable updates.")


# ========================== FUNCTIONS ==========================
def get_data(ticker: str) -> pd.DataFrame:
    """Fetch OHLCV data and compute Elder-Ray values."""
    try:
        df = yf.download(ticker, interval=interval, period="5d")
        df = df.tail(bars).copy()
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Datetime"}, inplace=True)

        # Compute EMA
        df["EMA"] = df["Close"].ewm(span=ema_period, adjust=False).mean()

        # Elder-Ray BullPower / BearPower
        df["BullPower"] = df["High"] - df["EMA"]
        df["BearPower"] = df["Low"] - df["EMA"]
        df["Dominance"] = df["BullPower"] + df["BearPower"]

        # Simple exit marker when dominance crosses from negative to positive
        df["ExitShort"] = (df["Dominance"].shift(1) < 0) & (df["Dominance"] > 0)
        return df
    except Exception as e:
        st.error(f"{ticker}: {e}")
        return None


def plot_price(view: pd.DataFrame, ema_period: int):
    fig = plt.figure(figsize=(11.0, 3.5))
    plt.plot(view["Datetime"], view["Close"], label="Close")
    plt.plot(view["Datetime"], view["EMA"], label=f"EMA {ema_period}")
    plt.title("Price with EMA")
    plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


def plot_power_combo(view: pd.DataFrame):
    """Combined BullPower + BearPower + Dominance in one chart"""
    fig = plt.figure(figsize=(11.0, 4.2))
    plt.plot(view["Datetime"], view["BullPower"], label="Bull Power")
    plt.plot(view["Datetime"], view["BearPower"], label="Bear Power")
    plt.plot(view["Datetime"], view["Dominance"], label="Dominance")
    plt.axhline(0, linewidth=1, color="black")
    ex = view[view["ExitShort"]]
    if not ex.empty:
        plt.scatter(ex["Datetime"], ex["Dominance"], s=28, marker="x", label="Dom –→+")
    plt.title("Elder-Ray: Bull, Bear & Dominance")
    plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


# ========================== MAIN ==========================
if not tickers:
    st.warning("Please select at least one ticker.")
    st.stop()

latest_rows = {}
bad = []

for t in tickers:
    df = get_data(t)
    if df is None or df.empty:
        bad.append(t)
    else:
        latest_rows[t] = df.iloc[-1]

if bad:
    st.warning(f"Some symbols failed to load: {', '.join(bad)}")

if not latest_rows:
    st.error("No data available.")
    st.stop()

# ========================== OVERVIEW ==========================
st.subheader("Overview (latest bar per ticker)")

ov = (
    pd.DataFrame(latest_rows)
    .T.reset_index()
    .rename(columns={"index": "Ticker"})
    [["Ticker", "Close", "EMA", "BullPower", "BearPower", "Dominance"]]
)
st.dataframe(ov.style.format({"Close": "{:.2f}", "EMA": "{:.2f}", "BullPower": "{:.2f}", "BearPower": "{:.2f}", "Dominance": "{:.2f}"}), use_container_width=True)

# ========================== DETAILS ==========================
st.subheader("Details per Ticker")

tabs = st.tabs(list(latest_rows.keys()))

for i, t in enumerate(latest_rows.keys()):
    with tabs[i]:
        df = get_data(t)
        if df is None:
            continue
        view = df.tail(bars)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = plot_price(view, ema_period)
            st.pyplot(fig)
            plt.close(fig)

            fig = plot_power_combo(view)
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            st.markdown("**Latest values**")
            row = view.iloc[-1]
            st.metric("Close", f"{row['Close']:.2f}")
            st.metric("Bull Power", f"{row['BullPower']:.2f}")
            st.metric("Bear Power", f"{row['BearPower']:.2f}")
            st.metric("Dominance", f"{row['Dominance']:.2f}")

# ========================== AUTO REFRESH ==========================
if refresh > 0:
    time.sleep(refresh)
    st.experimental_rerun()
