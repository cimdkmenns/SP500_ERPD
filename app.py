import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------- Page Setup ----------
st.set_page_config(page_title="Elder-Ray Dominance (Live)",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ---------- Sidebar controls ----------
st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.selectbox("Ticker", ["SPY", "^GSPC", "ES=F"], index=0,
                              help="SPY (ETF), ^GSPC (index), ES=F (E-mini futures)")
interval = st.sidebar.selectbox("Interval",
                                ["1m","5m","15m","1h","1d"],
                                index=1)
ema_period = st.sidebar.number_input("EMA period", min_value=2, max_value=200, value=13, step=1)
bars_to_show = st.sidebar.number_input("Bars to display", min_value=100, max_value=5000, value=600, step=50)

refresh_secs = st.sidebar.number_input("Auto-refresh (seconds, 0=off)", min_value=0, max_value=300, value=15, step=5)
if refresh_secs > 0:
    st_autorefresh(interval=refresh_secs*1000, key="autorefresh")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For intraday intervals (1–15m) use SPY or ES=F for more reliable updates.")

# ---------- Helpers ----------
def interval_to_period(iv):
    return {
        "1m": "1d",
        "5m": "5d",
        "15m": "5d",
        "1h": "60d",
        "1d": "1y",
    }.get(iv, "5d")

@st.cache_data(ttl=5*60, show_spinner=False)
def fetch_data(ticker, interval):
    period = interval_to_period(interval)
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=False, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]).strip()
                      for col in df.columns.values]

    df = df.reset_index()

    # Ensure we have a datetime column
    if "Datetime" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "Datetime"})
    if "Datetime" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "Datetime"})

    # Normalize OHLC column names
    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if "open" in c and "open" not in rename_map.values():
            rename_map[col] = "Open"
        elif "high" in c and "high" not in rename_map.values():
            rename_map[col] = "High"
        elif "low" in c and "low" not in rename_map.values():
            rename_map[col] = "Low"
        elif "close" in c and "adj" not in c and "close" not in rename_map.values():
            rename_map[col] = "Close"
    df = df.rename(columns=rename_map)

    # Ensure numeric OHLC
    for c in ["Open", "High", "Low", "Close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop NaN rows
    df = df.dropna(subset=["Open", "High", "Low", "Close"], how="any")

    return df

def compute_elder_ray(df, ema_period):
    out = df.copy()
    out["EMA"] = out["Close"].ewm(span=ema_period, adjust=False).mean()
    out["BullPower"] = out["High"] - out["EMA"]
    out["BearPower"] = out["Low"] - out["EMA"]
    out["Dominance"] = out["BullPower"] + out["BearPower"]
    # Signals
    out["ShortEntry"] = (out["BearPower"].shift(1) > 0) & (out["BearPower"] <= 0)
    out["ExitShort"] = (out["Dominance"].shift(1) < 0) & (out["Dominance"] >= 0)
    return out

def plot_price(view):
    fig = plt.figure(figsize=(16,6))
    plt.plot(view["Datetime"], view["Close"], label="Close")
    plt.plot(view["Datetime"], view["EMA"], label=f"EMA {ema_period}")
    se = view[view["ShortEntry"]]
    ex = view[view["ExitShort"]]
    plt.scatter(se["Datetime"], se["Close"], s=45, label="Short Entry (Bear +→–)")
    plt.scatter(ex["Datetime"], ex["Close"], s=55, marker="x", label="Exit (Dom –→+)")
    plt.title("Price & EMA with Elder-Ray Short/Exit Signals")
    plt.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig

def plot_power_combo(view):
    fig = plt.figure(figsize=(16,6))
    plt.plot(view["Datetime"], view["BullPower"], label="Bull Power")
    plt.plot(view["Datetime"], view["BearPower"], label="Bear Power")
    plt.plot(view["Datetime"], view["Dominance"], label="Dominance", linewidth=2, linestyle="--")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Elder-Ray Bull, Bear & Dominance")
    plt.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig

# ---------- Main ----------
st.title("📈 Elder-Ray Power Dominance — Live on S&P Data")
st.caption("Runs on live market data fetched via Yahoo Finance. For true tick-level live feeds, switch to a broker API (Polygon, Alpaca, IB, etc.).")

try:
    raw = fetch_data(ticker, interval)
    if len(raw) < 10:
        st.warning("Not enough data returned. Try a different interval or ticker.")
        st.stop()
    full = compute_elder_ray(raw, ema_period)
    view = full.tail(int(bars_to_show))

    # Notify on fresh signals
    last_row = full.iloc[-1]
    if "last_signal_time" not in st.session_state:
        st.session_state["last_signal_time"] = None

    latest_time = str(last_row["Datetime"])
    new_signal = None
    if last_row["ShortEntry"]:
        new_signal = f"SHORT ENTRY detected at {latest_time}"
    elif last_row["ExitShort"]:
        new_signal = f"SHORT EXIT (Dom –→+) at {latest_time}"

    if new_signal and st.session_state["last_signal_time"] != latest_time:
        st.toast(new_signal)
        st.session_state["last_signal_time"] = latest_time

    # Layout
    c1, c2 = st.columns([5,2], gap="large")  # make chart side wider
    with c1:
        st.pyplot(plot_price(view), use_container_width=True)
        st.pyplot(plot_power_combo(view), use_container_width=True)
    with c2:
        st.metric("Last Price", f"{last_row['Close']:.2f}")
        st.metric("Dominance", f"{last_row['Dominance']:.2f}")
        st.metric("Bear Power", f"{last_row['BearPower']:.2f}")
        st.metric("Bull Power", f"{last_row['BullPower']:.2f}")

    with st.expander("Show latest rows"):
        st.dataframe(full.tail(50))

    st.download_button(
        "Download computed dataset (CSV)",
        data=full.to_csv(index=False).encode("utf-8"),
        file_name=f"elder_ray_{ticker.replace('^','')}_{interval}.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Error fetching or plotting data: {e}")
    st.stop()
