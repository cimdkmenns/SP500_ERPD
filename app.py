# app.py — Elder-Ray Power Dominance (Multi-Index + Combined Power Chart)
# ----------------------------------------------------------------------
import time
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --------------------- Page & Sidebar ---------------------
st.set_page_config(
    page_title="Elder-Ray Power Dominance — Multi-Index",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("⚙️ Settings")

DEFAULT_TICKERS = [
    "SPY", "^GSPC", "ES=F",
    "QQQ", "^NDX", "NQ=F",
    "DIA", "^DJI", "YM=F",
    "IWM", "RTY=F",
]

tickers = st.sidebar.multiselect(
    "Select indices / tickers",
    options=sorted(DEFAULT_TICKERS),
    default=["SPY", "QQQ"],
)

interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1)
ema_period = st.sidebar.number_input("EMA period", min_value=2, max_value=200, value=13, step=1)
bars_to_show = st.sidebar.number_input("Bars to display", min_value=50, max_value=5000, value=600, step=50)

refresh_secs = st.sidebar.number_input("Auto-refresh (seconds, 0=off)", min_value=0, max_value=300, value=15, step=5)
if refresh_secs > 0:
    st_autorefresh(interval=refresh_secs * 1000, key="autorefresh")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For intraday intervals (1–15m) use liquid tickers like SPY/QQQ/ES=F/NQ=F.")

# --------------------- Helpers ---------------------
def interval_to_period(iv: str) -> str:
    # slightly longer periods for slower intervals
    return {
        "1m": "1d",
        "5m": "5d",
        "15m": "5d",
        "30m": "30d",
        "1h": "60d",
        "1d": "1y",
    }.get(iv, "5d")


def _flatten_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten yfinance MultiIndex columns and map to canonical single-level:
    Datetime, Open, High, Low, Close, Adj Close, Volume
    Works whether columns look like 'Close', ('Close','SPY'), 'Close_SPY', 'SPY_Close', etc.
    """
    # Reset index so we have a Datetime column
    df = df.reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if x is not None and str(x) != ""])
            for tup in df.columns.to_list()
        ]

    # Standardize datetime column name
    if "Datetime" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
        else:
            df.insert(0, "Datetime", pd.to_datetime(df.index))

    # Build a case-insensitive lookup for OHLCV
    cols = list(df.columns)
    norm = {c: c.lower().replace(" ", "") for c in cols}

    def _find(name):
        target = name.lower().replace(" ", "")
        # exact
        for c in cols:
            if norm[c] == target:
                return c
        # suffix/prefix patterns like Close_SPY / SPY_Close
        for c in cols:
            if norm[c].endswith("_" + target) or norm[c].startswith(target + "_"):
                return c
        return None

    mapping = {}
    for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        c = _find(field)
        if c is not None:
            mapping[field] = c

    # Require core OHLC
    for req in ["Open", "High", "Low", "Close"]:
        if req not in mapping:
            raise ValueError(f"Missing column {req}")

    out = pd.DataFrame()
    out["Datetime"] = pd.to_datetime(df["Datetime"])
    out["Open"] = pd.to_numeric(df[mapping["Open"]], errors="coerce")
    out["High"] = pd.to_numeric(df[mapping["High"]], errors="coerce")
    out["Low"] = pd.to_numeric(df[mapping["Low"]], errors="coerce")
    out["Close"] = pd.to_numeric(df[mapping["Close"]], errors="coerce")

    if "Adj Close" in mapping:
        out["Adj Close"] = pd.to_numeric(df[mapping["Adj Close"]], errors="coerce")
    if "Volume" in mapping:
        out["Volume"] = pd.to_numeric(df[mapping["Volume"]], errors="coerce")

    return out.dropna(subset=["Close"])


@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV for one ticker and normalize columns to simple OHLCV.
    This avoids the 'multiple columns to single column' error.
    """
    period = interval_to_period(interval)
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise ValueError("No data returned")
    df = _flatten_and_map_columns(raw)
    keep = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    return df[keep].copy()


def compute_elder_ray(df: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    out = df.copy()
    out["EMA"] = out["Close"].ewm(span=ema_period, adjust=False).mean()
    out["BullPower"] = out["High"] - out["EMA"]
    out["BearPower"] = out["Low"] - out["EMA"]
    out["Dominance"] = out["BullPower"] + out["BearPower"]
    out["ShortEntry"] = (out["BearPower"].shift(1) > 0) & (out["BearPower"] <= 0)
    out["ExitShort"]  = (out["Dominance"].shift(1) < 0) & (out["Dominance"] >= 0)
    return out


# Plot helpers
def plot_price(view: pd.DataFrame, ema_period: int):
    fig = plt.figure(figsize=(11.0, 4.0))
    plt.plot(view["Datetime"], view["Close"], label="Close")
    plt.plot(view["Datetime"], view["EMA"], label=f"EMA {ema_period}")
    se = view[view["ShortEntry"]]
    ex = view[view["ExitShort"]]
    if not se.empty:
        plt.scatter(se["Datetime"], se["Close"], s=28, label="Short Entry (Bear +→–)")
    if not ex.empty:
        plt.scatter(ex["Datetime"], ex["Close"], s=28, marker="x", label="Exit (Dom –→+)")
    plt.title("Price & EMA with Elder-Ray Short/Exit Signals")
    plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


def plot_power_combo(view: pd.DataFrame):
    """Combined BullPower + BearPower + Dominance."""
    fig = plt.figure(figsize=(11.0, 4.2))
    plt.plot(view["Datetime"], view["BullPower"], label="Bull Power")
    plt.plot(view["Datetime"], view["BearPower"], label="Bear Power")
    plt.plot(view["Datetime"], view["Dominance"], label="Dominance")
    plt.axhline(0, linewidth=1)
    ex = view[view["ExitShort"]]
    if not ex.empty:
        plt.scatter(ex["Datetime"], ex["Dominance"], s=28, marker="x", label="Dom –→+")
    plt.title("Elder-Ray: Bull, Bear & Dominance")
    plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


# --------------------- Title ---------------------
st.title("📈 Elder-Ray Power Dominance — Multi-Index")
st.caption(
    "Pick one or more indices in the sidebar. The overview aggregates latest Elder-Ray readings for each. "
    "Click into per-ticker tabs to see charts (Price+EMA, combined Bull/Bear/Dominance). Data by Yahoo Finance."
)

# --------------------- Main ---------------------
if not tickers:
    st.info("Select at least one ticker in the sidebar to begin.")
    st.stop()

overview_rows = []
details = {}
errors = []

for tk in tickers:
    try:
        raw = fetch_data(tk, interval)
        full = compute_elder_ray(raw, ema_period)
        view = full.tail(int(bars_to_show)).copy()
        details[tk] = view

        last = view.iloc[-1]
        overview_rows.append({
            "Ticker": tk,
            "Last Time": str(last["Datetime"]),
            "Last Price": float(last["Close"]),
            "BullPower": float(last["BullPower"]),
            "BearPower": float(last["BearPower"]),
            "Dominance": float(last["Dominance"]),
            "Short Entry?": bool(last["ShortEntry"]),
            "Exit Short?": bool(last["ExitShort"]),
        })

    except Exception as e:
        errors.append(f"{tk}: {e}")

# Overview table
if overview_rows:
    ov = pd.DataFrame(overview_rows)
    ov = ov[["Ticker", "Last Time", "Last Price", "BullPower", "BearPower", "Dominance", "Short Entry?", "Exit Short?"]]
    for c in ["Last Price", "BullPower", "BearPower", "Dominance"]:
        ov[c] = ov[c].map(lambda x: f"{x:.2f}")
    st.subheader("Overview")
    st.dataframe(ov, use_container_width=True)

if errors:
    st.warning("Some symbols failed to load:\n\n- " + "\n- ".join(errors))

# Per-ticker detail tabs
if details:
    st.subheader("Details")
    tabs = st.tabs([f"🔎 {tk}" for tk in details.keys()])
    for (tk, tab) in zip(details.keys(), tabs):
        with tab:
            view = details[tk]
            last = view.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last Price", f"{last['Close']:.2f}")
            m2.metric("Dominance", f"{last['Dominance']:.2f}")
            m3.metric("Bear Power", f"{last['BearPower']:.2f}")
            m4.metric("Bull Power", f"{last['BullPower']:.2f}")

            c1, c2 = st.columns([2, 1])
            with c1:
                fig = plot_price(view, ema_period);   st.pyplot(fig);  plt.close(fig)
                fig = plot_power_combo(view);         st.pyplot(fig);  plt.close(fig)
            with c2:
                st.write(" ")  # spacer / keep layout clean

            with st.expander("Show latest rows"):
                st.dataframe(view.tail(50), use_container_width=True)

            st.download_button(
                f"Download computed dataset (CSV) — {tk}",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name=f"elder_ray_{tk.replace('^','')}_{interval}.csv",
                mime="text/csv",
                use_container_width=True,
            )
