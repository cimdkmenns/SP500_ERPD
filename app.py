# app.py — Elder-Ray Power Dominance (Multi-Index + Overview)
# -----------------------------------------------------------
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --------------------- Page & Sidebar ---------------------
st.set_page_config(
    page_title="Elder-Ray Dominance (Multi-Index)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("⚙️ Settings")

# Common US/Global indices & ETFs (edit to taste)
DEFAULT_TICKERS = [
    "SPY", "^GSPC", "ES=F",   # S&P 500 ETF, Index, E-mini futures
    "QQQ", "^NDX", "NQ=F",    # Nasdaq 100 ETF, Index, futures
    "DIA", "^DJI", "YM=F",    # Dow
    "IWM",                    # Russell 2000 ETF
    "^FTSE",                  # FTSE 100
    "^GDAXI",                 # DAX
    "^N225",                  # Nikkei 225
]

# MULTISELECT (Option 2)
tickers = st.sidebar.multiselect(
    "Select indices / tickers",
    options=sorted(DEFAULT_TICKERS),
    default=["SPY", "QQQ", "IWM"],
)

interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
ema_period = st.sidebar.number_input("EMA period", min_value=2, max_value=200, value=13, step=1)
bars_to_show = st.sidebar.number_input("Bars to display", min_value=100, max_value=5000, value=600, step=50)

refresh_secs = st.sidebar.number_input("Auto-refresh (seconds, 0=off)", min_value=0, max_value=300, value=15, step=5)
if refresh_secs > 0:
    st_autorefresh(interval=refresh_secs * 1000, key="autorefresh")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: For intraday intervals (1–15m) use liquid tickers like SPY/QQQ/ES=F/NQ=F for more reliable updates."
)

# --------------------- Helpers ---------------------
def interval_to_period(iv: str) -> str:
    return {
        "1m": "1d",
        "5m": "5d",
        "15m": "5d",
        "1h": "60d",
        "1d": "1y",
    }.get(iv, "5d")


def _map_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure single-level OHLCV columns exist:
    Datetime, Open, High, Low, Close, Adj Close, Volume
    Works even if yfinance returned flattened multi-index names like 'Close_SPY'.
    """
    cols = list(df.columns)

    # Datetime
    if "Datetime" not in cols:
        if "Date" in cols:
            df = df.rename(columns={"Date": "Datetime"})
        else:
            # last resort: build from index
            df.insert(0, "Datetime", pd.to_datetime(df.index))

    # For each required field, try to find a column that ends with it
    wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for req in wanted:
        if req not in df.columns:
            target_lower = req.lower().replace(" ", "")
            cand = None
            for c in cols:
                c_norm = str(c).lower().replace(" ", "")
                # common patterns: 'close', 'close_spy', 'spy_close'
                if c_norm == target_lower or c_norm.endswith("_" + target_lower) or c_norm.startswith(target_lower + "_"):
                    cand = c
                    break
            if cand is not None:
                df[req] = df[cand]

    # Final check
    for r in ["Open", "High", "Low", "Close"]:
        if r not in df.columns:
            raise ValueError(f"Missing column {r}")

    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(ticker: str, interval: str) -> pd.DataFrame:
    """Download OHLCV from Yahoo, normalize to flat columns."""
    period = interval_to_period(interval)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned")

    # Reset index and flatten multi-index columns if present
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if x is not None and str(x) != ""])
            for tup in df.columns.to_list()
        ]

    # Normalize canonical column names
    df = _map_ohlc_columns(df)

    # Keep only what we need
    keep = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    return df[keep].copy()


def compute_elder_ray(df: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    out = df.copy()
    out["EMA"] = out["Close"].ewm(span=ema_period, adjust=False).mean()
    out["BullPower"] = out["High"] - out["EMA"]
    out["BearPower"] = out["Low"] - out["EMA"]
    out["Dominance"] = out["BullPower"] + out["BearPower"]
    # Example signal logic
    out["ShortEntry"] = (out["BearPower"].shift(1) > 0) & (out["BearPower"] <= 0)
    out["ExitShort"]  = (out["Dominance"].shift(1) < 0) & (out["Dominance"] >= 0)
    return out


# Plot helpers
def plot_price(view: pd.DataFrame, ema_period: int):
    fig = plt.figure(figsize=(11.0, 4.8))
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


def plot_powers(view: pd.DataFrame):
    fig = plt.figure(figsize=(11.0, 3.8))
    plt.plot(view["Datetime"], view["BullPower"], label="Bull Power")
    plt.plot(view["Datetime"], view["BearPower"], label="Bear Power")
    plt.axhline(0)
    plt.title("Elder-Ray Bull & Bear Power")
    plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


def plot_dominance(view: pd.DataFrame):
    fig = plt.figure(figsize=(11.0, 3.6))
    plt.plot(view["Datetime"], view["Dominance"], label="Dominance")
    plt.axhline(0)
    ex = view[view["ExitShort"]]
    if not ex.empty:
        plt.scatter(ex["Datetime"], ex["Dominance"], s=28, marker="x", label="Dom –→+")
    plt.title("Elder-Ray Power Dominance")
    plt.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig


# --------------------- Title ---------------------
st.title("📈 Elder-Ray Power Dominance — Multi-Index")

st.caption(
    "Pick one or more indices in the sidebar. The overview aggregates latest Elder-Ray readings for each. "
    "Click into per-ticker tabs to see charts (Price+EMA, Bull/Bear Power, Dominance). Data by Yahoo Finance."
)

# --------------------- Main ---------------------
if not tickers:
    st.info("Select at least one ticker in the sidebar to begin.")
    st.stop()

overview_rows = []
details = {}   # ticker -> computed df (trimmed)
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

# --------------------- Overview table (Option 3) ---------------------
if overview_rows:
    ov = pd.DataFrame(overview_rows)
    ov = ov[["Ticker", "Last Time", "Last Price", "BullPower", "BearPower", "Dominance", "Short Entry?", "Exit Short?"]]

    # Format numeric columns
    def _fmt_num(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return x

    show = ov.copy()
    for c in ["Last Price", "BullPower", "BearPower", "Dominance"]:
        show[c] = show[c].map(_fmt_num)

    st.subheader("Overview")
    st.dataframe(show, use_container_width=True)

if errors:
    st.warning("Some symbols failed to load:\n\n- " + "\n- ".join(errors))

# --------------------- Per-ticker detail tabs (Option 2) ---------------------
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
                fig = plot_price(view, ema_period);  st.pyplot(fig);  plt.close(fig)
                fig = plot_dominance(view);          st.pyplot(fig);  plt.close(fig)
            with c2:
                fig = plot_powers(view);             st.pyplot(fig);  plt.close(fig)

            with st.expander("Show latest rows"):
                st.dataframe(view.tail(50), use_container_width=True)

            st.download_button(
                f"Download computed dataset (CSV) — {tk}",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name=f"elder_ray_{tk.replace('^','')}_{interval}.csv",
                mime="text/csv",
                use_container_width=True,
            )
