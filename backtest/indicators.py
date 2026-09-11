"""Indicators matching MetaTrader 5 built-ins.

Every function returns an array the same length as its input, with NaN in the
warm-up region. Wilder smoothing (SMMA) is used wherever MT5 uses it: ATR, RSI
and ADX. Bollinger bands use the population standard deviation, as MT5 does.
"""

import numpy as np


def sma(x, period):
    out = np.full(len(x), np.nan)
    if len(x) < period:
        return out
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    out[period - 1:] = (c[period:] - c[:-period]) / period
    return out


def ema(x, period):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < period:
        return out
    k = 2.0 / (period + 1.0)
    out[period - 1] = x[:period].mean()
    for i in range(period, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1.0 - k)
    return out


def _wilder(x, period, seed_index):
    """Wilder's smoothed moving average, seeded with a simple mean."""
    out = np.full(len(x), np.nan)
    if len(x) <= seed_index + period:
        return out
    first = seed_index + period
    out[first - 1] = np.nanmean(x[seed_index:first])
    for i in range(first, len(x)):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def true_range(high, low, close):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    tr = np.empty(len(high))
    tr[0] = high[0] - low[0]
    prev = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)))
    return tr


def atr(high, low, close, period):
    return _wilder(true_range(high, low, close), period, seed_index=1)


def rsi(close, period):
    close = np.asarray(close, dtype=float)
    d = np.diff(close, prepend=close[0])
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag = _wilder(gain, period, seed_index=1)
    al = _wilder(loss, period, seed_index=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(al > 0, ag / al, np.inf)
        out = 100.0 - 100.0 / (1.0 + rs)
    out[np.isnan(ag) | np.isnan(al)] = np.nan
    out[(al == 0) & (ag == 0)] = 50.0
    return out


def adx(high, low, close, period):
    """Wilder's ADX. Returns the main ADX line only."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    up = np.zeros(n)
    dn = np.zeros(n)
    up[1:] = high[1:] - high[:-1]
    dn[1:] = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr_s = _wilder(true_range(high, low, close), period, seed_index=1)
    p_s = _wilder(plus_dm, period, seed_index=1)
    m_s = _wilder(minus_dm, period, seed_index=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * p_s / tr_s
        mdi = 100.0 * m_s / tr_s
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    dx[~np.isfinite(dx)] = np.nan

    out = np.full(n, np.nan)
    valid = np.where(~np.isnan(dx))[0]
    if len(valid) < period:
        return out
    start = valid[0] + period - 1
    out[start] = np.nanmean(dx[valid[0]:start + 1])
    for i in range(start + 1, n):
        if np.isnan(dx[i]):
            out[i] = out[i - 1]
        else:
            out[i] = (out[i - 1] * (period - 1) + dx[i]) / period
    return out


def bollinger(close, period, deviation):
    close = np.asarray(close, dtype=float)
    mid = sma(close, period)
    n = len(close)
    sd = np.full(n, np.nan)
    if n >= period:
        c1 = np.cumsum(np.insert(close, 0, 0.0))
        c2 = np.cumsum(np.insert(close * close, 0, 0.0))
        s = c1[period:] - c1[:-period]
        s2 = c2[period:] - c2[:-period]
        var = np.maximum(s2 / period - (s / period) ** 2, 0.0)
        sd[period - 1:] = np.sqrt(var)
    return mid, mid + deviation * sd, mid - deviation * sd
