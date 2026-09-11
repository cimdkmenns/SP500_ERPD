"""Price tapes for the backtester.

Two sources:

  load_mt5_csv()  - real bars exported from MetaTrader 5
                    (Tools > Options > Charts > max bars, then right-click a
                    chart > Save As, or the Terminal's History Center export).
                    Expected header: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE>
                    <TICKVOL> <VOL> <SPREAD>, tab separated.

  synthetic()     - a generated M1 tape with a permanent/transient price
                    decomposition, a session volatility profile and a session
                    spread profile.

The transient component is what a mean-reversion strategy can capture. Setting
transient_share = 0 gives a driftless random walk, on which any strategy must
lose exactly its transaction costs. That is the null test.

All prices in a tape are MID prices. Bid and ask are derived per bar from the
spread column, so the cost model lives in one place.
"""

import numpy as np

MINUTES_PER_DAY = 1440

# Relative volatility by broker-server hour (server time = UTC+2/+3).
# Overnight lull, London open, the New York overlap peak, the evening fade.
HOUR_VOL = np.array([
    0.35, 0.30, 0.45, 0.55, 0.50, 0.45, 0.50, 0.70,
    0.95, 1.15, 1.20, 1.05, 0.95, 1.10, 1.45, 1.55,
    1.40, 1.15, 0.90, 0.75, 0.60, 0.50, 0.40, 0.35,
])

# Spread multiplier by broker-server hour. Tightest in the liquid sessions,
# blown out across the rollover.
HOUR_SPREAD = np.array([
    3.0, 3.5, 1.8, 1.4, 1.4, 1.4, 1.3, 1.15,
    1.0, 1.0, 1.0, 1.05, 1.1, 1.0, 1.0, 1.0,
    1.0, 1.05, 1.15, 1.3, 1.5, 1.8, 2.4, 3.2,
])

PAIRS = {
    #                 digits point    contract  base_spread_pts  daily_vol
    "EURUSD": dict(digits=5, point=1e-5, contract=100_000, spread=8,  dvol=0.0050, price=1.0850, quote="USD"),
    "GBPUSD": dict(digits=5, point=1e-5, contract=100_000, spread=12, dvol=0.0065, price=1.2700, quote="USD"),
    "USDJPY": dict(digits=3, point=1e-3, contract=100_000, spread=10, dvol=0.0060, price=150.00, quote="JPY"),
}


class Tape:
    """M1 bars plus the derived signal/regime timeframes."""

    def __init__(self, symbol, t, o, h, l, c, spread_pts, spec):
        self.symbol = symbol
        self.t = t                      # int64 minutes since epoch
        self.o, self.h, self.l, self.c = o, h, l, c
        self.spread_pts = spread_pts
        self.spec = spec
        self.point = spec["point"]
        self.digits = spec["digits"]

    def __len__(self):
        return len(self.t)

    @property
    def hours(self):
        return (self.t // 60) % 24

    def resample(self, minutes):
        """Aggregate M1 to a higher timeframe.

        Returns (bucket_id_per_m1, open, high, low, close, start_index) where
        bucket_id_per_m1[i] is the index of the higher-timeframe bar that M1
        bar i belongs to.
        """
        bucket_key = self.t // minutes
        new = np.empty(len(bucket_key), dtype=bool)
        new[0] = True
        new[1:] = bucket_key[1:] != bucket_key[:-1]
        starts = np.where(new)[0]
        bid = np.cumsum(new) - 1              # bucket index per M1 bar

        n = len(starts)
        ends = np.append(starts[1:], len(self.t))
        o = self.o[starts]
        c = self.c[ends - 1]
        h = np.maximum.reduceat(self.h, starts)
        l = np.minimum.reduceat(self.l, starts)
        assert len(o) == n
        return bid, o, h, l, c, starts


def _session_mask(t_minutes):
    """Keep Monday 00:00 to Friday 23:59 server time; drop the weekend."""
    dow = ((t_minutes // MINUTES_PER_DAY) + 3) % 7      # 1970-01-01 was a Thursday
    return (dow >= 0) & (dow <= 4)


def synthetic(symbol, weeks=130, transient_share=0.0, half_life_min=45.0,
              seed=0, substeps=8):
    """Generate an M1 tape.

    transient_share  fraction of one-minute return variance that is transient
                     (mean reverting). 0.0 = pure random walk.
    half_life_min    half-life of the transient component, in minutes.
    """
    spec = PAIRS[symbol]
    rng = np.random.default_rng(seed)

    start = np.datetime64("2023-01-02T00:00", "m").astype("int64")
    raw_t = start + np.arange(weeks * 7 * MINUTES_PER_DAY, dtype=np.int64)
    raw_t = raw_t[_session_mask(raw_t)]
    n = len(raw_t)

    hours = (raw_t // 60) % 24
    vol_mult = HOUR_VOL[hours]

    # Per-minute sigma so that the daily move matches the pair's typical range.
    # Daily variance is the sum of per-minute variances, so normalise on the
    # root-mean-square of the hourly profile, not its mean.
    sigma_day = spec["dvol"]
    sigma_base = sigma_day / np.sqrt(60.0 * np.sum(HOUR_VOL ** 2))
    sigma_min = sigma_base * vol_mult

    ts = float(np.clip(transient_share, 0.0, 0.95))
    phi = 0.5 ** (1.0 / max(half_life_min, 1.0))

    # Permanent component: a random walk. Transient: AR(1) around it.
    sig_perm = sigma_min * np.sqrt(1.0 - ts)
    # For dev_t = phi*dev_{t-1} + eps the one-minute change has variance
    # sig_eps^2 * 2/(1+phi); invert that so the transient part contributes
    # exactly `ts` of the one-minute return variance.
    sig_dev = sigma_min * np.sqrt(ts * (1.0 + phi) / 2.0)

    perm = np.cumsum(rng.standard_normal(n) * sig_perm)

    dev = np.empty(n)
    eps = rng.standard_normal(n) * sig_dev
    d = 0.0
    for i in range(n):
        d = phi * d + eps[i]
        dev[i] = d

    log_close = np.log(spec["price"]) + perm + dev

    # Intrabar path: a Brownian bridge between consecutive closes, so the
    # high and low are realistic rather than just max/min of the endpoints.
    log_open = np.empty(n)
    log_open[0] = log_close[0]
    log_open[1:] = log_close[:-1]

    k = substeps
    w = rng.standard_normal((n, k)) * (sigma_min / np.sqrt(k))[:, None]
    path = np.cumsum(w, axis=1)
    ramp = np.linspace(0.0, 1.0, k)[None, :]
    path -= path[:, -1][:, None] * ramp                # pin the bridge end at 0
    path = path + log_open[:, None] + (log_close - log_open)[:, None] * ramp

    log_high = np.maximum(path.max(axis=1), np.maximum(log_open, log_close))
    log_low = np.minimum(path.min(axis=1), np.minimum(log_open, log_close))

    o = np.exp(log_open)
    c = np.exp(log_close)
    h = np.exp(log_high)
    l = np.exp(log_low)

    # Spread: session profile, a little noise, and occasional widenings.
    base = spec["spread"] * HOUR_SPREAD[hours]
    noise = rng.lognormal(mean=0.0, sigma=0.18, size=n)
    spike = np.where(rng.random(n) < 0.004, rng.uniform(2.0, 6.0, n), 1.0)
    spread = np.maximum(1.0, np.round(base * noise * spike)).astype(np.int32)

    return Tape(symbol, raw_t, o, h, l, c, spread, spec)


def load_mt5_csv(path, symbol):
    """Load M1 bars exported from MetaTrader 5.

    Accepts the standard MT5 export (tab or comma separated) with a header
    line beginning '<DATE>'. If a <SPREAD> column is present it is used;
    otherwise the pair's default spread is assumed and a warning is printed.
    """
    spec = dict(PAIRS.get(symbol.upper()[:6], PAIRS["EURUSD"]))

    rows_t, rows = [], []
    spread_col = None
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", "\t").replace(";", "\t").split("\t")
            parts = [p for p in (x.strip() for x in parts) if p != ""]
            if parts[0].upper().startswith("<DATE") or parts[0].upper() == "DATE":
                upper = [p.upper().strip("<>") for p in parts]
                spread_col = upper.index("SPREAD") if "SPREAD" in upper else None
                continue
            if len(parts) < 6:
                continue
            date = parts[0].replace("/", "-").replace(".", "-")
            time = parts[1] if ":" in parts[1] else "00:00"
            if len(time.split(":")) == 3:
                time = ":".join(time.split(":")[:2])
            try:
                tm = np.datetime64(f"{date}T{time}", "m").astype("int64")
            except Exception:
                continue
            vals = parts[2:]
            try:
                o, h, l, c = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
            except (ValueError, IndexError):
                continue
            sp = np.nan
            if spread_col is not None and len(parts) > spread_col:
                try:
                    sp = float(parts[spread_col])
                except ValueError:
                    sp = np.nan
            rows_t.append(tm)
            rows.append((o, h, l, c, sp))

    if not rows:
        raise ValueError(f"No usable bars parsed from {path}")

    t = np.array(rows_t, dtype=np.int64)
    arr = np.array(rows, dtype=float)
    order = np.argsort(t)
    t, arr = t[order], arr[order]

    # Infer the point size from the quoted decimals if the symbol is unknown.
    digits = max(len(f"{v:.10f}".rstrip("0").split(".")[1]) for v in arr[:200, 3])
    if symbol.upper()[:6] not in PAIRS:
        spec["digits"] = digits
        spec["point"] = 10.0 ** (-digits)

    spread = arr[:, 4]
    if np.all(np.isnan(spread)):
        print(f"  note: no <SPREAD> column in {path}; assuming a flat "
              f"{spec['spread']} points for {symbol}")
        spread = np.full(len(t), spec["spread"], dtype=float)
    else:
        med = np.nanmedian(spread)
        spread = np.where(np.isnan(spread), med, spread)

    return Tape(symbol, t, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
                np.maximum(1, spread).astype(np.int32), spec)
