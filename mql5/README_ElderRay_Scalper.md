# Elder-Ray Scalping Retrofit (NAS100, M2/M5)

`ElderRay_Scalper_M2_M5_EA.mq5` runs the frozen H4/M30 Elder-Ray model
(v6.31 strategy / v6.40 tester EA) on a scalping chart.

**All 59 of the frozen model's strategy parameters are unchanged.** The EMA,
ATR periods, pivot geometry, divergence rules, dominance threshold, neutral
zone, expiry windows, stop mode and multiplier, 4.0R target, trailing,
break-even, protection flags, risk percentage and drawdown throttle are all
exactly as you validated them. The only thing this EA changes is **which
timeframes the model reads**, plus two additive cost guards.

## The one real change: the anchor timeframe ratio

The frozen model trades M30 and reads H4 — an anchor **8x** its trade
timeframe. That ratio is not incidental. It is what keeps a swing on the
trade timeframe too small to flip the anchor trend at the moment a divergence
forms. A bull divergence prints at a swing low; if the anchor is fast enough
to read that same low as a downtrend, the setup is refused by construction.

The first M5 build anchored to M30, a ratio of only 1:6, and the 8-month run
measured the consequence:

```
Setups confirmed: 503 bull, 754 bear (1257 total, ~7.2/day)
Entries opened: 26

of the 4415 decisions inside a live entry window:
  anchor trend disagreed   4017   91.0%
  dominance                 197    4.5%
  outside session           168    3.8%
  passed all gates           33    0.7%
```

Setups were never scarce. The anchor refused 91% of them. `InpAnchorTimeframe`
now defaults to Auto with a ratio closer to the frozen model's:

| Chart | Anchor | Ratio | Note |
| --- | --- | --- | --- |
| M1–M3 | M15 | 1:5–1:15 | M2/M15 is 1:7.5, the closest match to frozen |
| M4–M6 | H1 | 1:10–1:15 | M5/H1 is 1:12 (was M30, 1:6) |
| M10–M15 | H4 | 1:16–1:24 | |
| *frozen* | *H4 from M30* | *1:8* | |

Test M30 / H1 / H4 explicitly on M5. The evidence says 1:6 is too fast; it
does not prove 1:12 is optimal.

## Added inputs (none change the strategy)

Cost guards — genuinely needed on a fast chart, and both refused 0% of
decisions in the 8-month run:

- `InpMaxSpreadATR` (0.30) — refuse an entry when the spread exceeds this
  share of ATR.
- `InpMinStopSpreadMultiple` (5.0) — widen a stop that is tight relative to
  the spread. Never binds at the frozen 3.0 × ATR stop.
- `InpMinATRPoints` (0, off) — skip dead tape.

Off by default, available if you want them:

- `InpUseEntrySession` + `InpEntryStartHour` / `InpEntryEndHour` /
  `InpServerGMTOffsetHours` / `InpSkipMinutesAfterSessionOpen` /
  `InpCloseAtSessionEnd`. Hours are GMT, converted to the broker's server
  clock. 13–21 GMT spans the US cash session under both DST regimes. It
  refused only 3.8% of live setup-bars, so do not assume it helps.
- `InpCooldownBars`, `InpMaxTradesPerDay`, `InpDailyLossStopPercent`.

Diagnostics and execution:

- `InpLogFilterStats` (on) — the two tables below.
- `InpAllowLiveTrading` (off) — the parent EA is Strategy-Tester-only.
- `InpRiskPeakBalanceOverride` — restore the drawdown throttle's balance peak
  after a restart; it is not persisted.

## Risk warning

`InpRiskPercent` is the frozen model's **8%**, restored because it is your
validated value. Be aware that it was validated at M30 trade frequency. If a
faster chart produces several times the trades, 8% per trade compounds a
losing streak far faster than it did in the original tests. Consider what that
does to a 5-trade losing run before running it forward.

## Diagnostics

Printed at the end of every run:

```
Setups confirmed: 503 bull, 754 bear (1257 total).
  -> entries opened       26
  -> expired unused      982
  -> invalidated by price 249
Gate tally below covers only bars inside a live entry window.
  anchor trend disagreed   4017  (91.0%)
  ...
--- exit routes (average R by how the trade ended) ---
  stop loss hit              16 trades  avg  -1.00R  total -16.00R
  take profit hit             4 trades  avg  +4.00R  total +16.00R
  trailing stop               5 trades  avg  +1.85R  total  +9.25R
  ALL                        25 trades  avg  +0.37R  total  +9.25R
```

A gate holding a large share is the one to question. An exit route with a
large count and a small positive average is cutting winners before the target
can pay for the losers.

## Testing notes

1. Model **Every tick based on real ticks**. M1 OHLC modelling is meaningless
   here.
2. Set spread to Current and enter your actual commission. On a scalp the
   round-trip cost is a double-digit percentage of R.
3. Run 6–12 months and aim for 100+ trades before reading the P&L. The
   earlier 6-trade and 26-trade samples could not distinguish a working
   strategy from a broken one.
4. Sweep `InpAnchorTimeframe` first. It is the binding constraint.

## Honest note on M2

Round-trip cost is roughly 10–15% of R on M5 at a 3 × ATR stop, and 20–30% on
M2. M2/M15 reproduces the frozen model's 1:7.5 geometry most closely, so it
may well produce the cleanest signal set — but it also pays the most cost.
Start on M5, and only move to M2 on a raw-spread account.
