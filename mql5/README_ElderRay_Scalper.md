# Elder-Ray Scalping Retrofit (NAS100, M2/M5)

`ElderRay_Scalper_M2_M5_EA.mq5` is a retrofit of the frozen H4/M30 Elder-Ray
model (v6.31 strategy / v6.40 tester EA). The strategy logic is unchanged —
only the timeframe plumbing, the defaults, and the cost/frequency guards a
scalp needs are new.

## What is identical to the M30 EA

- Elder-Ray Bull/Bear Power measured against an EMA(13).
- Divergence detected between two confirmed swing pivots (price makes a new
  extreme, the corresponding power does not).
- A higher "anchor" timeframe trend gate (EMA + ATR neutral band) plus an
  optional anchor EMA slope filter.
- Dominance confirmation on the signal candle.
- Separate entry-window and exit-watch expiry; setup invalidated when a close
  breaks back through the setup pivot.
- ATR / setup-pivot / tighter-of-both stop modes, R-multiple target,
  break-even, close-based trailing, cut-losing-reversals.
- Risk-based sizing with partial profit reinvestment, a sizing-capital ceiling,
  a realized-drawdown throttle, and reduced risk on weak shorts.
- Closed-bar processing only; the anchor timeframe is always read one bar back
  so there is no look-ahead.

## What changed

| Area | M30 EA | Scalper |
| --- | --- | --- |
| Trade timeframe | M30 only | M1–M15 (M5 recommended, M2 supported) |
| Anchor timeframe | H4 fixed | `InpAnchorTimeframe`, default Auto (M15/M30/H1) |
| Stop | 3.0 × ATR | 2.0 × ATR |
| Target | 4.0R | 2.0R |
| Break-even / trail | off / 3.0R start | 1.0R / 1.5R start, 0.75R distance |
| Max holding | off | 48 bars |
| Risk per trade | 8% | 1% (input capped at 5%) |
| Session | off | on, 13:00–20:00 GMT |
| Divergence noise floor | 0.00 ATR | 0.10 ATR (also dominance 0.05 → 0.10) |
| Live trading | blocked | opt-in via `InpAllowLiveTrading` |

New scalping-only controls:

- `InpMaxSpreadATR` — refuse an entry when the spread exceeds this share of ATR.
- `InpMinStopSpreadMultiple` — widen a stop that is too tight relative to the
  spread (position size shrinks to match, risk stays constant).
- `InpMinATRPoints` — skip dead tape. Off by default because point size is
  broker-dependent on NAS100.
- `InpCooldownBars`, `InpMaxTradesPerDay`, `InpDailyLossStopPercent` — churn and
  bad-day brakes.
- `InpSkipMinutesAfterSessionOpen` — sit out the widest-spread minutes.
- `InpCloseAtSessionEnd` — flat at the end of the window, so a scalp cannot
  become an overnight index gap.
- `InpServerGMTOffsetHours` — session hours are given in GMT and converted to
  the broker's server clock, so the window keeps tracking the US cash session
  through DST changes. **Set this to your broker's server offset** (commonly 2,
  or 3 during its summer time).

## Before you trust a backtest

1. Model **Every tick based on real ticks**. M1 OHLC modelling is meaningless at
   this timeframe.
2. Set the tester's spread to **Current** or a realistic fixed value, and enter
   your actual commission in the symbol settings. On a scalp the round-trip cost
   is a double-digit percentage of R; a zero-cost backtest will look good and be
   wrong.
3. Check the trade list for entries at times your broker's NAS100 is illiquid.
4. Compare M5 against M2 on the same period before committing to M2.

## Honest note on M2

On NAS100 the round-trip cost is roughly 10–15% of R on M5 with a 2×ATR stop,
but 20–30% on M2. The strategy structure is timeframe-agnostic, so it will
generate signals on M2 — the question is whether the edge survives the cost.
Start on M5, and only move to M2 on a raw-spread/ECN account after the M5
results hold up.
