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
| Break-even | off | off (see note) |
| Trail | 3.0R start, 1.0R distance | 1.5R start, 0.5R distance (same ratio to target) |
| Max holding | off | off (session-end flat bounds it) |
| Risk per trade | 8% | 1% (input capped at 5%) |
| Session | off | on, 13:00–21:00 GMT |
| Noise floors | as frozen | as frozen (see note) |
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

## Note on the first backtest (1.5 months, M5, 6 trades)

That run lost money, and it is worth being precise about what it did and did
not show.

**What it could not show:** anything about edge. Six trades is not a sample.
With a 2R target the break-even win rate is 33%, and drawing 1 win from 6 has
a ~35% probability even if the true win rate is a perfectly healthy 33%. The
profit factor of 0.03 is noise around an unmeasured quantity.

**What it did show, and what was fixed:**

1. *Trade frequency was broken.* 0.19 trades/day is not a scalper — it is the
   M30 EA with extra steps. That is a fact about the filters, not about luck.
   Three noise floors had been raised above the frozen model's validated
   values on reasoning alone (dominance 0.05→0.10, divergence 0.00→0.10,
   anchor neutral band 0.00→0.10). All three are now back at the frozen
   values. Undoing an unvalidated change is not curve fitting.
2. *Exits were asymmetric against the strategy.* All five losers ran the full
   -1R; the single winner closed at +0.16R. That is the signature of
   break-even at 1.0R against a 2.0R target: price touches 1R, the stop moves
   to entry + 0.1R, price retraces, and a would-be winner becomes a scratch —
   while losers are given the whole stop. Break-even is now off, as in the
   frozen model, and the trail is set to the same start/distance ratio to
   target that the frozen model used. The 48-bar hard holding cap is off too;
   session-end flat already bounds holding time.
3. The session window was 13:00–20:00 GMT, which fits US summer time but cuts
   the last hour of the cash session in winter. It is now 13:00–21:00.

**None of the above was chosen to make those six trades profitable.** They are
reversions to the validated model plus one structural fix. Do not read them as
a tuned configuration.

## Diagnosing trade frequency (`InpLogFilterStats`)

Rather than guessing which filter starves the EA, it now counts. For every
completed bar where a setup was pending, it records the first gate that
refused the entry, and prints the tally at the end of the run:

```
=== Elder-Ray scalper filter statistics (PERIOD_M5 / PERIOD_M30 anchor) ===
Setups confirmed: 214 bull, 198 bear. Entries opened: 31.
Setup-bar decisions: 1620
  anchor trend disagreed              742  ( 45.8%)
  dominance                           410  ( 25.3%)
  outside session                     338  ( 20.9%)
  ...
```

Read it as: a gate holding a large share is the one to question first, and a
gate sitting at 0% can be ruled out entirely. This is the number to bring to
the next tuning conversation — it is far more informative than a P&L curve
built on a handful of trades.

## Before you trust a backtest

1. Model **Every tick based on real ticks**. M1 OHLC modelling is meaningless at
   this timeframe.
2. Set the tester's spread to **Current** or a realistic fixed value, and enter
   your actual commission in the symbol settings. On a scalp the round-trip cost
   is a double-digit percentage of R; a zero-cost backtest will look good and be
   wrong.
3. Check the trade list for entries at times your broker's NAS100 is illiquid.
4. Compare M5 against M2 on the same period before committing to M2.
5. **Run at least 6–12 months.** Aim for 100+ trades before reading the P&L as
   evidence of anything. At the current rate 1.5 months cannot produce that;
   check the filter statistics first to confirm the trade rate is sane, then
   extend the window.
6. Optimise nothing until the trade count is adequate. Fitting parameters to a
   6-trade sample produces a configuration that describes those six trades and
   predicts nothing.

## Honest note on M2

On NAS100 the round-trip cost is roughly 10–15% of R on M5 with a 2×ATR stop,
but 20–30% on M2. The strategy structure is timeframe-agnostic, so it will
generate signals on M2 — the question is whether the edge survives the cost.
Start on M5, and only move to M2 on a raw-spread/ECN account after the M5
results hold up.
