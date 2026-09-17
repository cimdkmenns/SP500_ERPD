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

## Note on the second backtest (8 months, M5, 26 trades)

The exit fix worked. Profit factor moved 0.03 → 0.58, the win rate is a
normal-looking 38.5%, and winners now average about the same size as losers
instead of scratching at +0.16R. But the run still lost money, for a reason
that is arithmetic rather than bad luck.

**The 2.0R target was the mistake.** Average win 0.86R against average loss
0.92R is a payoff ratio of 0.93, which needs a **51.7% win rate** to break
even. The largest winner in 8 months was 1.94R — exactly the target, capped.
The frozen model does not work by winning often; it wins ~35-40% of the time
and needs those winners to pay **4x** a loser. Halving the target to 2.0R for
"scalping realism" removed the tail the entire edge depends on.

The target is therefore back at the frozen 4.0R, with the trail back at 3.0R
start / 1.0R distance. The stop stays at the tighter scalping 2.0 × ATR, so a
4R target is 8 × ATR(M5) — on NAS100 roughly 120–160 index points, which is a
reachable cash-session move rather than a swing-sized one.

This is a genuine trade-off, not a free win: a more distant target will lower
the hit rate. Break-even is 33.3% at a 2:1 payoff and 25% at 3:1, against
38.5% observed at a 1:1 payoff. Whether the hit rate holds up far enough is
the thing the next run has to answer.

**26 trades still cannot condemn the strategy.** P(≤10 wins of 26 | true rate
= break-even) = 0.12. That is not significant at 95%.

## The frequency problem is still unsolved

0.15 trades/day over 8 months is a swing rate, not a scalping rate, and it is
the thing blocking every other question — you cannot measure a 38% win rate
properly at 26 trades.

**The prime suspect is the session filter.** The frozen model ran 24 hours a
day. The scalper gates entries to 13:00–21:00 GMT, which is 8 hours of 24 and
should on its own cut trade count by roughly two thirds. That is a deliberate
cost-control trade, and it is now competing directly with sample size.

Do not guess at this — the EA counts it. Run once and read the two tables it
prints at the end, then sweep this grid in the optimizer, judging on
**expectancy per trade**, not total net profit:

| Input | Values to test | Why |
| --- | --- | --- |
| `InpUseEntrySession` | true / false | Largest single frequency lever |
| `InpRequireEntryDominance` | true / false | Rarely true within 4 bars of a pivot |
| `InpEntrySetupExpiryBars` | 4 / 8 / 16 | 4 bars is 20 min here vs 2 h in the M30 model |
| `InpAnchorTimeframe` | M15 / M30 | A faster anchor agrees more often |

Be prepared for one honest outcome: the filter statistics may show that an
Elder-Ray divergence confirmed by a higher-timeframe trend is simply a **rare
structure on M5**, and that relaxing the gates enough to get scalping
frequency also removes what makes the setup work. If that is what the numbers
say, the conclusion is that this strategy's edge lives at M30 and the right
move is to trade it there rather than to force it onto a 5-minute chart.

## Exit-route reporting

Alongside the filter statistics the EA now attributes every closing deal to
how the trade ended and converts it to R:

```
--- exit routes (average R by how the trade ended) ---
  stop loss hit                         16 trades  avg  -1.00R  total -16.00R
  take profit hit                        4 trades  avg  +4.00R  total +16.00R
  trailing stop                          5 trades  avg  +1.85R  total  +9.25R
  session end                            6 trades  avg  +0.20R  total  +1.20R
  ALL                                   31 trades  avg  +0.34R  total +10.45R
```

A route with a large count and a small positive average is cutting winners
before the target can pay for the losers — that is precisely how the 2.0R
target and the old break-even rule were losing money invisibly.

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
