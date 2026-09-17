# Handoff prompt — Elder-Ray scalping EA, MT5 optimisation

Copy everything below the line into a new Claude Cowork session, and attach
`ElderRay_Scalper_M2_M5_EA.mq5`.

---

I need help optimising an MT5 Expert Advisor. You can drive MT5 on this Mac
directly, so please run the backtests yourself rather than handing me steps.

## What the EA is

`ElderRay_Scalper_M2_M5_EA.mq5` (attached) is a retrofit of my own validated
strategy — a "frozen" Elder-Ray divergence model (v6.31 strategy / v6.40
tester EA) that I developed and validated on **M30 charts with an H4 trend
anchor**, trading NAS100. The retrofit runs the same model on scalping
timeframes (M1–M15, targeting M2/M5).

The strategy: Elder-Ray Bull/Bear Power against an EMA(13); a divergence
between two confirmed swing pivots (price makes a new extreme, the matching
power does not) creates a setup; the setup fires only when a higher
"anchor" timeframe trend agrees, dominance confirms on the signal candle, and
the setup has not expired or been invalidated by a close back through the
pivot. Stops are ATR- or pivot-based, targets are R multiples, sizing is
risk-based with partial profit reinvestment, a drawdown throttle, and reduced
risk on weak shorts.

**All 59 of the frozen model's strategy parameters in this file are
byte-identical to my original EA.** The only things the retrofit changes are
which timeframes are read, plus additive diagnostics and cost guards.

## The single most important rule

**Do not change strategy parameters on reasoning alone.** Earlier in this
project a previous assistant "improved" the defaults for scalping — raised
the divergence/dominance noise floors, cut the target from 4R to 2R, added
break-even at 1R, tightened stops. Every one of those made results worse and
had to be reverted. A 2R target in particular destroyed the edge: the model
wins only ~35-40% of the time and needs 4R winners to pay for 1R losers.

Treat the frozen values as the baseline. Change a strategy parameter only
when a measurement justifies it, and tell me explicitly when you have
deviated from a frozen value and why.

## Environment

- macOS, MT5 running under Wine. The MT5 data folder is hard to reach;
  the EA prints `TerminalInfoString(TERMINAL_DATA_PATH)` as its first Journal
  line if you need it.
- **MT5 caches the last-used inputs per EA name and recompiling does NOT
  reset them to the compiled defaults.** This has already corrupted one run's
  results. Always confirm the inputs before trusting a result.
  Quick check: if the exit-route table shows `session end` or
  `maximum holding period` while `InpUseEntrySession=false` and
  `InpMaxHoldingBars=0`, you are not running the defaults.
- Symbol NAS100, chart M5, model **Every tick based on real ticks**, history
  quality 99%, ~50,372 M5 bars ≈ 8 months. Account $1,000 (GBP).
- Everything (EA, presets, a macOS/Wine installer script, full README) is on
  GitHub: `cimdkmenns/SP500_ERPD`, branch `claude/tender-meitner-rbyzx1`,
  folder `mql5/`.

## Diagnostics already built into the EA

With `InpLogFilterStats=true` the EA prints four things at the end of every
run. Use these instead of guessing:

1. **Setup funnel** — setups confirmed (bull/bear), entries opened, expired
   unused, invalidated by price.
2. **Gate tally** — for every bar with a live pending setup, which gate first
   refused the entry. Covers only bars inside a live entry window.
3. **Anchor survey** — at each live setup-bar it asks M15, M30, H1 and H4
   whether each *would* have supported the trade. One run answers the anchor
   question for all four; no sweep needed. It only counts, it never affects a
   trading decision.
4. **Exit routes** — every closing deal attributed to how it ended (stop,
   target, trail, session end, potential exit, anchor reversal) and reported
   as an average R multiple, keyed to `DEAL_POSITION_ID`.

A useful cross-check: if `passed all gates` noticeably exceeds
`Entries opened`, trades are clearing every filter and then being dropped at
sizing because the risk budget is below the broker's minimum lot. The EA
never rounds up to the minimum; it skips the trade.

## What has been established

Three runs, all M5, 8 months, real ticks:

| Run | Config | Trades | PF | Net |
| --- | --- | --- | --- | --- |
| 1 | M30 anchor, altered params | 26 | 0.58 | −47 |
| 2 | H1 anchor, stale cached inputs | 35 | 1.27 | +30 |
| 3 | H1 anchor, **all frozen values** | 91 | 0.86 | −256 |

Run 2's inputs were not what they were believed to be — treat it as void.

**Finding 1 — the anchor timeframe ratio was the cause of low trade count.**
The frozen model reads an anchor 8x its trade timeframe (M30/H4). That ratio
keeps a trade-timeframe swing too small to flip the anchor at the moment a
divergence forms. M5 anchored to M30 is only 1:6, so an M5 swing low reads as
an M30 downtrend and refuses the bull setup it just created — the filter
fought itself. The gate tally measured it: the anchor refused **91%** of live
setup-bars. Setups were never scarce (1,257 over 8 months, ~7.2/day).

Anchor survey support rates on M5:

| Anchor | Support | Ratio |
| --- | --- | --- |
| M15 | 8.3% | 1:3 |
| M30 | 9.1% | 1:6 |
| H1 | 18.2% | 1:12 |
| H4 | 39.6% | 1:48 |

`InpAnchorTimeframe=Auto` currently selects H1 for M5. H4 is untested.

**Finding 2 — the short side is what loses.** Splitting run 3 by direction
(shorts are half-sized by `InpShortRiskMultiplier=0.5`; this reconstruction
reproduces the reported net to the cent):

| | Trades | Win rate | Net | PF |
| --- | --- | --- | --- | --- |
| Longs | 31 | 38.7% | **+457** | **1.60** |
| Shorts | 60 | 11.7% | **−713** | **0.33** |
| Combined | 91 | 20.9% | −256 | 0.86 |

Blended payoff 3.26 → break-even is a 23.5% win rate. Against that null,
shorts test p=0.017 and longs p=0.042. Two tests on one dataset, so the short
result is the solid one and the long result is borderline.

This extends the frozen model's own logic rather than contradicting it: that
model already halves short risk, trails only shorts (`InpProtectLongs=false`),
and cuts only losing shorts on an anchor reversal (`InpCutLosingLongs=false`).
Mechanism: the log counted 754 bear setups against 503 bull — in an uptrend,
higher highs on fading momentum print constantly, so the strategy generates
most of its signals against the drift.

**Finding 3 — risk.** Run 3 contained **20 consecutive losses** and a 54% max
drawdown. `InpRiskPercent` is the frozen model's 8%, validated at M30
frequency. Twenty straight 1R losses leaves 19% of capital at 8% risk, 67% at
2%, 82% at 1%.

## What I want you to do

In roughly this order, running each yourself and reporting the Journal
diagnostics alongside the summary:

1. **Long-only.** `InpAllowShortEntries=false`, `InpRiskPercent=2`. Everything
   else frozen. Expect ~31 trades, PF near 1.6, far smaller drawdown.
2. **H4 anchor.** Repeat with `InpAnchorTimeframe=ER_ANCHOR_H4`, which the
   survey says supports 2.2x more setup-bars than H1.
3. **Regime robustness — this is the one that decides it.** Eight months of
   NAS100 in an uptrend punishes shorts no matter what the strategy does.
   Re-run the best configuration over a window containing a genuine NAS100
   correction. If long-only still works there it is a property of the
   strategy; if it collapses it was a property of the period.
4. **M2 with an M15 anchor** — 1:7.5, the closest reproduction of the frozen
   model's geometry of any configuration, though it pays the most spread.
5. **Cost sensitivity.** Re-run the winner with commission and a realistic
   spread. On a scalp the round-trip cost is roughly 10-15% of R on M5 and
   20-30% on M2, so a zero-cost backtest will flatter it.

## How to judge results

- **Expectancy per trade, not total net profit.** Total profit just rewards
  whichever configuration happened to take the most trades.
- 91 trades is a modest sample and the directional subsets are smaller. State
  significance or confidence rather than reporting a profit factor as fact.
- Prefer out-of-sample or walk-forward validation over in-sample optimisation.
  If you sweep parameters, hold back a period.
- Tell me plainly if the honest conclusion is that this strategy's edge lives
  at M30 and does not survive being forced onto a 5-minute chart. That is an
  acceptable answer; a curve-fitted one is not.
