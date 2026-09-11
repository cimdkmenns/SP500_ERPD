# Backtest assessment: SnapScalp_MR vs Dark Venus

Date: 2026-09-11. All numbers reproducible with the scripts in this folder.

---

## 0. What was and was not possible here

**Your live positions were never at risk.** This session runs in an isolated
Linux container with no MetaTrader installation, no broker credentials and no
network path to any trading venue. Nothing here can reach your terminal or your
account.

**The MT5 Strategy Tester could not be run.** MetaTrader 5 is a Windows
application; this container has no MT5, no Wine, and no `MetaTrader5` Python
package. The outbound proxy also blocks every commercial market-data host
(Dukascopy, HistData, Yahoo, Stooq, ECB, the lot), so real GBPUSD, EURUSD and
USDJPY history could not be downloaded either.

**What was done instead.** A faithful event-driven replica of both EAs was
built, together with a bar-replay engine that models the broker side properly:
bid/ask from a per-bar spread, stop-wins-ties intrabar fills, slippage on every
market order, commission per lot round turn, margin tracking and a stop-out.
It was run on generated tapes with a calibrated session volatility profile and
a controllable amount of genuine mean reversion.

**What that can and cannot tell you.**

| Question | Answered here? |
| --- | --- |
| Is the replica free of lookahead and accounting leaks? | Yes, and it is (section 1) |
| What does a round turn actually cost, in points? | Yes, exactly (section 2) |
| Which timeframe can carry that cost? | Yes (section 3) |
| Does the risk model work as designed? | Yes (section 5) |
| How dangerous is the Dark Venus grid? | Yes, quantitatively (section 6) |
| **Is there a real edge in real GBPUSD/EURUSD/USDJPY?** | **No. Only your own tester on real tick data can answer that.** |

Section 7 is the exact procedure to answer that last question yourself, and the
engine here will re-run against your MT5 export with one command.

---

## 1. The engine is unbiased

On a driftless random walk no strategy can have an edge, so any replay must
show a loss exactly equal to its transaction costs. Twelve seeds per pair, one
year each, v2 as shipped:

| Pair | Trades | Net | Costs paid | Gross before costs | Gross per trade | ±2 SE | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| EURUSD | 853 | -11,082 | 12,982 | 1,900 | 2.23 | 3.42 | clean |
| GBPUSD | 975 | -10,565 | 12,556 | 1,991 | 2.04 | 3.31 | clean |
| USDJPY | 909 | -11,147 | 12,783 | 1,636 | 1.80 | 3.24 | clean |

Gross edge sits inside two standard errors of zero on all three pairs. There is
no lookahead, no fill optimism and no accounting leak. Everything below can be
trusted as arithmetic, whatever you think of the tape.

---

## 2. The cost hurdle is the whole story

Measured from the same runs, at a 1.0 pip spread with 7.00 per lot round-turn
commission and 2 points of slippage per market order:

| Pair | Average lots per trade | Cost per trade | Cost per round turn |
| --- | ---: | ---: | ---: |
| EURUSD | 0.80 | $15.22 | 19.0 points |
| GBPUSD | 0.56 | $12.88 | 23.2 points |
| USDJPY | 0.82 | $14.06 | 22.5 points |

Roughly **1.9 to 2.3 pips per round turn**. Against v2's default stop of 1.6 ATR
(about 6.4 pips on M5 EURUSD) that is **25 to 30 percent of the risk on every
single trade**. With the mean-reverting target sitting closer than the stop, the
break-even win rate at the shipped M5 settings is about 62 to 65 percent.

This number does not care about your signal. It is the bar every trade has to
clear before anything else matters.

---

## 3. Timeframe is the dominant lever, and M5 was the wrong choice

Cost per round turn is fixed in points. It does not shrink when you trade
faster. Gross edge per trade does. Measured with guards off and risk held at
0.25 percent, on a tape with 45 percent of M5 variance mean-reverting:

| Signal TF | Trades (2 yr) | Win % | Gross pts/trade | Cost pts/trade | Net pts/trade |
| --- | ---: | ---: | ---: | ---: | ---: |
| M5 | 1,462 | 58.4 | 5.5 | 18.9 | **-13.4** |
| M15 | 630 | 66.8 | 6.0 | 18.8 | **-12.9** |
| M30 | 345 | 71.2 | 10.3 | 18.8 | **-8.5** |
| H1 | 114 | 70.5 | 4.6 | 18.8 | **-14.2** |

(EURUSD, retail costs. GBPUSD is the same shape, peaking at M30 with 16.0 gross
against 23.1 cost.)

Two conclusions, both robust because they follow from the cost arithmetic
rather than from the tape:

1. **M5 cannot work at retail costs.** It pays the same toll as M30 for a third
   of the edge.
2. **M30 is the sweet spot.** H1 falls back only because the trade count
   collapses; the per-trade edge there is noise.

Changing the cost model matters just as much. At a raw-spread account
(0.3 pip + $6 per lot) cost falls from 18.9 to 11.9 points on EURUSD, which is
worth more than any signal change tested.

---

## 4. An honest correction to the earlier review

The v2 improvements I shipped were aimed at the wrong lever. Adding a second
entry model roughly doubled trade count on M5, and when edge per trade is
smaller than cost per trade, **more trades means more loss**. That is visible
directly: on identical tapes v1 lost 1 to 2.4 percent a year and v2 lost 8.7 to
9.7 percent, and almost all of that gap is v2 trading three times as often at
50 percent more risk per trade.

Two things in v2 did survive contact with the evidence:

- **The band snap-back entry is necessary at M30.** With the ATR-stretch model
  alone, M30 produced 11 trades in two years, which is no strategy at all. Both
  models together give about 180 a year.
- **The scale-out earns its second commission.** At M30 it lifted profit factor
  from 0.71 to 0.90 at retail costs and from 0.97 to 1.04 at raw costs. It
  lowers the win rate but converts full-stop losses into half-stop losses.

What changed in the code as a result (version 2.10):

| Input | Was | Now | Why |
| --- | --- | --- | --- |
| `InpSignalTF` | M5 | **M30** | Section 3 |
| `InpRiskPercent` | 0.75 | **0.50** | Edge is thin; do not lever a marginal edge |
| `InpMaxTradesPerDay` | 8 | **4** | M30 averages under one trade a day |

`InpEntryModel` stays on both models and `InpPartialPct` stays at 50, on the
evidence above. Nothing else was touched.

---

## 5. The recommended configuration, guards on

M30, both entry models, scale-out at 50 percent, target at the mean, risk 0.50
percent, all guards active. Three pairs, four seeds, two years each.

| Tape | Cost model | Pair | Trades/yr | Win % | PF | Ann. return | Max DD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 15% reverting | retail | EURUSD | 77 | 63.4 | 0.63 | -4.6% | 10.1% |
| 15% reverting | retail | GBPUSD | 85 | 63.1 | 0.67 | -4.5% | 10.0% |
| 15% reverting | retail | USDJPY | 98 | 65.6 | 0.71 | -4.5% | 10.0% |
| 15% reverting | raw | USDJPY | 154 | 69.4 | 0.85 | -2.8% | 9.6% |
| 45% reverting | retail | EURUSD | 119 | 70.7 | 0.82 | -2.6% | 9.0% |
| 45% reverting | retail | GBPUSD | 142 | 71.4 | 0.92 | -1.6% | 8.0% |
| 45% reverting | retail | USDJPY | 183 | 73.0 | 0.95 | -1.4% | 7.4% |
| 45% reverting | **raw** | EURUSD | 171 | 74.4 | **1.05** | **+1.1%** | 6.0% |
| 45% reverting | **raw** | GBPUSD | 172 | 75.0 | **1.15** | **+3.2%** | 4.8% |
| 45% reverting | **raw** | USDJPY | 199 | 75.2 | **1.12** | **+2.9%** | 5.1% |

Read this as a map of what has to be true, not as a forecast:

- The strategy turns profitable **only** in the bottom three rows, which need
  both a raw-spread account **and** a genuinely strongly mean-reverting tape.
- Even then the return is 1 to 3 percent a year at a 5 to 6 percent drawdown.
  That is a real but small edge, and it is fragile to any cost increase.
- The risk model behaves exactly as designed throughout. Drawdown never exceeds
  the 10 percent hard halt, position sizing tracks the stop, and the daily loss
  guard did not need to fire once at M30 with 0.5 percent risk.

The honest summary: **the machinery is sound and the risk control works. Whether
there is enough edge to pay for it is a question about the real tape, not about
the code.**

---

## 6. Dark Venus: 43 percent of accounts destroyed inside two years

Same engine, same cost model, $10,000 account, 1:100 leverage, 50 percent
stop-out. Default configuration: Bollinger counter-trend, lot-sum grid at
coefficient 1.0 (a strict doubling), no stop loss. Seventy-two runs.

| Pair | Runs | Win % | Median return | 10th pct | Worst | Median max DD | Worst max DD | Peak lots | Accounts destroyed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | 24 | 70.5 | +62.2% | -93.8% | -99.9% | 55.3% | 99.9% | 10.26 | 9 |
| GBPUSD | 24 | 71.3 | -36.4% | -99.9% | -99.9% | 76.4% | 100.0% | 10.25 | 13 |
| USDJPY | 24 | 71.3 | +39.7% | -80.2% | -99.9% | 44.8% | 99.9% | 10.28 | 9 |

Across all 72 runs: median **+45.5%**, best **+127.6%**, worst **-99.9%**, and
the account was destroyed in **31 of 72 runs (43%) within two years**.

This is precisely the pattern that makes these systems sell. A 70 percent win
rate, a median return that looks excellent, and a smooth equity curve, because a
losing basket is not a loss in the report until it is closed. Then the grid
reaches 10.25 lots from a 0.01 start (ten doublings), margin runs out, and the
account goes to zero in one move.

GBPUSD is the worst of the three, which is what you would expect: it trends
hardest, and a counter-trend grid with no stop is a bet against trends.

**The original verdict stands and is now quantified.** Do not run Dark Venus on
funded money in this configuration.

---

## 7. What to do next, in your own MT5

The one question left is whether real majors contain enough short-horizon mean
reversion. Answer it like this.

1. **Export real bars.** In MT5: View > Symbols > select the pair > Bars > M1,
   set the date range, Export. Or right-click a chart > Save As. You want the
   standard header `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL>
   <SPREAD>` — the `<SPREAD>` column matters, it is your real cost.

2. **Re-run this engine against it**, which takes seconds and uses your real
   spreads:

   ```
   python3 backtest/run.py --strategy v2 --tf 30 --commission 7 \
       --mt5 GBPUSD=/path/GBPUSD_M1.csv \
       --mt5 EURUSD=/path/EURUSD_M1.csv \
       --mt5 USDJPY=/path/USDJPY_M1.csv

   Feed it M1 bars whatever the signal timeframe; `--tf` sets the signal
   timeframe the EA runs on, and M1 is what the engine replays for fills.
   Compare `--tf 5` against `--tf 30` on your own data to check section 3
   holds there too.
   ```

   If the reported net points per trade is negative on real data the way it is
   on the 15 percent tape, the strategy does not have an edge at your costs and
   no amount of parameter tuning will create one.

3. **Then run the MT5 Strategy Tester** as the authority. Compile
   `mql5/Experts/SnapScalp_MR_v2.mq5`, load the matching preset from
   `mql5/presets/`, and set:
   - Model: **Every tick based on real ticks**. Nothing less is meaningful.
   - Period: 2022 to date, so the 2022 GBPUSD trend and the 2024 USDJPY
     intervention days are in the sample.
   - Deposit and leverage matching your live account.
   - Commission set to your broker's actual figure, in the tester **and** in
     `InpCommissionRT`.

4. **Optimise on the first 70 percent, validate on the last 30 percent
   untouched.** Use the built-in `OnTester` score, which is net profit per unit
   of drawdown scaled by trade count. A parameter set that only works on the
   in-sample window is noise.

5. **Check the cost sensitivity before anything else.** Re-run the best
   parameter set with commission raised 50 percent. If it stops being
   profitable, it was never profitable; it was sitting on the cost assumption.

6. **Then demo for a month** before any live allocation.

### The single highest-value change

Not a parameter. **Move to the lowest-cost execution you can get.** Going from
1.0 pip + $7 to 0.3 pip + $6 cut cost per round turn from 18.9 to 11.9 points on
EURUSD, and that alone was the difference between a profit factor of 0.82 and
1.05. No signal change tested came close to that.

---

## Reproducing everything here

```
python3 backtest/final.py       # sections 1, 2, 5, 6
python3 backtest/diag.py        # section 3
python3 backtest/tune.py        # section 4 structural comparison
python3 backtest/assess.py      # broader suite incl. v1 vs v2 at M5
python3 backtest/make_sets.py   # regenerate the MT5 .set presets
```

| File | What it is |
| --- | --- |
| `indicators.py` | EMA, ATR, RSI, ADX, Bollinger, matching MT5's Wilder smoothing |
| `tape.py` | MT5 CSV loader and the generated-tape model |
| `engine.py` | Broker, fills, margin, and the three strategy replicas |
| `run.py` | CLI, metrics, MT5-export entry point |
| `final_results.json` | Raw numbers behind every table above |

**Standing caveat.** Generated tapes test machinery and cost arithmetic. They
cannot tell you an edge is real. Every conclusion above about *cost* is exact;
every conclusion about *profitability* is conditional on how much mean reversion
your real data contains.
