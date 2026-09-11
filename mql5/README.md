# FX scalp EA review and improved build

Folder layout

| Path | What it is |
| --- | --- |
| `reference/DarkVenus_Reconstruction.mq5` | Reviewed as-is. Bollinger counter-trend grid with lot-sum doubling. |
| `reference/SnapScalp_MR_v1.mq5` | Reviewed as-is. ATR-stretch mean reversion, one position, hard stop. |
| `Experts/SnapScalp_MR_v2.mq5` | The improved EA. Copy this into `MQL5/Experts/` and compile in MetaEditor. |

## 1. Verdict

**SnapScalp_MR is the better EA, and it is the one that was improved.**

Dark Venus is the one that *looks* more profitable on a short backtest, and that is exactly the problem. It has no stop loss by default, and every grid level adds the sum of all open lots (a strict doubling). A losing basket is not a loss in the report until it is finally closed, so the equity curve stays smooth right up to the margin call. "Profit maximisation" is not a meaningful goal for a system whose expected value includes a ruin event. Its own header says not to run it on live funds.

SnapScalp has a hard stop on every trade, one position at a time, size derived from the stop distance, and three independent equity guards. Its profit is bounded per trade, so it is the only one of the two where profit can be pushed up without silently buying tail risk.

### Side-by-side

| Dimension | Dark Venus | SnapScalp v1 |
| --- | --- | --- |
| Stop loss | Disabled by default | Broker-level SL on every trade |
| Position sizing | Lot sum × coefficient at each grid level (doubles) | Fixed fraction of equity from SL distance |
| Max concurrent exposure | Up to 50 orders per side | One position |
| Worst case | Margin call | One risk unit (0.5% of equity by default) |
| Short-run equity curve | Very smooth, high win rate | Choppier, lower win rate |
| Long-run expectancy | Negative once the tail is priced in | Positive if the edge holds, otherwise small negative |
| Cost awareness | Commission folded into the basket target | Trade rejected unless target clears cost by 3× |
| Suitability for GBPUSD / EURUSD / USDJPY | Ranges only; trends on GBPUSD are fatal | Works on any of the three with session filters |

### Defects found while reviewing

Dark Venus

- Off-by-one in the signal. The bands are copied from shift 1 for two elements, so index 0 is bar 2, but the close it compares against is bar 1. The counter-trend test uses the wrong bar's band. The central-band cross logic has the same inversion.
- Money-management sizing formula is a guess, as the code itself notes.
- With `Stop Target Mode = Disabled`, `Monetary SL = off` and `Percent Loss = off` (the defaults) there is no exit for a losing basket other than margin.

SnapScalp v1

- Break-even is marked done before the broker accepts the modify. If the modify is rejected for minimum distance, BE never retries.
- A partial close (if ever added) would be counted as a finished trade because the close handler fires on any out-deal.
- After a terminal restart the time stop is disabled for the recovered position and the R multiple is rebuilt only if a stop is present.
- The time stop closes at 18 bars regardless of P/L, dumping trades that are in profit but short of target.
- Weekly guard uses `day_of_year / 7`, which does not align to Mondays.
- Lot ceiling of 0.30 stops sizing from scaling with equity.
- Entry frequency is low: 1.3 ATR stretch, RSI(2) below 8, rejection bar, ADX below 28 and session windows all have to line up.

## 2. What v2 changes

Profit side

- **Second entry model.** Bollinger snap-back: the bar before last closes outside the 20/2 band, the last bar closes back inside toward the mean, the outside bar's RSI(2) was extreme, and price still sits at least 0.6 ATR from the EMA. This is the good idea inside Dark Venus, kept, with the grid removed. Roughly doubles signal count. Selectable: stretch only, band only, or both.
- **Scale-out.** Half the position is banked at 0.6 ATR. The runner then targets the mean. After the scale-out the stop moves to break-even plus 0.1R, so the runner cannot turn a banked winner into a net loser.
- **Mean-revert target that follows the mean.** The take profit is re-anchored to the live EMA once per bar. If price crosses the EMA between updates the trade is closed at market. Mean reversion's edge ends at the mean; the target moves with it.
- **Anti-martingale sizing.** Risk is halved after two consecutive losses and raised 1.25× after three consecutive wins. Risk after losses can never go above base. Init refuses a configuration where the boosted risk exceeds 5%.
- **Symbol presets.** Auto-detected from the chart symbol. GBPUSD spread cap 2.0 pips, EURUSD 1.2 pips, USDJPY 1.5 pips with the Tokyo window added. Presets scale correctly on 4-digit and 2-digit feeds.
- **Lot ceiling** raised to 5.0 so size scales with equity. Risk percent is what bounds the loss, not the lot cap.
- **Cost gate** lowered from 3.0× to 2.5× cost. The scale-out target must independently clear 1.5× cost or the trade runs as a single target.

Loss side

- **Volatility band filter.** No entries when ATR(14) is below 0.5× or above 2.5× ATR(100). Dead tape has no reversion, news spikes have no mean.
- **Optional higher-timeframe bias.** When enabled, only dips in an H1 uptrend are bought and only rallies in an H1 downtrend are sold. Off by default. Try it on GBPUSD, which trends the hardest of the three.
- **Time stop that respects working trades.** At 18 bars, only trades below 0.3R are closed. Anything above that is left to its stop, target or trail. Hard close at 36 bars regardless.
- **Break-even retries** every tick until the broker accepts it.
- **Restart-safe state.** Ticket, initial risk, initial lots, scale-out status and open bar are saved to terminal global variables and restored on the first tick. A position with no saved state is rebuilt from the live stop and open time.
- **One trade settles as one trade.** Net P/L for streak counting is summed from history by position id, so the partial close and the final close count once.
- **No re-entry on the exit bar.** Configurable minimum bars between an exit and the next entry.
- **Calendar-correct guard anchors.** Daily anchor keyed on year and day-of-year, weekly anchor keyed on the Monday date.

Unchanged on purpose: hard stop on every trade, one position, no grid, no averaging, no lot multiplication after a loss, three equity guards with flatten-on-trip.

## 3. Recommended settings per pair

Leave `InpPreset = PRESET_AUTO`. The preset only sets the spread cap and the Tokyo session. Everything below is a starting point for the optimiser, not a tuned result.

| Input | GBPUSD | EURUSD | USDJPY |
| --- | --- | --- | --- |
| Signal TF | M5 | M5 | M5 |
| Stretch ATR | 1.30 to 1.50 | 1.20 to 1.30 | 1.20 to 1.40 |
| ADX max | 25 | 28 | 28 |
| Bias mode | Try `BIAS_WITH_TREND` | `BIAS_NONE` | `BIAS_NONE` |
| SL ATR | 1.60 | 1.50 | 1.60 |
| Partial ATR | 0.60 | 0.50 | 0.60 |
| Time stop bars | 18 | 18 | 24 |
| Risk % | 0.75 | 0.75 | 0.75 |
| Sessions | London + NY | London + NY | Tokyo + London + NY |

Use a separate magic number per chart.

## 4. How to test properly

1. Compile `SnapScalp_MR_v2.mq5` in MetaEditor. It was written against the standard `Trade\Trade.mqh` library and has not been compiled in this repository's environment, which has no MetaEditor. Fix any warnings your build reports before testing.
2. Strategy Tester, model **Every tick based on real ticks**, at least two years, ideally 2022 to date so the 2022 GBPUSD trend and the 2024 USDJPY intervention days are in the sample.
3. Set the commission in the tester to match your broker and put the same figure in `InpCommissionRT`.
4. Optimise on the first 70% of the window with the built-in `OnTester` score (net profit per unit drawdown, scaled by trade count). Validate on the last 30% untouched.
5. Run the three pairs together on one account in the tester's multi-symbol mode to see combined drawdown. The daily and weekly guards are per chart; if combined drawdown is too high, lower `InpRiskPercent` rather than the guards.
6. Forward test on a demo account for at least a month before any live allocation.

## 5. What to expect

Mean reversion on M5 majors is a high-frequency, small-edge business. Realistic outcomes after costs are a profit factor in the 1.2 to 1.5 range with a 55% to 65% win rate. The scale-out raises the win rate and lowers the average winner. A result far outside that range on a backtest is more likely a data or cost-model problem than a real edge.

The equity guards will halt trading on bad days. That is the design. If the guards trip often in the backtest, the parameters are wrong for that pair or the spread model is too optimistic, and the answer is not to widen the guards.
