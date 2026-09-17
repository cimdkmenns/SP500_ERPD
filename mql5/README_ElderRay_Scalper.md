# Elder-Ray Scalping Retrofit (NAS100, M2/M5)

`ElderRay_Scalper_M2_M5_EA.mq5` runs the frozen H4/M30 Elder-Ray model
(v6.31 strategy / v6.40 tester EA) on a scalping chart.

## Nothing to configure

**The settings are already compiled into the file.** All 72 input defaults
in the source are exactly the configuration described here, and all 59 of
your frozen model's strategy parameters are byte-identical to your
original EA. To run the diagnostic:

1. Copy `ElderRay_Scalper_M2_M5_EA.mq5` to
   `<MT5 data folder>\MQL5\Experts\` (File → Open Data Folder).
2. Compile it in MetaEditor (F7).
3. Strategy Tester: symbol NAS100, period **M5**, model **Every tick
   based on real ticks**, 6–12 months.
4. Inputs tab → **Load** → `ElderRay_Scalper_baseline.set`. **This step is
   required, not optional.**

> **MT5 remembers the last inputs you used for an EA, per EA name, and
> recompiling does NOT reset them to the new defaults.** An earlier run of
> this EA was demonstrably executed against inputs from a previous build: the
> exit log contained `session end` and `maximum holding period` exits, and
> both of those routes are unreachable when `InpUseEntrySession=false` and
> `InpMaxHoldingBars=0`, which are the shipped defaults. Always load the
> preset before a run whose numbers you intend to trust.

Copy `ElderRay_Scalper_baseline.set` into
`<MT5 data folder>\MQL5\Presets\` so it appears in the Load dialog.

### macOS + Wine: finding the data folder

`File → Open Data Folder` usually cannot open anything under Wine. Two ways
round it:

**The EA tells you.** On every run its first Journal line prints the folder:

```
MT5 data folder (put the EA in MQL5\Experts, presets in MQL5\Presets):
C:\users\you\AppData\Roaming\MetaQuotes\Terminal\<hash>
```

**Or let the script find it.** From the folder holding these files:

```sh
chmod +x install_mac_wine.sh
./install_mac_wine.sh --dry-run   # show what it found and would do
./install_mac_wine.sh             # install after confirming
```

It searches the usual bottle locations, installs the EA into `MQL5/Experts`
and both presets into `MQL5/Presets`, and renames any existing file to
`*.bak-<timestamp>` rather than overwriting it. If your bottle lives
somewhere unusual, point it there with `--root "/path/to/bottle"`.

Roots it searches, which are also the paths to look in by hand:

| Install method | Prefix root |
| --- | --- |
| MetaQuotes MT5 for Mac | `~/Library/Application Support/net.metaquotes.wine.metatrader5` |
| CrossOver | `~/Library/Application Support/CrossOver/Bottles` |
| PlayOnMac | `~/Library/PlayOnMac/wineprefix` |
| Whisky | `~/Library/Containers/com.isaacmarovitz.Whisky/Bottles` |
| Wineskin wrapper | `~/Applications/<Name>.app/Contents/SharedSupport/prefix` |
| Plain Wine | `~/.wine` |

Inside a prefix the data folder is normally
`drive_c/users/<user>/AppData/Roaming/MetaQuotes/Terminal/<32-char hash>/`,
or `drive_c/Program Files/MetaTrader 5/` for a portable install.

Sanity check after any run: if the exit-route table lists `session end` or
`maximum holding period`, you were not running the baseline.

The single run reports which anchor timeframe you should be using, so
there is no anchor sweep to set up by hand.

## Complete input reference

Generated from the source file, so it cannot drift from what compiles.

| Input | Type | Default |
| --- | --- | --- |
| **Elder-Ray core (unchanged principles)** | | |
| `InpEMAPeriod` | int | `13` |
| `InpAppliedPrice` | ENUM_APPLIED_PRICE | `PRICE_CLOSE` |
| `InpAnchorTimeframe` | ER_ANCHOR_TF | `ER_ANCHOR_AUTO` |
| `InpAnchorATRPeriod` | int | `14` |
| `InpAnchorNeutralZoneATR` | double | `0.00` |
| `InpTradeATRPeriod` | int | `14` |
| `InpDominanceThresholdATR` | double | `0.05` |
| `InpPivotLeftBars` | int | `2` |
| `InpPivotRightBars` | int | `2` |
| `InpMinPivotSeparationBars` | int | `3` |
| `InpPivotInitializationLookback` | int | `500` |
| `InpEntrySetupExpiryBars` | int | `4` |
| `InpExitWatchExpiryBars` | int | `48` |
| `InpMinDivergenceDeltaATR` | double | `0.00` |
| **Confirmation filters** | | |
| `InpRequireEntryDominance` | bool | `true` |
| `InpRequireAnchorEMASlope` | bool | `true` |
| `InpAnchorEMASlopeLookbackBars` | int | `1` |
| `InpRequireTradeEMASlope` | bool | `false` |
| `InpTradeEMASlopeLookbackBars` | int | `1` |
| `InpAllowLongEntries` | bool | `true` |
| `InpAllowShortEntries` | bool | `true` |
| **Session (off, as in the frozen model)** | | |
| `InpUseEntrySession` | bool | `false` |
| `InpEntryStartHour` | int | `0` |
| `InpEntryEndHour` | int | `24` |
| `InpServerGMTOffsetHours` | int | `0` |
| `InpSkipMinutesAfterSessionOpen` | int | `0` |
| `InpCloseAtSessionEnd` | bool | `false` |
| **Exits and trade protection** | | |
| `InpExitMode` | ER_EXIT_MODE | `ER_EXIT_OPPOSITE_CONFIRMED_ONLY` |
| `InpStopMode` | ER_STOP_MODE | `ER_STOP_TF_ATR` |
| `InpStopATRMultiplier` | double | `3.0` |
| `InpPivotStopBufferATR` | double | `0.25` |
| `InpTakeProfitR` | double | `4.0` |
| `InpMaxHoldingBars` | int | `0` |
| `InpBreakEvenAtR` | double | `0.0` |
| `InpBreakEvenOffsetR` | double | `0.0` |
| `InpTrailStartR` | double | `3.0` |
| `InpTrailDistanceR` | double | `1.0` |
| `InpProtectLongs` | bool | `false` |
| `InpProtectShorts` | bool | `true` |
| `InpAdaptiveProtection` | bool | `false` |
| `InpProtectionADXThreshold` | double | `20.0` |
| `InpWeakTrendBreakEvenAtR` | double | `0.5` |
| `InpCutLosingAnchorReversals` | bool | `true` |
| `InpCutLosingLongs` | bool | `false` |
| `InpAnchorReversalLossR` | double | `0.25` |
| **Scalping cost and frequency guards (new)** | | |
| `InpMaxSpreadATR` | double | `0.30` |
| `InpMinStopSpreadMultiple` | double | `5.0` |
| `InpMinATRPoints` | double | `0.0` |
| `InpCooldownBars` | int | `0` |
| `InpMaxTradesPerDay` | int | `0` |
| `InpDailyLossStopPercent` | double | `0.0` |
| **Position sizing and account risk** | | |
| `InpUseRiskBasedVolume` | bool | `true` |
| `InpRiskPercent` | double | `8.0` |
| `InpRiskCapitalBase` | double | `10000.0` |
| `InpProfitReinvestmentFraction` | double | `0.5` |
| `InpMaxSizingCapitalMultiple` | double | `1.5` |
| `InpUseDrawdownThrottle` | bool | `true` |
| `InpDrawdownThrottleStartPct` | double | `30.0` |
| `InpDrawdownThrottleRecoveryPct` | double | `10.0` |
| `InpDrawdownThrottleMultiplier` | double | `0.5` |
| `InpShortRiskMultiplier` | double | `0.5` |
| `InpScaleOnlyWeakShorts` | bool | `true` |
| `InpFullShortRiskMinADX` | double | `15.0` |
| `InpCapVolumeToMargin` | bool | `false` |
| `InpMaxFreeMarginUsePercent` | double | `75.0` |
| `InpLots` | double | `1.0` |
| **Execution** | | |
| `InpLogFilterStats` | bool | `true` |
| `InpAllowLiveTrading` | bool | `false` |
| `InpRiskPeakBalanceOverride` | double | `0.0` |
| `InpMagicNumber` | ulong | `41302641` |
| `InpDeviationPoints` | ulong | `20` |
| `InpTradeComment` | string | `ER Scalp` |

## Baseline .set contents

If you would rather create the preset yourself, save this as
`ElderRay_Scalper_baseline.set` in `MQL5\Presets\`:

```
InpEMAPeriod=13||0||0||0||N
InpAppliedPrice=1||0||0||0||N
InpAnchorTimeframe=0||0||0||0||N
InpAnchorATRPeriod=14||0||0||0||N
InpAnchorNeutralZoneATR=0.00||0||0||0||N
InpTradeATRPeriod=14||0||0||0||N
InpDominanceThresholdATR=0.05||0||0||0||N
InpPivotLeftBars=2||0||0||0||N
InpPivotRightBars=2||0||0||0||N
InpMinPivotSeparationBars=3||0||0||0||N
InpPivotInitializationLookback=500||0||0||0||N
InpEntrySetupExpiryBars=4||0||0||0||N
InpExitWatchExpiryBars=48||0||0||0||N
InpMinDivergenceDeltaATR=0.00||0||0||0||N
InpRequireEntryDominance=true||0||0||0||N
InpRequireAnchorEMASlope=true||0||0||0||N
InpAnchorEMASlopeLookbackBars=1||0||0||0||N
InpRequireTradeEMASlope=false||0||0||0||N
InpTradeEMASlopeLookbackBars=1||0||0||0||N
InpAllowLongEntries=true||0||0||0||N
InpAllowShortEntries=true||0||0||0||N
InpUseEntrySession=false||0||0||0||N
InpEntryStartHour=0||0||0||0||N
InpEntryEndHour=24||0||0||0||N
InpServerGMTOffsetHours=0||0||0||0||N
InpSkipMinutesAfterSessionOpen=0||0||0||0||N
InpCloseAtSessionEnd=false||0||0||0||N
InpExitMode=0||0||0||0||N
InpStopMode=1||0||0||0||N
InpStopATRMultiplier=3.0||0||0||0||N
InpPivotStopBufferATR=0.25||0||0||0||N
InpTakeProfitR=4.0||0||0||0||N
InpMaxHoldingBars=0||0||0||0||N
InpBreakEvenAtR=0.0||0||0||0||N
InpBreakEvenOffsetR=0.0||0||0||0||N
InpTrailStartR=3.0||0||0||0||N
InpTrailDistanceR=1.0||0||0||0||N
InpProtectLongs=false||0||0||0||N
InpProtectShorts=true||0||0||0||N
InpAdaptiveProtection=false||0||0||0||N
InpProtectionADXThreshold=20.0||0||0||0||N
InpWeakTrendBreakEvenAtR=0.5||0||0||0||N
InpCutLosingAnchorReversals=true||0||0||0||N
InpCutLosingLongs=false||0||0||0||N
InpAnchorReversalLossR=0.25||0||0||0||N
InpMaxSpreadATR=0.30||0||0||0||N
InpMinStopSpreadMultiple=5.0||0||0||0||N
InpMinATRPoints=0.0||0||0||0||N
InpCooldownBars=0||0||0||0||N
InpMaxTradesPerDay=0||0||0||0||N
InpDailyLossStopPercent=0.0||0||0||0||N
InpUseRiskBasedVolume=true||0||0||0||N
InpRiskPercent=8.0||0||0||0||N
InpRiskCapitalBase=10000.0||0||0||0||N
InpProfitReinvestmentFraction=0.5||0||0||0||N
InpMaxSizingCapitalMultiple=1.5||0||0||0||N
InpUseDrawdownThrottle=true||0||0||0||N
InpDrawdownThrottleStartPct=30.0||0||0||0||N
InpDrawdownThrottleRecoveryPct=10.0||0||0||0||N
InpDrawdownThrottleMultiplier=0.5||0||0||0||N
InpShortRiskMultiplier=0.5||0||0||0||N
InpScaleOnlyWeakShorts=true||0||0||0||N
InpFullShortRiskMinADX=15.0||0||0||0||N
InpCapVolumeToMargin=false||0||0||0||N
InpMaxFreeMarginUsePercent=75.0||0||0||0||N
InpLots=1.0||0||0||0||N
InpLogFilterStats=true||0||0||0||N
InpAllowLiveTrading=false||0||0||0||N
InpRiskPeakBalanceOverride=0.0||0||0||0||N
InpMagicNumber=41302641||0||0||0||N
InpDeviationPoints=20||0||0||0||N
InpTradeComment=ER Scalp||0||0||0||N
```
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

## Anchor survey — one run instead of four

Rather than backtesting each candidate anchor separately, the EA now asks all
four at every live setup-bar whether they *would* have supported the trade,
and reports it at the end of a single run:

```
--- anchor survey: which anchor would have allowed the trade ---
  PERIOD_M15  supported   1180 of   4415 setup-bars ( 26.7%)  ratio 1:3
  PERIOD_M30  supported    398 of   4415 setup-bars (  9.0%)  ratio 1:6   <- in use
  PERIOD_H1   supported   1372 of   4415 setup-bars ( 31.1%)  ratio 1:12
  PERIOD_H4   supported   2104 of   4415 setup-bars ( 47.7%)  ratio 1:24
```

(Numbers illustrative.) The survey never touches a trading decision — it only
counts. Take the anchor with the highest support rate, set
`InpAnchorTimeframe` to it, and confirm with a real run.

Note the support rate does **not** by itself mean profitability: a very slow
anchor allows more trades because it filters less. Compare expectancy per
trade between the top two candidates before settling.

## Included .set files

- `ElderRay_Scalper_baseline.set` — every input at the frozen value. Load via
  Strategy Tester → Inputs → Load.
- `ElderRay_Scalper_sweep.set` — 32-pass frequency sweep over the four levers
  that matter (anchor, dominance, entry expiry, anchor slope). Optimisation:
  *Slow complete algorithm*, forward *No*.

Judge the sweep on **expectancy per trade**, not total net profit — total
profit rewards whichever pass happened to take the most trades.

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
