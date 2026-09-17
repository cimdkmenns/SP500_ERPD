//+------------------------------------------------------------------+
//| Elder-Ray Scalping Retrofit EA (M1-M15, tuned for M2/M5)         |
//| Same strategy skeleton as the frozen H4/M30 v6.31/v6.40 model:   |
//| Elder-Ray pivot divergence on the trade timeframe, confirmed by  |
//| a higher "anchor" timeframe trend, dominance, and R-based risk.  |
//| Retrofitted with the cost/frequency controls a scalp needs.      |
//+------------------------------------------------------------------+
#property copyright "User-developed strategy tester companion"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

// Backtests are not a guarantee of live returns. On a scalping timeframe the
// spread/commission is a large fraction of the stop distance, so validate
// with real-tick data and this broker's actual costs before going live.

enum ER_ANCHOR_TF
{
   ER_ANCHOR_AUTO = 0,   // Auto (targets the frozen model's 8x ratio)
   ER_ANCHOR_M15  = 1,
   ER_ANCHOR_M30  = 2,
   ER_ANCHOR_H1   = 3,
   ER_ANCHOR_H4   = 4
};

enum ER_EXIT_MODE
{
   ER_EXIT_OPPOSITE_CONFIRMED_ONLY = 0,
   ER_EXIT_POTENTIAL_ONLY          = 1,
   ER_EXIT_ANCHOR_TREND_LOSS_ONLY  = 2,
   ER_EXIT_POTENTIAL_OR_ANCHOR_LOSS = 3
};

enum ER_STOP_MODE
{
   ER_STOP_NONE             = 0,
   ER_STOP_TF_ATR           = 1,
   ER_STOP_SETUP_PIVOT      = 2,
   ER_STOP_TIGHTER_OF_BOTH  = 3
};

input group "Elder-Ray core (unchanged principles)"
input int                 InpEMAPeriod                  = 13;
input ENUM_APPLIED_PRICE  InpAppliedPrice               = PRICE_CLOSE;
input ER_ANCHOR_TF        InpAnchorTimeframe            = ER_ANCHOR_AUTO;
input int                 InpAnchorATRPeriod            = 14;
// Every value in this file is the frozen model's. The only thing this EA
// changes is WHICH timeframes the model reads. Cost guards are additive and
// default to a setting that refuses nothing.
input double              InpAnchorNeutralZoneATR       = 0.00;
input int                 InpTradeATRPeriod             = 14;
input double              InpDominanceThresholdATR      = 0.05;
input int                 InpPivotLeftBars              = 2;
input int                 InpPivotRightBars             = 2;
input int                 InpMinPivotSeparationBars     = 3;
input int                 InpPivotInitializationLookback = 500;
input int                 InpEntrySetupExpiryBars       = 4;
input int                 InpExitWatchExpiryBars        = 48;
input double              InpMinDivergenceDeltaATR      = 0.00;

input group "Confirmation filters"
input bool                InpRequireEntryDominance      = true;
input bool                InpRequireAnchorEMASlope      = true;
input int                 InpAnchorEMASlopeLookbackBars = 1;
input bool                InpRequireTradeEMASlope       = false;
input int                 InpTradeEMASlopeLookbackBars  = 1;
input bool                InpAllowLongEntries           = true;
input bool                InpAllowShortEntries          = true;

input group "Session (off, as in the frozen model)"
// Hours below are GMT. Set InpServerGMTOffsetHours to this broker's server
// offset from GMT (e.g. 2 for a UTC+2 server, 3 during its DST) so the
// window stays anchored to the US cash session across DST changes.
// Frozen default is the full day (0-24, disabled). If you want the US cash
// session, 13-21 GMT spans it under both DST regimes (13:30-20:00 summer,
// 14:30-21:00 winter). Test it as a change, do not assume it helps: it
// refused only 3.8% of live setup-bars in the 8-month M5 run.
input bool                InpUseEntrySession            = false;
input int                 InpEntryStartHour             = 0;
input int                 InpEntryEndHour               = 24;
input int                 InpServerGMTOffsetHours       = 0;
// Only meaningful when InpEntryStartHour IS the cash open. The default
// window starts before it, so this skips nothing and stays off.
input int                 InpSkipMinutesAfterSessionOpen = 0;
// A scalp that is still open overnight is no longer a scalp: it is an
// unhedged index gap. Set false to let a winner run past the window.
input bool                InpCloseAtSessionEnd          = false;

input group "Exits and trade protection"
input ER_EXIT_MODE        InpExitMode                   = ER_EXIT_OPPOSITE_CONFIRMED_ONLY;
input ER_STOP_MODE        InpStopMode                   = ER_STOP_TF_ATR;
input double              InpStopATRMultiplier          = 3.0;
input double              InpPivotStopBufferATR         = 0.25;
input double              InpTakeProfitR                = 4.0;
input int                 InpMaxHoldingBars             = 0;
input double              InpBreakEvenAtR               = 0.0;
input double              InpBreakEvenOffsetR           = 0.0;
input double              InpTrailStartR                = 3.0;
input double              InpTrailDistanceR             = 1.0;
input bool                InpProtectLongs               = false;
input bool                InpProtectShorts              = true;
input bool                InpAdaptiveProtection         = false;
input double              InpProtectionADXThreshold     = 20.0;
input double              InpWeakTrendBreakEvenAtR      = 0.5;
input bool                InpCutLosingAnchorReversals   = true;
input bool                InpCutLosingLongs             = false;
input double              InpAnchorReversalLossR        = 0.25;

input group "Scalping cost and frequency guards (new)"
// A scalp only works while the round-trip cost stays small relative to the
// stop. These gates refuse the trade instead of paying the spread blind.
input double              InpMaxSpreadATR               = 0.30;
input double              InpMinStopSpreadMultiple      = 5.0;
input double              InpMinATRPoints               = 0.0;
input int                 InpCooldownBars               = 0;
input int                 InpMaxTradesPerDay            = 0;
input double              InpDailyLossStopPercent       = 0.0;

input group "Position sizing and account risk"
// Risk-based sizing never rounds a position up to the broker minimum. If the
// minimum lot would exceed the risk budget, the trade is skipped. Fixed-lot
// mode retains the original EA's broker-volume normalization.
input bool                InpUseRiskBasedVolume         = true;
// The frozen model's 8%. At scalping trade counts this compounds a losing
// streak far faster than it does on M30 - see the README before trading it.
input double              InpRiskPercent                = 8.0;
input double              InpRiskCapitalBase            = 10000.0;
input double              InpProfitReinvestmentFraction = 0.5;
input double              InpMaxSizingCapitalMultiple   = 1.5;
input bool                InpUseDrawdownThrottle        = true;
input double              InpDrawdownThrottleStartPct   = 30.0;
input double              InpDrawdownThrottleRecoveryPct = 10.0;
input double              InpDrawdownThrottleMultiplier = 0.5;
input double              InpShortRiskMultiplier        = 0.5;
input bool                InpScaleOnlyWeakShorts        = true;
input double              InpFullShortRiskMinADX        = 15.0;
input bool                InpCapVolumeToMargin          = false;
input double              InpMaxFreeMarginUsePercent    = 75.0;
input double              InpLots                       = 1.0;

input group "Execution"
// The parent EA is Strategy-Tester-only. Live use stays opt-in here, and the
// balance peak used by the drawdown throttle must be restored by hand after
// a restart because it is not persisted.
// Counts, for every completed bar on which a setup was pending, the FIRST
// gate that refused the entry. The tally is printed when the test ends and
// is the only reliable way to see which filter is starving the EA.
input bool                InpLogFilterStats             = true;
input bool                InpAllowLiveTrading           = false;
input double              InpRiskPeakBalanceOverride    = 0.0;
input ulong               InpMagicNumber                = 41302641;
input ulong               InpDeviationPoints            = 20;
input string              InpTradeComment               = "ER Scalp";

enum ANCHOR_TREND_STATE
{
   ANCHOR_TREND_BEAR    = -1,
   ANCHOR_TREND_NEUTRAL = 0,
   ANCHOR_TREND_BULL    = 1
};

// Ordered so the tally attributes a refusal to the most informative cause:
// signal-shape gates first, operating gates last.
enum ER_BLOCK_REASON
{
   ER_BLOCK_NONE = 0,
   ER_BLOCK_DIRECTION_DISABLED,
   ER_BLOCK_ENTRY_EXPIRED,
   ER_BLOCK_AFTER_POTENTIAL_EXIT,
   ER_BLOCK_ANCHOR_TREND,
   ER_BLOCK_ANCHOR_SLOPE,
   ER_BLOCK_TRADE_SLOPE,
   ER_BLOCK_DOMINANCE,
   ER_BLOCK_SESSION,
   ER_BLOCK_COOLDOWN,
   ER_BLOCK_DAY_BUDGET,
   ER_BLOCK_VOLATILITY,
   ER_BLOCK_SPREAD,
   ER_BLOCK_REASON_COUNT
};

#define ER_MAX_EXIT_REASONS 16

long blockCounts[ER_BLOCK_REASON_COUNT];
string exitReasonName[ER_MAX_EXIT_REASONS];
long exitReasonCount[ER_MAX_EXIT_REASONS];
double exitReasonSumR[ER_MAX_EXIT_REASONS];
int exitReasonsUsed = 0;

// One position at a time, so a single slot tracks the open trade.
double openRiskMoney = 0.0;
bool openStopWasTrailed = false;
string pendingCloseReason = "";
long bullSetupsConfirmed = 0;
long bearSetupsConfirmed = 0;
long entriesOpened = 0;
long setupsExpiredUnused = 0;
long setupsInvalidated = 0;

// Anchor survey: at every live setup-bar, ask each candidate anchor whether
// it WOULD have supported the trade. One backtest then answers which anchor
// timeframe to use, instead of one backtest per candidate.
#define ER_SURVEY_COUNT 4
ENUM_TIMEFRAMES surveyTF[ER_SURVEY_COUNT];
int surveyEMAHandle[ER_SURVEY_COUNT];
int surveyATRHandle[ER_SURVEY_COUNT];
long surveyPass[ER_SURVEY_COUNT];
long surveyTotal[ER_SURVEY_COUNT];

CTrade trade;
double riskBalancePeak = 0.0;
bool drawdownThrottled = false;

ENUM_TIMEFRAMES tradeTF = PERIOD_CURRENT;
ENUM_TIMEFRAMES anchorTF = PERIOD_M30;
int tradeTFSeconds = 0;

int tradeEMAHandle  = INVALID_HANDLE;
int tradeATRHandle  = INVALID_HANDLE;
int anchorEMAHandle = INVALID_HANDLE;
int anchorATRHandle = INVALID_HANDLE;
int anchorADXHandle = INVALID_HANDLE;

datetime lastTradeBarOpen = 0;
bool engineReady = false;

bool haveLowPivot = false;
bool haveHighPivot = false;
datetime lastLowPivotTime = 0;
datetime lastHighPivotTime = 0;
double lastLowPrice = 0.0;
double lastLowBearPower = 0.0;
double lastHighPrice = 0.0;
double lastHighBullPower = 0.0;

bool bullSetupActive = false;
bool bearSetupActive = false;
datetime bullSetupConfirmedTime = 0;
datetime bearSetupConfirmedTime = 0;
double bullSetupPivotPrice = 0.0;
double bearSetupPivotPrice = 0.0;
bool bullEntryWindowExpired = false;
bool bearEntryWindowExpired = false;
bool bullExitSignalIssued = false;
bool bearExitSignalIssued = false;
datetime lastPotentialLongExitBar = 0;
datetime lastPotentialShortExitBar = 0;

// Scalping session/day bookkeeping.
bool hadManagedPosition = false;
datetime cooldownReleaseTime = 0;
int currentTradingDay = -1;
double dayStartBalance = 0.0;
int tradesToday = 0;
bool dailyLossStopHit = false;

//+------------------------------------------------------------------+
int OnInit()
{
   if(!MQLInfoInteger(MQL_TESTER) && !InpAllowLiveTrading)
   {
      Print("Elder-Ray scalper: Strategy Tester only unless "
            "InpAllowLiveTrading is enabled deliberately.");
      return INIT_PARAMETERS_INCORRECT;
   }

   tradeTF = (ENUM_TIMEFRAMES)_Period;
   tradeTFSeconds = PeriodSeconds(tradeTF);
   if(tradeTFSeconds < 60 || tradeTFSeconds > 15 * 60)
   {
      Print("Attach this scalping EA on M1-M15 (M2 or M5 recommended). "
            "Use the original H4/M30 EA for swing timeframes.");
      return INIT_PARAMETERS_INCORRECT;
   }
   anchorTF = ResolveAnchorTimeframe();
   if(PeriodSeconds(anchorTF) <= tradeTFSeconds)
   {
      Print("The anchor timeframe must be higher than the chart timeframe.");
      return INIT_PARAMETERS_INCORRECT;
   }

   riskBalancePeak = InpRiskPeakBalanceOverride > 0.0
                     ? InpRiskPeakBalanceOverride
                     : AccountInfoDouble(ACCOUNT_BALANCE);
   drawdownThrottled = false;
   hadManagedPosition = false;
   cooldownReleaseTime = 0;
   currentTradingDay = -1;
   dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   tradesToday = 0;
   dailyLossStopHit = false;
   ArrayInitialize(blockCounts, 0);
   ArrayInitialize(exitReasonCount, 0);
   ArrayInitialize(exitReasonSumR, 0.0);
   exitReasonsUsed = 0;
   openRiskMoney = 0.0;
   openStopWasTrailed = false;
   pendingCloseReason = "";
   bullSetupsConfirmed = 0;
   bearSetupsConfirmed = 0;
   entriesOpened = 0;
   setupsExpiredUnused = 0;
   setupsInvalidated = 0;
   surveyTF[0] = PERIOD_M15;
   surveyTF[1] = PERIOD_M30;
   surveyTF[2] = PERIOD_H1;
   surveyTF[3] = PERIOD_H4;
   ArrayInitialize(surveyPass, 0);
   ArrayInitialize(surveyTotal, 0);
   ArrayInitialize(surveyEMAHandle, INVALID_HANDLE);
   ArrayInitialize(surveyATRHandle, INVALID_HANDLE);

   if(InpEMAPeriod < 2 || InpAnchorATRPeriod < 2 ||
      InpAnchorNeutralZoneATR < 0.0 || InpTradeATRPeriod < 2 ||
      InpDominanceThresholdATR < 0.0 ||
      InpPivotLeftBars < 1 || InpPivotRightBars < 1 ||
      InpMinPivotSeparationBars < 1 ||
      InpPivotInitializationLookback < 10 ||
      InpEntrySetupExpiryBars < 1 ||
      InpExitWatchExpiryBars < InpEntrySetupExpiryBars ||
      InpMinDivergenceDeltaATR < 0.0 ||
      InpAnchorEMASlopeLookbackBars < 1 ||
      InpTradeEMASlopeLookbackBars < 1 ||
      InpEntryStartHour < 0 || InpEntryStartHour > 23 ||
      InpEntryEndHour < 1 || InpEntryEndHour > 24 ||
      InpServerGMTOffsetHours < -12 || InpServerGMTOffsetHours > 14 ||
      InpSkipMinutesAfterSessionOpen < 0 ||
      InpSkipMinutesAfterSessionOpen >= 60 ||
      InpStopATRMultiplier <= 0.0 || InpPivotStopBufferATR < 0.0 ||
      InpTakeProfitR < 0.0 || InpMaxHoldingBars < 0 ||
      InpBreakEvenAtR < 0.0 || InpBreakEvenOffsetR < 0.0 ||
      InpMaxFreeMarginUsePercent <= 0.0 || InpMaxFreeMarginUsePercent > 100.0 ||
      InpTrailStartR < 0.0 || InpTrailDistanceR <= 0.0 ||
      InpProtectionADXThreshold <= 0.0 || InpWeakTrendBreakEvenAtR <= 0.0 ||
      InpAnchorReversalLossR < 0.0 ||
      InpMaxSpreadATR < 0.0 || InpMinStopSpreadMultiple < 0.0 ||
      InpMinATRPoints < 0.0 || InpCooldownBars < 0 ||
      InpMaxTradesPerDay < 0 || InpDailyLossStopPercent < 0.0 ||
      InpDailyLossStopPercent >= 100.0 ||
      (InpTrailStartR > 0.0 && InpTakeProfitR <= 0.0) ||
      (InpBreakEvenAtR > 0.0 && InpTakeProfitR <= 0.0) ||
      InpRiskPercent <= 0.0 || InpRiskPercent > 8.0 || InpLots <= 0.0 ||
      InpDrawdownThrottleStartPct <= 0.0 || InpDrawdownThrottleStartPct >= 100.0 ||
      InpDrawdownThrottleRecoveryPct < 0.0 ||
      InpDrawdownThrottleRecoveryPct >= InpDrawdownThrottleStartPct ||
      InpDrawdownThrottleMultiplier <= 0.0 || InpDrawdownThrottleMultiplier > 1.0 ||
      InpRiskCapitalBase <= 0.0 ||
      InpProfitReinvestmentFraction < 0.0 || InpProfitReinvestmentFraction > 1.0 ||
      (InpMaxSizingCapitalMultiple != 0.0 && InpMaxSizingCapitalMultiple < 1.0) ||
      InpShortRiskMultiplier <= 0.0 || InpShortRiskMultiplier > 1.0 ||
      InpFullShortRiskMinADX < 0.0 ||
      (InpUseRiskBasedVolume && InpStopMode == ER_STOP_NONE))
   {
      Print("Invalid Elder-Ray scalper input parameters.");
      return INIT_PARAMETERS_INCORRECT;
   }

   tradeEMAHandle = iMA(_Symbol, tradeTF, InpEMAPeriod, 0,
                        MODE_EMA, InpAppliedPrice);
   tradeATRHandle = iATR(_Symbol, tradeTF, InpTradeATRPeriod);
   anchorEMAHandle = iMA(_Symbol, anchorTF, InpEMAPeriod, 0,
                         MODE_EMA, InpAppliedPrice);
   anchorATRHandle = iATR(_Symbol, anchorTF, InpAnchorATRPeriod);
   anchorADXHandle = iADX(_Symbol, anchorTF, 14);

   if(tradeEMAHandle == INVALID_HANDLE || tradeATRHandle == INVALID_HANDLE ||
      anchorEMAHandle == INVALID_HANDLE || anchorATRHandle == INVALID_HANDLE ||
      anchorADXHandle == INVALID_HANDLE)
   {
      PrintFormat("Unable to create indicator handles. Error %d", GetLastError());
      ReleaseHandles();
      return INIT_FAILED;
   }

   if(InpLogFilterStats)
   {
      for(int index = 0; index < ER_SURVEY_COUNT; index++)
      {
         if(PeriodSeconds(surveyTF[index]) <= tradeTFSeconds)
            continue;
         surveyEMAHandle[index] = iMA(_Symbol, surveyTF[index], InpEMAPeriod,
                                      0, MODE_EMA, InpAppliedPrice);
         surveyATRHandle[index] = iATR(_Symbol, surveyTF[index],
                                       InpAnchorATRPeriod);
      }
   }

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);
   trade.SetAsyncMode(false);
   trade.SetTypeFillingBySymbol(_Symbol);

   PrintFormat("Elder-Ray scalper initialized on %s with %s anchor. "
               "Exit mode %d, stop mode %d, risk sizing %s.",
               EnumToString(tradeTF), EnumToString(anchorTF),
               (int)InpExitMode, (int)InpStopMode,
               InpUseRiskBasedVolume ? "ON" : "OFF");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   PrintFilterStatistics();
   ReleaseHandles();
}

//+------------------------------------------------------------------+
ENUM_TIMEFRAMES ResolveAnchorTimeframe()
{
   if(InpAnchorTimeframe != ER_ANCHOR_AUTO)
   {
      switch(InpAnchorTimeframe)
      {
         case ER_ANCHOR_M15: return PERIOD_M15;
         case ER_ANCHOR_M30: return PERIOD_M30;
         case ER_ANCHOR_H1:  return PERIOD_H1;
         case ER_ANCHOR_H4:  return PERIOD_H4;
      }
   }

   // The frozen model reads an anchor 8x its trade timeframe (M30/H4). That
   // ratio is what keeps a trade-timeframe swing too small to flip the anchor
   // at the moment a divergence forms. At 1:6 (M5/M30) the anchor disagreed
   // with 91% of live setup-bars, because an M5 swing IS a move on M30.
   // M2 -> M15 is 1:7.5 and reproduces the frozen geometry most closely.
   if(tradeTFSeconds <= 3 * 60)
      return PERIOD_M15;
   if(tradeTFSeconds <= 6 * 60)
      return PERIOD_H1;
   return PERIOD_H4;
}

//+------------------------------------------------------------------+
void OnTick()
{
   UpdateRiskThrottle();
   datetime currentBarOpen = iTime(_Symbol, tradeTF, 0);
   if(currentBarOpen == 0)
      return;

   if(!engineReady)
   {
      if(!DataReady() || !SeedPivotState())
         return;

      // Begin generating trades from the next completed candle. Older data is
      // used only to seed the preceding pivot comparisons.
      lastTradeBarOpen = currentBarOpen;
      engineReady = true;
      PrintFormat("Signal engine ready at %s",
                  TimeToString(currentBarOpen, TIME_DATE | TIME_MINUTES));
      return;
   }

   if(currentBarOpen == lastTradeBarOpen)
      return;

   // If ticks/data skipped more than one bar opening, process every newly
   // completed candle chronologically. Trades still occur at the current tick.
   int previousOpenShift = iBarShift(_Symbol, tradeTF, lastTradeBarOpen, true);
   if(previousOpenShift < 1)
      previousOpenShift = 1;

   for(int closedShift = previousOpenShift;
       closedShift >= 1; closedShift--)
      ProcessClosedTradeBar(closedShift);

   lastTradeBarOpen = currentBarOpen;
}

//+------------------------------------------------------------------+
bool DataReady()
{
   int tradeNeeded = InpPivotInitializationLookback +
                     InpPivotLeftBars + InpPivotRightBars + 10;
   int anchorNeeded = MathMax(InpEMAPeriod, InpAnchorATRPeriod) + 10;
   if(InpScaleOnlyWeakShorts || InpAdaptiveProtection)
      anchorNeeded = MathMax(anchorNeeded, 38);

   if(Bars(_Symbol, tradeTF) < tradeNeeded ||
      Bars(_Symbol, anchorTF) < anchorNeeded)
      return false;

   return BarsCalculated(tradeEMAHandle) >= tradeNeeded &&
          BarsCalculated(tradeATRHandle) >= tradeNeeded &&
          BarsCalculated(anchorEMAHandle) >= anchorNeeded &&
          BarsCalculated(anchorATRHandle) >= anchorNeeded &&
          (!(InpScaleOnlyWeakShorts || InpAdaptiveProtection) ||
           BarsCalculated(anchorADXHandle) >= anchorNeeded);
}

//+------------------------------------------------------------------+
void ReleaseHandles()
{
   if(tradeEMAHandle != INVALID_HANDLE)
      IndicatorRelease(tradeEMAHandle);
   if(tradeATRHandle != INVALID_HANDLE)
      IndicatorRelease(tradeATRHandle);
   if(anchorEMAHandle != INVALID_HANDLE)
      IndicatorRelease(anchorEMAHandle);
   if(anchorATRHandle != INVALID_HANDLE)
      IndicatorRelease(anchorATRHandle);
   if(anchorADXHandle != INVALID_HANDLE)
      IndicatorRelease(anchorADXHandle);

   tradeEMAHandle = INVALID_HANDLE;
   tradeATRHandle = INVALID_HANDLE;
   anchorEMAHandle = INVALID_HANDLE;
   anchorATRHandle = INVALID_HANDLE;
   anchorADXHandle = INVALID_HANDLE;

   for(int index = 0; index < ER_SURVEY_COUNT; index++)
   {
      if(surveyEMAHandle[index] != INVALID_HANDLE)
         IndicatorRelease(surveyEMAHandle[index]);
      if(surveyATRHandle[index] != INVALID_HANDLE)
         IndicatorRelease(surveyATRHandle[index]);
      surveyEMAHandle[index] = INVALID_HANDLE;
      surveyATRHandle[index] = INVALID_HANDLE;
   }
}

//+------------------------------------------------------------------+
bool SeedPivotState()
{
   haveLowPivot = false;
   haveHighPivot = false;

   // Pivots at shifts <= right-bars are not yet confirmed. Iterate from
   // oldest to newest so the stored pivot is the latest confirmed one.
   int firstShift = InpPivotRightBars + 1;
   int lastShift = firstShift + InpPivotInitializationLookback - 1;

   for(int shift = lastShift; shift >= firstShift; shift--)
   {
      if(!BarExists(tradeTF, shift + InpPivotLeftBars))
         continue;

      if(IsPivotLow(shift))
      {
         double ema = 0.0;
         if(!BufferValue(tradeEMAHandle, shift, ema))
            return false;

         haveLowPivot = true;
         lastLowPivotTime = iTime(_Symbol, tradeTF, shift);
         lastLowPrice = iLow(_Symbol, tradeTF, shift);
         lastLowBearPower = lastLowPrice - ema;
      }

      if(IsPivotHigh(shift))
      {
         double ema = 0.0;
         if(!BufferValue(tradeEMAHandle, shift, ema))
            return false;

         haveHighPivot = true;
         lastHighPivotTime = iTime(_Symbol, tradeTF, shift);
         lastHighPrice = iHigh(_Symbol, tradeTF, shift);
         lastHighBullPower = lastHighPrice - ema;
      }
   }

   return haveLowPivot || haveHighPivot;
}

//+------------------------------------------------------------------+
void ProcessClosedTradeBar(int closedShift)
{
   if(closedShift < 1)
      return;

   datetime closedBarTime = iTime(_Symbol, tradeTF, closedShift);
   if(closedBarTime == 0)
      return;

   CheckNewlyConfirmedPivot(closedShift);
   ExpireOldSetups(closedShift);
   InvalidateBrokenSetups(closedShift);

   // Restore setup state after a reconnect, but never submit a series of
   // backdated orders. Only the latest completed candle can manage/enter.
   if(closedShift > 1)
      return;

   RollTradingDay(closedBarTime);
   FlattenAfterSessionEnd(closedBarTime);

   int anchorTrend = GetClosedAnchorTrend(closedBarTime);
   double currentEMA = 0.0;
   double currentATR = 0.0;
   bool dominanceReady =
      BufferValue(tradeEMAHandle, closedShift, currentEMA) &&
      BufferValue(tradeATRHandle, closedShift, currentATR);
   double dominance = 0.0;
   double threshold = 0.0;
   if(dominanceReady)
   {
      double bullPower = iHigh(_Symbol, tradeTF, closedShift) - currentEMA;
      double bearPower = iLow(_Symbol, tradeTF, closedShift) - currentEMA;
      dominance = MathAbs(bullPower) - MathAbs(bearPower);
      threshold = currentATR * InpDominanceThresholdATR;
   }

   bool bearDominant = dominanceReady && dominance < -threshold;
   bool bullDominant = dominanceReady && dominance > threshold;

   // Identical conjunction to the frozen model, evaluated as an ordered gate
   // chain so the first refusal can be attributed and counted.
   int longBlock = EvaluateEntryBlock(POSITION_TYPE_BUY, closedBarTime,
                                      closedShift, anchorTrend,
                                      bullDominant, currentATR);
   int shortBlock = EvaluateEntryBlock(POSITION_TYPE_SELL, closedBarTime,
                                       closedShift, anchorTrend,
                                       bearDominant, currentATR);
   if(longBlock >= 0)
      blockCounts[longBlock]++;
   if(shortBlock >= 0)
      blockCounts[shortBlock]++;

   bool longEntry = (longBlock == ER_BLOCK_NONE);
   bool shortEntry = (shortBlock == ER_BLOCK_NONE);

   bool usePotentialExit =
      InpExitMode == ER_EXIT_POTENTIAL_ONLY ||
      InpExitMode == ER_EXIT_POTENTIAL_OR_ANCHOR_LOSS;
   bool useAnchorLossExit =
      InpExitMode == ER_EXIT_ANCHOR_TREND_LOSS_ONLY ||
      InpExitMode == ER_EXIT_POTENTIAL_OR_ANCHOR_LOSS;

   // Potential exits match the indicator: the completed anchor trend must
   // still support the open position, while the opposite trade-timeframe
   // setup and dominance warn that the move is deteriorating. Entry-window
   // expiry does not end the longer exit-watch eligibility.
   bool potentialLongExit =
      usePotentialExit &&
      HasManagedPosition(POSITION_TYPE_BUY) &&
      anchorTrend == ANCHOR_TREND_BULL && bearSetupActive &&
      !bearExitSignalIssued && bearDominant &&
      BarIsAtOrAfter(closedBarTime, bearSetupConfirmedTime);
   bool potentialShortExit =
      usePotentialExit &&
      HasManagedPosition(POSITION_TYPE_SELL) &&
      anchorTrend == ANCHOR_TREND_BEAR && bullSetupActive &&
      !bullExitSignalIssued && bullDominant &&
      BarIsAtOrAfter(closedBarTime, bullSetupConfirmedTime);

   // When both setup directions overlap, give the newer setup priority. This
   // prevents a fresh same-direction entry and its exit warning from acting
   // on the same completed candle.
   if(longEntry && potentialLongExit)
   {
      if(bullSetupConfirmedTime >= bearSetupConfirmedTime)
         potentialLongExit = false;
      else
         longEntry = false;
   }
   if(shortEntry && potentialShortExit)
   {
      if(bearSetupConfirmedTime >= bullSetupConfirmedTime)
         potentialShortExit = false;
      else
         shortEntry = false;
   }

   ApplyBreakEvenAtClosedBar(closedShift);
   ApplyTrailingAtClosedBar(closedShift);
   CutLosingAnchorReversals(closedShift, anchorTrend);

   // Hard management exits are evaluated before the optional Elder-Ray
   // potential-exit warning on the same completed candle.
   if(HasManagedPosition(POSITION_TYPE_BUY))
   {
      if(InpMaxHoldingBars > 0 &&
         ManagedPositionHeldBars(POSITION_TYPE_BUY, closedBarTime) >=
         InpMaxHoldingBars)
      {
         CloseManagedPositions(POSITION_TYPE_BUY, "maximum holding period");
      }
      else if(useAnchorLossExit && anchorTrend != ANCHOR_TREND_BULL)
      {
         CloseManagedPositions(POSITION_TYPE_BUY, "completed anchor trend loss");
      }
   }

   if(HasManagedPosition(POSITION_TYPE_SELL))
   {
      if(InpMaxHoldingBars > 0 &&
         ManagedPositionHeldBars(POSITION_TYPE_SELL, closedBarTime) >=
         InpMaxHoldingBars)
      {
         CloseManagedPositions(POSITION_TYPE_SELL, "maximum holding period");
      }
      else if(useAnchorLossExit && anchorTrend != ANCHOR_TREND_BEAR)
      {
         CloseManagedPositions(POSITION_TYPE_SELL, "completed anchor trend loss");
      }
   }

   potentialLongExit = potentialLongExit &&
                       HasManagedPosition(POSITION_TYPE_BUY);
   potentialShortExit = potentialShortExit &&
                        HasManagedPosition(POSITION_TYPE_SELL);

   if(potentialLongExit)
   {
      PrintFormat("POTENTIAL_LONG_EXIT confirmed at close %s: "
                  "bear setup + bear dominance. Closing BUY to flat.",
                  TimeToString(closedBarTime + tradeTFSeconds,
                               TIME_DATE | TIME_MINUTES));
      if(CloseManagedPositions(POSITION_TYPE_BUY, "POTENTIAL_LONG_EXIT"))
      {
         bearExitSignalIssued = true;
         lastPotentialLongExitBar = closedBarTime;
      }
   }

   if(potentialShortExit)
   {
      PrintFormat("POTENTIAL_SHORT_EXIT confirmed at close %s: "
                  "bull setup + bull dominance. Closing SELL to flat.",
                  TimeToString(closedBarTime + tradeTFSeconds,
                               TIME_DATE | TIME_MINUTES));
      if(CloseManagedPositions(POSITION_TYPE_SELL, "POTENTIAL_SHORT_EXIT"))
      {
         bullExitSignalIssued = true;
         lastPotentialShortExitBar = closedBarTime;
      }
   }

   if(longEntry)
   {
      PrintFormat("LONG_ENTRY confirmed at close %s: anchor bullish + "
                  "active bull setup. Executing at next available tick.",
                  TimeToString(closedBarTime + tradeTFSeconds,
                               TIME_DATE | TIME_MINUTES));
      if(ExecuteConfirmedDirection(POSITION_TYPE_BUY, closedBarTime,
                                   currentATR, bullSetupPivotPrice))
      {
         tradesToday++;
         entriesOpened++;
      }
      ConsumeBullSetup();
   }

   if(shortEntry)
   {
      PrintFormat("SHORT_ENTRY confirmed at close %s: anchor bearish + "
                  "active bear setup. Executing at next available tick.",
                  TimeToString(closedBarTime + tradeTFSeconds,
                               TIME_DATE | TIME_MINUTES));
      if(ExecuteConfirmedDirection(POSITION_TYPE_SELL, closedBarTime,
                                   currentATR, bearSetupPivotPrice))
      {
         tradesToday++;
         entriesOpened++;
      }
      ConsumeBearSetup();
   }

   UpdateCooldownState(closedBarTime);
}

//+------------------------------------------------------------------+
// Returns -1 when no setup is pending in this direction (nothing to refuse,
// so nothing is counted), ER_BLOCK_NONE when every gate passes, otherwise
// the first gate that refused.
int EvaluateEntryBlock(ENUM_POSITION_TYPE desiredType,
                       datetime closedBarTime,
                       int closedShift,
                       int anchorTrend,
                       bool dominant,
                       double currentATR)
{
   bool isLong = (desiredType == POSITION_TYPE_BUY);
   bool setupActive = isLong ? bullSetupActive : bearSetupActive;
   datetime confirmed = isLong ? bullSetupConfirmedTime : bearSetupConfirmedTime;
   if(!setupActive || !BarIsAtOrAfter(closedBarTime, confirmed))
      return -1;

   if(isLong ? !InpAllowLongEntries : !InpAllowShortEntries)
      return ER_BLOCK_DIRECTION_DISABLED;
   // An expired setup stays alive for the exit watch. Counting it on every
   // one of those bars swamped the tally; it is reported per setup instead.
   if(isLong ? bullEntryWindowExpired : bearEntryWindowExpired)
      return -1;

   datetime lastExit = isLong ? lastPotentialLongExitBar
                              : lastPotentialShortExitBar;
   if(lastExit != 0 && confirmed <= lastExit)
      return ER_BLOCK_AFTER_POTENTIAL_EXIT;

   int desiredTrend = isLong ? ANCHOR_TREND_BULL : ANCHOR_TREND_BEAR;
   SurveyAnchors(closedBarTime, desiredTrend);
   if(anchorTrend != desiredTrend)
      return ER_BLOCK_ANCHOR_TREND;
   if(!AnchorSlopeSupports(closedBarTime, desiredTrend))
      return ER_BLOCK_ANCHOR_SLOPE;
   if(!TradeSlopeSupports(closedShift, desiredTrend))
      return ER_BLOCK_TRADE_SLOPE;
   if(InpRequireEntryDominance && !dominant)
      return ER_BLOCK_DOMINANCE;

   if(!EntryWindowAllowed(closedBarTime))
      return ER_BLOCK_SESSION;
   if(closedBarTime < cooldownReleaseTime)
      return ER_BLOCK_COOLDOWN;
   if(!TradingBudgetAllows())
      return ER_BLOCK_DAY_BUDGET;
   if(!VolatilityAllowsEntry(currentATR))
      return ER_BLOCK_VOLATILITY;
   if(!SpreadAllowsEntry(currentATR))
      return ER_BLOCK_SPREAD;
   return ER_BLOCK_NONE;
}

//+------------------------------------------------------------------+
string BlockReasonName(int reason)
{
   switch(reason)
   {
      case ER_BLOCK_NONE:                  return "passed all gates";
      case ER_BLOCK_DIRECTION_DISABLED:    return "direction disabled";
      case ER_BLOCK_ENTRY_EXPIRED:         return "entry window expired";
      case ER_BLOCK_AFTER_POTENTIAL_EXIT:  return "setup predates last exit";
      case ER_BLOCK_ANCHOR_TREND:          return "anchor trend disagreed";
      case ER_BLOCK_ANCHOR_SLOPE:          return "anchor EMA slope";
      case ER_BLOCK_TRADE_SLOPE:           return "trade EMA slope";
      case ER_BLOCK_DOMINANCE:             return "dominance";
      case ER_BLOCK_SESSION:               return "outside session";
      case ER_BLOCK_COOLDOWN:              return "cooldown";
      case ER_BLOCK_DAY_BUDGET:            return "day trade/loss cap";
      case ER_BLOCK_VOLATILITY:            return "ATR below floor";
      case ER_BLOCK_SPREAD:                return "spread too wide";
   }
   return "unknown";
}

//+------------------------------------------------------------------+
void PrintFilterStatistics()
{
   if(!InpLogFilterStats)
      return;

   long total = 0;
   for(int reason = 0; reason < ER_BLOCK_REASON_COUNT; reason++)
      total += blockCounts[reason];

   PrintFormat("=== Elder-Ray scalper filter statistics (%s / %s anchor) ===",
               EnumToString(tradeTF), EnumToString(anchorTF));
   long setupsTotal = bullSetupsConfirmed + bearSetupsConfirmed;
   PrintFormat("Setups confirmed: %I64d bull, %I64d bear (%I64d total).",
               bullSetupsConfirmed, bearSetupsConfirmed, setupsTotal);
   PrintFormat("  -> entries opened      %I64d", entriesOpened);
   PrintFormat("  -> expired unused      %I64d", setupsExpiredUnused);
   PrintFormat("  -> invalidated by price %I64d", setupsInvalidated);
   Print("Gate tally below covers only bars inside a live entry window.");
   if(total <= 0)
   {
      Print("No setup was ever pending on a completed bar.");
      return;
   }

   PrintFormat("Setup-bar decisions: %I64d", total);
   for(int reason = 0; reason < ER_BLOCK_REASON_COUNT; reason++)
   {
      if(blockCounts[reason] <= 0)
         continue;
      PrintFormat("  %-28s %8I64d  (%5.1f%%)", BlockReasonName(reason),
                  blockCounts[reason], 100.0 * blockCounts[reason] / total);
   }
   Print("A gate holding a large share is the one to question first. A gate "
         "at 0% is doing nothing and can be ruled out as the cause.");

   long surveyBase = 0;
   for(int index = 0; index < ER_SURVEY_COUNT; index++)
      surveyBase = MathMax(surveyBase, surveyTotal[index]);
   if(surveyBase > 0)
   {
      Print("--- anchor survey: which anchor would have allowed the trade ---");
      for(int index = 0; index < ER_SURVEY_COUNT; index++)
      {
         if(surveyTotal[index] <= 0)
            continue;
         PrintFormat("  %-8s supported %6I64d of %6I64d setup-bars (%5.1f%%)"
                     "  ratio 1:%.0f%s",
                     EnumToString(surveyTF[index]), surveyPass[index],
                     surveyTotal[index],
                     100.0 * surveyPass[index] / surveyTotal[index],
                     (double)PeriodSeconds(surveyTF[index]) / tradeTFSeconds,
                     surveyTF[index] == anchorTF ? "   <- in use" : "");
      }
      Print("Pick the anchor with the highest support rate, then confirm it "
            "with a real run. This survey costs one backtest, not four.");
   }

   if(exitReasonsUsed <= 0)
      return;

   long exitTotal = 0;
   double sumR = 0.0;
   for(int index = 0; index < exitReasonsUsed; index++)
   {
      exitTotal += exitReasonCount[index];
      sumR += exitReasonSumR[index];
   }
   Print("--- exit routes (average R by how the trade ended) ---");
   for(int index = 0; index < exitReasonsUsed; index++)
   {
      if(exitReasonCount[index] <= 0)
         continue;
      PrintFormat("  %-34s %5I64d trades  avg %+6.2fR  total %+7.2fR",
                  exitReasonName[index], exitReasonCount[index],
                  exitReasonSumR[index] / exitReasonCount[index],
                  exitReasonSumR[index]);
   }
   if(exitTotal > 0)
      PrintFormat("  %-34s %5I64d trades  avg %+6.2fR  total %+7.2fR",
                  "ALL", exitTotal, sumR / exitTotal, sumR);
   Print("An exit route with a large count and a small positive average is "
         "cutting winners before the target pays for the losers.");
}

//+------------------------------------------------------------------+
void CheckNewlyConfirmedPivot(int closedShift)
{
   int pivotShift = closedShift + InpPivotRightBars;
   if(!BarExists(tradeTF, pivotShift + InpPivotLeftBars))
      return;

   datetime pivotTime = iTime(_Symbol, tradeTF, pivotShift);
   datetime confirmationTime = iTime(_Symbol, tradeTF, closedShift);
   double pivotEMA = 0.0;
   double pivotATR = 0.0;
   if(pivotTime == 0 || confirmationTime == 0 ||
      !BufferValue(tradeEMAHandle, pivotShift, pivotEMA) ||
      !BufferValue(tradeATRHandle, pivotShift, pivotATR))
      return;

   double minDelta = pivotATR * InpMinDivergenceDeltaATR;

   if(IsPivotLow(pivotShift))
   {
      double pivotLow = iLow(_Symbol, tradeTF, pivotShift);
      double pivotBearPower = pivotLow - pivotEMA;
      int separation = BarsBetween(lastLowPivotTime, pivotShift);

      bool bullishDivergence = haveLowPivot &&
         separation >= InpMinPivotSeparationBars &&
         pivotLow < lastLowPrice &&
         pivotBearPower < 0.0 &&
         pivotBearPower > lastLowBearPower + minDelta;

      if(bullishDivergence)
      {
         bullSetupActive = true;
         bullSetupsConfirmed++;
         bullSetupConfirmedTime = confirmationTime;
         bullSetupPivotPrice = pivotLow;
         bullEntryWindowExpired = false;
         bullExitSignalIssued = false;
         PrintFormat("BULL_SETUP confirmed at %s from pivot %s (pivot %.5f)",
                     TimeToString(confirmationTime, TIME_DATE | TIME_MINUTES),
                     TimeToString(pivotTime, TIME_DATE | TIME_MINUTES),
                     pivotLow);
      }

      haveLowPivot = true;
      lastLowPivotTime = pivotTime;
      lastLowPrice = pivotLow;
      lastLowBearPower = pivotBearPower;
   }

   if(IsPivotHigh(pivotShift))
   {
      double pivotHigh = iHigh(_Symbol, tradeTF, pivotShift);
      double pivotBullPower = pivotHigh - pivotEMA;
      int separation = BarsBetween(lastHighPivotTime, pivotShift);

      bool bearishDivergence = haveHighPivot &&
         separation >= InpMinPivotSeparationBars &&
         pivotHigh > lastHighPrice &&
         pivotBullPower > 0.0 &&
         pivotBullPower < lastHighBullPower - minDelta;

      if(bearishDivergence)
      {
         bearSetupActive = true;
         bearSetupsConfirmed++;
         bearSetupConfirmedTime = confirmationTime;
         bearSetupPivotPrice = pivotHigh;
         bearEntryWindowExpired = false;
         bearExitSignalIssued = false;
         PrintFormat("BEAR_SETUP confirmed at %s from pivot %s (pivot %.5f)",
                     TimeToString(confirmationTime, TIME_DATE | TIME_MINUTES),
                     TimeToString(pivotTime, TIME_DATE | TIME_MINUTES),
                     pivotHigh);
      }

      haveHighPivot = true;
      lastHighPivotTime = pivotTime;
      lastHighPrice = pivotHigh;
      lastHighBullPower = pivotBullPower;
   }
}

//+------------------------------------------------------------------+
void ExpireOldSetups(int currentShift)
{
   if(bullSetupActive)
   {
      int elapsed = BarsElapsed(bullSetupConfirmedTime, currentShift);
      if(elapsed > InpExitWatchExpiryBars)
      {
         Print("BULL_SETUP fully expired.");
         ConsumeBullSetup();
      }
      else if(!bullEntryWindowExpired &&
              elapsed > InpEntrySetupExpiryBars)
      {
         bullEntryWindowExpired = true;
         setupsExpiredUnused++;
         Print("BULL_SETUP entry window expired; it cannot generate an entry.");
      }
   }

   if(bearSetupActive)
   {
      int elapsed = BarsElapsed(bearSetupConfirmedTime, currentShift);
      if(elapsed > InpExitWatchExpiryBars)
      {
         Print("BEAR_SETUP fully expired.");
         ConsumeBearSetup();
      }
      else if(!bearEntryWindowExpired &&
              elapsed > InpEntrySetupExpiryBars)
      {
         bearEntryWindowExpired = true;
         setupsExpiredUnused++;
         Print("BEAR_SETUP entry window expired; it cannot generate an entry.");
      }
   }
}

//+------------------------------------------------------------------+
void InvalidateBrokenSetups(int currentShift)
{
   double barClose = iClose(_Symbol, tradeTF, currentShift);
   datetime barTime = iTime(_Symbol, tradeTF, currentShift);

   if(bullSetupActive &&
      BarIsAtOrAfter(barTime, bullSetupConfirmedTime) &&
      barClose < bullSetupPivotPrice)
   {
      PrintFormat("BULL_SETUP invalidated at %s: close %.5f below pivot %.5f",
                  TimeToString(barTime, TIME_DATE | TIME_MINUTES),
                  barClose, bullSetupPivotPrice);
      setupsInvalidated++;
      ConsumeBullSetup();
   }

   if(bearSetupActive &&
      BarIsAtOrAfter(barTime, bearSetupConfirmedTime) &&
      barClose > bearSetupPivotPrice)
   {
      PrintFormat("BEAR_SETUP invalidated at %s: close %.5f above pivot %.5f",
                  TimeToString(barTime, TIME_DATE | TIME_MINUTES),
                  barClose, bearSetupPivotPrice);
      setupsInvalidated++;
      ConsumeBearSetup();
   }
}

//+------------------------------------------------------------------+
// Same rule as GetClosedAnchorTrend, against any candidate anchor. Used by
// the survey only; it never affects a trading decision.
// Returns 1 supported, 0 not supported, -1 data not available yet. A slower
// anchor warms up later, so an unavailable bar must not count against it.
int SurveyAnchorSupports(int index, datetime tradeBarOpenTime,
                         int desiredTrend)
{
   if(surveyEMAHandle[index] == INVALID_HANDLE ||
      surveyATRHandle[index] == INVALID_HANDLE)
      return -1;

   ENUM_TIMEFRAMES tf = surveyTF[index];
   datetime closeTime = tradeBarOpenTime + tradeTFSeconds;
   int containingShift = iBarShift(_Symbol, tf, closeTime, false);
   if(containingShift < 0)
      return -1;

   int closedShift = containingShift + 1;
   double ema = 0.0, atr = 0.0, olderEMA = 0.0;
   if(!BufferValue(surveyEMAHandle[index], closedShift, ema) ||
      !BufferValue(surveyATRHandle[index], closedShift, atr))
      return -1;

   double close = iClose(_Symbol, tf, closedShift);
   if(close == 0.0)
      return -1;

   double band = atr * InpAnchorNeutralZoneATR;
   int trend = close > ema + band ? ANCHOR_TREND_BULL
             : (close < ema - band ? ANCHOR_TREND_BEAR : ANCHOR_TREND_NEUTRAL);
   if(trend != desiredTrend)
      return 0;

   if(!InpRequireAnchorEMASlope)
      return 1;
   if(!BufferValue(surveyEMAHandle[index],
                   closedShift + InpAnchorEMASlopeLookbackBars, olderEMA))
      return -1;
   bool slopeOk = desiredTrend == ANCHOR_TREND_BULL ? ema > olderEMA
                                                    : ema < olderEMA;
   return slopeOk ? 1 : 0;
}

//+------------------------------------------------------------------+
void SurveyAnchors(datetime tradeBarOpenTime, int desiredTrend)
{
   if(!InpLogFilterStats)
      return;
   for(int index = 0; index < ER_SURVEY_COUNT; index++)
   {
      int verdict = SurveyAnchorSupports(index, tradeBarOpenTime, desiredTrend);
      if(verdict < 0)
         continue;
      surveyTotal[index]++;
      if(verdict == 1)
         surveyPass[index]++;
   }
}

//+------------------------------------------------------------------+
int GetClosedAnchorTrend(datetime tradeBarOpenTime)
{
   // At this candle's close, use only an anchor candle that had fully
   // completed. The +1 shift prevents any higher-timeframe look-ahead.
   datetime closeTime = tradeBarOpenTime + tradeTFSeconds;
   int containingShift = iBarShift(_Symbol, anchorTF, closeTime, false);
   if(containingShift < 0)
      return ANCHOR_TREND_NEUTRAL;

   int closedAnchorShift = containingShift + 1;
   double anchorEMA = 0.0;
   double anchorATR = 0.0;
   if(!BufferValue(anchorEMAHandle, closedAnchorShift, anchorEMA) ||
      !BufferValue(anchorATRHandle, closedAnchorShift, anchorATR))
      return ANCHOR_TREND_NEUTRAL;

   double anchorClose = iClose(_Symbol, anchorTF, closedAnchorShift);
   if(anchorClose == 0.0)
      return ANCHOR_TREND_NEUTRAL;

   double band = anchorATR * InpAnchorNeutralZoneATR;
   if(anchorClose > anchorEMA + band)
      return ANCHOR_TREND_BULL;
   if(anchorClose < anchorEMA - band)
      return ANCHOR_TREND_BEAR;
   return ANCHOR_TREND_NEUTRAL;
}

//+------------------------------------------------------------------+
bool AnchorSlopeSupports(datetime tradeBarOpenTime, int desiredTrend)
{
   if(!InpRequireAnchorEMASlope)
      return true;

   datetime closeTime = tradeBarOpenTime + tradeTFSeconds;
   int containingShift = iBarShift(_Symbol, anchorTF, closeTime, false);
   if(containingShift < 0)
      return false;

   int recentClosedShift = containingShift + 1;
   double recentEMA = 0.0;
   double olderEMA = 0.0;
   if(!BufferValue(anchorEMAHandle, recentClosedShift, recentEMA) ||
      !BufferValue(anchorEMAHandle,
                   recentClosedShift + InpAnchorEMASlopeLookbackBars,
                   olderEMA))
      return false;

   if(desiredTrend == ANCHOR_TREND_BULL)
      return recentEMA > olderEMA;
   if(desiredTrend == ANCHOR_TREND_BEAR)
      return recentEMA < olderEMA;
   return false;
}

//+------------------------------------------------------------------+
bool TradeSlopeSupports(int closedShift, int desiredTrend)
{
   if(!InpRequireTradeEMASlope)
      return true;

   double recentEMA = 0.0;
   double olderEMA = 0.0;
   if(!BufferValue(tradeEMAHandle, closedShift, recentEMA) ||
      !BufferValue(tradeEMAHandle,
                   closedShift + InpTradeEMASlopeLookbackBars,
                   olderEMA))
      return false;

   if(desiredTrend == ANCHOR_TREND_BULL)
      return recentEMA > olderEMA;
   if(desiredTrend == ANCHOR_TREND_BEAR)
      return recentEMA < olderEMA;
   return false;
}

//+------------------------------------------------------------------+
// Session hours are interpreted in GMT via InpServerGMTOffsetHours so the
// window keeps tracking the US cash session when the server clock shifts.
bool BarCloseGMT(datetime closedBarOpenTime, MqlDateTime &parts)
{
   long evaluation = (long)closedBarOpenTime + (long)tradeTFSeconds -
                     (long)InpServerGMTOffsetHours * 3600;
   if(evaluation <= 0)
      return false;
   TimeToStruct((datetime)evaluation, parts);
   return true;
}

//+------------------------------------------------------------------+
bool InSessionWindow(int hour)
{
   return InpEntryStartHour < InpEntryEndHour
          ? (hour >= InpEntryStartHour && hour < InpEntryEndHour)
          : (hour >= InpEntryStartHour || hour < InpEntryEndHour);
}

//+------------------------------------------------------------------+
bool EntryWindowAllowed(datetime closedBarOpenTime)
{
   if(!InpUseEntrySession)
      return true;

   MqlDateTime parts;
   if(!BarCloseGMT(closedBarOpenTime, parts) || !InSessionWindow(parts.hour))
      return false;

   // The first minutes after the cash open are the widest-spread, most
   // reversal-prone part of the day for an index scalp.
   if(InpSkipMinutesAfterSessionOpen > 0 && parts.hour == InpEntryStartHour &&
      parts.min < InpSkipMinutesAfterSessionOpen)
      return false;
   return true;
}

//+------------------------------------------------------------------+
// Flatten once the traded session is over. This runs before any other
// management so the position cannot survive the window on a stop or target.
void FlattenAfterSessionEnd(datetime closedBarOpenTime)
{
   if(!InpUseEntrySession || !InpCloseAtSessionEnd)
      return;

   MqlDateTime parts;
   if(!BarCloseGMT(closedBarOpenTime, parts) || InSessionWindow(parts.hour))
      return;

   if(HasManagedPosition(POSITION_TYPE_BUY))
      CloseManagedPositions(POSITION_TYPE_BUY, "session end");
   if(HasManagedPosition(POSITION_TYPE_SELL))
      CloseManagedPositions(POSITION_TYPE_SELL, "session end");
}

//+------------------------------------------------------------------+
// Per-day trade count and realized-loss brake. Both use realized balance so
// an unrealized intrabar swing cannot switch them on or off.
void RollTradingDay(datetime closedBarOpenTime)
{
   MqlDateTime parts;
   TimeToStruct(closedBarOpenTime, parts);
   int dayKey = parts.year * 10000 + parts.mon * 100 + parts.day;
   if(dayKey == currentTradingDay)
      return;

   currentTradingDay = dayKey;
   dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   tradesToday = 0;
   dailyLossStopHit = false;
}

//+------------------------------------------------------------------+
bool TradingBudgetAllows()
{
   if(InpMaxTradesPerDay > 0 && tradesToday >= InpMaxTradesPerDay)
      return false;

   if(InpDailyLossStopPercent <= 0.0 || dayStartBalance <= 0.0)
      return true;

   double loss = dayStartBalance - AccountInfoDouble(ACCOUNT_BALANCE);
   double limit = dayStartBalance * InpDailyLossStopPercent / 100.0;
   if(loss >= limit)
   {
      if(!dailyLossStopHit)
      {
         dailyLossStopHit = true;
         PrintFormat("Daily loss stop reached (%.2f of %.2f). No further "
                     "entries today.", loss, limit);
      }
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
// Called once at the end of every completed bar. A position that vanished
// since the previous bar closed (strategy exit, stop, or target) starts the
// cooldown; a reversal that re-enters on the same bar does not.
void UpdateCooldownState(datetime closedBarOpenTime)
{
   bool hasPosition = HasManagedPosition(POSITION_TYPE_BUY) ||
                      HasManagedPosition(POSITION_TYPE_SELL);
   if(InpCooldownBars > 0 && hadManagedPosition && !hasPosition)
      cooldownReleaseTime = (datetime)((long)closedBarOpenTime +
                                       (long)InpCooldownBars * tradeTFSeconds);
   hadManagedPosition = hasPosition;
}

//+------------------------------------------------------------------+
bool VolatilityAllowsEntry(double currentATR)
{
   if(InpMinATRPoints <= 0.0)
      return true;
   if(currentATR <= 0.0 || _Point <= 0.0)
      return false;
   return currentATR / _Point >= InpMinATRPoints;
}

//+------------------------------------------------------------------+
// The scalping gate the M30 model never needed: refuse a setup whose cost is
// a large share of its own stop distance.
bool SpreadAllowsEntry(double currentATR)
{
   if(InpMaxSpreadATR <= 0.0)
      return true;
   if(currentATR <= 0.0)
      return false;

   double spread = CurrentSpreadPrice();
   if(spread <= 0.0)
      return true;
   if(spread <= currentATR * InpMaxSpreadATR)
      return true;

   PrintFormat("Entry skipped: spread %.5f exceeds %.2f x ATR (%.5f).",
               spread, InpMaxSpreadATR, currentATR);
   return false;
}

//+------------------------------------------------------------------+
double CurrentSpreadPrice()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask > 0.0 && bid > 0.0 && ask > bid)
      return ask - bid;
   return (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
}

//+------------------------------------------------------------------+
void ApplyBreakEvenAtClosedBar(int closedShift)
{
   if(InpBreakEvenAtR <= 0.0 || InpTakeProfitR <= 0.0)
      return;

   double barHigh = iHigh(_Symbol, tradeTF, closedShift);
   double barLow = iLow(_Symbol, tradeTF, closedShift);
   double triggerR = InpBreakEvenAtR;
   if(InpAdaptiveProtection)
   {
      datetime evaluation = iTime(_Symbol, tradeTF, closedShift) + tradeTFSeconds;
      int anchorShift = iBarShift(_Symbol, anchorTF, evaluation, false);
      double adx = 0.0;
      if(anchorShift < 0 || !BufferValue(anchorADXHandle, anchorShift + 1, adx))
         return;
      if(adx < InpProtectionADXThreshold)
         triggerR = InpWeakTrendBreakEvenAtR;
   }
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   for(int index = PositionsTotal() - 1; index >= 0; index--)
   {
      ulong ticket = PositionGetTicket(index);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      ENUM_POSITION_TYPE positionType =
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if((positionType == POSITION_TYPE_BUY && !InpProtectLongs) ||
         (positionType == POSITION_TYPE_SELL && !InpProtectShorts))
         continue;
      double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      double takeProfit = PositionGetDouble(POSITION_TP);
      if(entryPrice <= 0.0 || takeProfit <= 0.0)
         continue;

      double initialRisk = MathAbs(takeProfit - entryPrice) / InpTakeProfitR;
      if(initialRisk <= 0.0)
         continue;

      bool trigger = positionType == POSITION_TYPE_BUY
                     ? barHigh >= entryPrice + initialRisk * triggerR
                     : barLow <= entryPrice - initialRisk * triggerR;
      double newSL = positionType == POSITION_TYPE_BUY
                     ? entryPrice + initialRisk * InpBreakEvenOffsetR
                     : entryPrice - initialRisk * InpBreakEvenOffsetR;
      bool improves = positionType == POSITION_TYPE_BUY
                      ? (currentSL == 0.0 || newSL > currentSL) &&
                        currentSL < entryPrice
                      : (currentSL == 0.0 || newSL < currentSL) &&
                        currentSL > entryPrice;
      if(!trigger || !improves)
         continue;

      newSL = NormalizeDouble(newSL, digits);
      if(!trade.PositionModify(ticket, newSL, takeProfit) ||
         (trade.ResultRetcode() != TRADE_RETCODE_DONE &&
          trade.ResultRetcode() != TRADE_RETCODE_NO_CHANGES))
      {
         PrintFormat("Break-even modification failed for position %I64u. "
                     "Retcode %u: %s", ticket, trade.ResultRetcode(),
                     trade.ResultRetcodeDescription());
      }
      else
      {
         PrintFormat("Position %I64u protected at %.5f after %.2fR.",
                     ticket, newSL, triggerR);
      }
   }
}

//+------------------------------------------------------------------+
bool ExecuteConfirmedDirection(ENUM_POSITION_TYPE desiredType,
                               datetime signalBarTime,
                               double signalATR,
                               double setupPivotPrice)
{
   // Especially on netting accounts, an order could alter another strategy's
   // position. Do not enter while a foreign/manual position uses this symbol.
   for(int index = PositionsTotal() - 1; index >= 0; index--)
   {
      if(PositionGetTicket(index) != 0 &&
         PositionGetString(POSITION_SYMBOL) == _Symbol &&
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
      {
         Print("Entry skipped: a manual/other-EA position exists on this symbol.");
         return false;
      }
   }
   ENUM_POSITION_TYPE oppositeType = desiredType == POSITION_TYPE_BUY
                                     ? POSITION_TYPE_SELL
                                     : POSITION_TYPE_BUY;

   // Usually the potential-exit route has already closed the position. If it
   // did not occur, an opposite confirmed entry remains the final fallback
   // reversal so a position cannot survive a fully confirmed regime change.
   if(!CloseManagedPositions(oppositeType, "opposite confirmed entry"))
   {
      Print("Reversal aborted because the opposite position could not close.");
      return false;
   }

   // One position at a time. A repeated signal in the current direction is
   // recorded in the journal but does not pyramid or reset the position.
   if(HasManagedPosition(desiredType))
   {
      Print("Confirmed signal matches the active position; no additional "
            "position opened.");
      return false;
   }

   double entryPrice = desiredType == POSITION_TYPE_BUY
                       ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                       : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double stopLoss = 0.0;
   double takeProfit = 0.0;
   if(entryPrice <= 0.0 ||
      !BuildOrderLevels(desiredType, entryPrice, signalATR,
                        setupPivotPrice, stopLoss, takeProfit))
   {
      Print("Entry order failed: protective order levels could not be built.");
      return false;
   }

   double tradeVolume = CalculateTradeVolume(desiredType, entryPrice,
                                             stopLoss, signalBarTime);
   if(tradeVolume <= 0.0)
   {
      Print("Entry skipped: the requested risk is below the symbol's minimum "
            "trade size or the volume is otherwise invalid.");
      return false;
   }

   bool sent = false;
   if(desiredType == POSITION_TYPE_BUY)
      sent = trade.Buy(tradeVolume, _Symbol, 0.0, stopLoss, takeProfit,
                       InpTradeComment);
   else
      sent = trade.Sell(tradeVolume, _Symbol, 0.0, stopLoss, takeProfit,
                        InpTradeComment);

   if(!sent || (trade.ResultRetcode() != TRADE_RETCODE_DONE &&
                trade.ResultRetcode() != TRADE_RETCODE_DONE_PARTIAL))
   {
      PrintFormat("Entry order failed for signal bar %s. Retcode %u: %s",
                  TimeToString(signalBarTime, TIME_DATE | TIME_MINUTES),
                  trade.ResultRetcode(), trade.ResultRetcodeDescription());
      return false;
   }

   // Cache this trade's money risk so every exit can be reported in R.
   double filledPrice = trade.ResultPrice() > 0.0 ? trade.ResultPrice()
                                                  : entryPrice;
   double riskAtEntry = 0.0;
   ENUM_ORDER_TYPE filledType = desiredType == POSITION_TYPE_BUY
                                ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(OrderCalcProfit(filledType, _Symbol, tradeVolume, filledPrice,
                      stopLoss, riskAtEntry))
      openRiskMoney = MathAbs(riskAtEntry);
   else
      openRiskMoney = 0.0;
   openStopWasTrailed = false;
   pendingCloseReason = "";

   PrintFormat("%s opened: order %I64u, deal %I64u, price %.5f, "
               "SL %.5f, TP %.5f, volume %.2f",
               desiredType == POSITION_TYPE_BUY ? "BUY" : "SELL",
               trade.ResultOrder(), trade.ResultDeal(), trade.ResultPrice(),
               stopLoss, takeProfit, tradeVolume);
   return true;
}

//+------------------------------------------------------------------+
bool BuildOrderLevels(ENUM_POSITION_TYPE desiredType,
                      double entryPrice,
                      double signalATR,
                      double setupPivotPrice,
                      double &stopLoss,
                      double &takeProfit)
{
   stopLoss = 0.0;
   takeProfit = 0.0;
   if(InpStopMode == ER_STOP_NONE)
      return !InpUseRiskBasedVolume;
   if(signalATR <= 0.0)
      return false;

   double atrDistance = signalATR * InpStopATRMultiplier;
   double atrStop = desiredType == POSITION_TYPE_BUY
                    ? entryPrice - atrDistance
                    : entryPrice + atrDistance;
   double pivotStop = desiredType == POSITION_TYPE_BUY
                      ? setupPivotPrice - signalATR * InpPivotStopBufferATR
                      : setupPivotPrice + signalATR * InpPivotStopBufferATR;
   bool validPivot = setupPivotPrice > 0.0 &&
      ((desiredType == POSITION_TYPE_BUY && pivotStop < entryPrice) ||
       (desiredType == POSITION_TYPE_SELL && pivotStop > entryPrice));

   if(InpStopMode == ER_STOP_TF_ATR || !validPivot)
      stopLoss = atrStop;
   else if(InpStopMode == ER_STOP_SETUP_PIVOT)
      stopLoss = pivotStop;
   else if(desiredType == POSITION_TYPE_BUY)
      stopLoss = MathMax(atrStop, pivotStop);
   else
      stopLoss = MathMin(atrStop, pivotStop);

   // A scalp stop only a spread or two wide is noise, not risk. Widen it so
   // the cost stays a known fraction of R; sizing then shrinks accordingly.
   double minDistance =
      (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   if(InpMinStopSpreadMultiple > 0.0)
      minDistance = MathMax(minDistance,
                            CurrentSpreadPrice() * InpMinStopSpreadMultiple);
   if(minDistance > 0.0)
   {
      if(desiredType == POSITION_TYPE_BUY &&
         entryPrice - stopLoss < minDistance)
         stopLoss = entryPrice - minDistance;
      if(desiredType == POSITION_TYPE_SELL &&
         stopLoss - entryPrice < minDistance)
         stopLoss = entryPrice + minDistance;
   }

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   stopLoss = NormalizeDouble(stopLoss, digits);
   double riskDistance = MathAbs(entryPrice - stopLoss);
   if(riskDistance <= 0.0)
      return false;

   if(InpTakeProfitR > 0.0)
   {
      takeProfit = desiredType == POSITION_TYPE_BUY
                   ? entryPrice + riskDistance * InpTakeProfitR
                   : entryPrice - riskDistance * InpTakeProfitR;
      takeProfit = NormalizeDouble(takeProfit, digits);
   }
   return true;
}

//+------------------------------------------------------------------+
// Realized account balance only: unrealized intrabar peaks cannot activate it.
// Hysteresis avoids repeated risk changes around the start threshold.
void UpdateRiskThrottle()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(balance <= 0.0)
      return;
   riskBalancePeak = MathMax(riskBalancePeak, balance);
   if(!InpUseDrawdownThrottle || riskBalancePeak <= 0.0)
   {
      drawdownThrottled = false;
      return;
   }
   double dd = 100.0 * (riskBalancePeak - balance) / riskBalancePeak;
   bool previous = drawdownThrottled;
   if(!drawdownThrottled && dd >= InpDrawdownThrottleStartPct)
      drawdownThrottled = true;
   else if(drawdownThrottled && dd <= InpDrawdownThrottleRecoveryPct)
      drawdownThrottled = false;
   if(previous != drawdownThrottled)
      PrintFormat("Risk throttle %s: realized balance drawdown %.2f%%",
                  drawdownThrottled ? "ON" : "OFF", dd);
}

//+------------------------------------------------------------------+
double CalculateTradeVolume(ENUM_POSITION_TYPE desiredType,
                            double entryPrice,
                            double stopLoss,
                            datetime signalBarTime)
{
   if(!InpUseRiskBasedVolume)
      return NormalizeTradeVolume(InpLots, true);
   if(stopLoss <= 0.0)
      return 0.0;

   double oneLotResult = 0.0;
   ENUM_ORDER_TYPE orderType = desiredType == POSITION_TYPE_BUY
                               ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(!OrderCalcProfit(orderType, _Symbol, 1.0, entryPrice,
                       stopLoss, oneLotResult))
      return 0.0;

   double riskPerLot = MathAbs(oneLotResult);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double sizingCapital = equity <= InpRiskCapitalBase ? equity :
      InpRiskCapitalBase + (equity - InpRiskCapitalBase) * InpProfitReinvestmentFraction;
   if(InpMaxSizingCapitalMultiple > 0.0)
      sizingCapital = MathMin(sizingCapital,
                              InpRiskCapitalBase * InpMaxSizingCapitalMultiple);
   UpdateRiskThrottle(); // Include any opposite-position close just executed.
   double riskBudget = sizingCapital * InpRiskPercent / 100.0;
   if(drawdownThrottled)
      riskBudget *= InpDrawdownThrottleMultiplier;
   if(desiredType == POSITION_TYPE_SELL)
   {
      bool useReducedRisk = true;
      if(InpScaleOnlyWeakShorts)
      {
         datetime evaluation = signalBarTime + tradeTFSeconds;
         int shift = iBarShift(_Symbol, anchorTF, evaluation, false);
         double adx = 0.0, plusDI[1], minusDI[1];
         if(shift < 0 || !BufferValue(anchorADXHandle, shift + 1, adx) ||
            CopyBuffer(anchorADXHandle, 1, shift + 1, 1, plusDI) != 1 ||
            CopyBuffer(anchorADXHandle, 2, shift + 1, 1, minusDI) != 1)
            return 0.0;
         useReducedRisk = !(adx >= InpFullShortRiskMinADX && minusDI[0] > plusDI[0]);
      }
      if(useReducedRisk)
         riskBudget *= InpShortRiskMultiplier;
   }
   if(riskPerLot <= 0.0 || riskBudget <= 0.0)
      return 0.0;

   double volume = NormalizeTradeVolume(riskBudget / riskPerLot, false);
   if(!InpCapVolumeToMargin || volume <= 0.0)
      return volume;

   double marginBudget = AccountInfoDouble(ACCOUNT_MARGIN_FREE) *
                          InpMaxFreeMarginUsePercent / 100.0;
   double neededMargin = 0.0;
   if(marginBudget <= 0.0 ||
      !OrderCalcMargin(orderType, _Symbol, volume, entryPrice, neededMargin))
      return 0.0;
   if(neededMargin <= marginBudget)
      return volume;

   // Search broker volume steps; never exceed the risk or margin budget.
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0)
      return 0.0;
   long low = 0, high = (long)MathFloor(volume / step + 1e-9);
   while(low < high)
   {
      long middle = (low + high + 1) / 2;
      if(!OrderCalcMargin(orderType, _Symbol, middle * step,
                          entryPrice, neededMargin))
         return 0.0;
      if(neededMargin <= marginBudget)
         low = middle;
      else
         high = middle - 1;
   }
   double capped = NormalizeTradeVolume(low * step, false);
   PrintFormat("Margin cap: requested %.2f lots, affordable %.2f lots.",
               volume, capped);
   return capped;
}

//+------------------------------------------------------------------+
// Trail using a completed candle's CLOSE. Do not retrospectively assume
// an intrabar high/low could have moved the stop before its reversal.
void ApplyTrailingAtClosedBar(int closedShift)
{
   if(InpTrailStartR <= 0.0 || InpTakeProfitR <= 0.0)
      return;
   double barClose = iClose(_Symbol, tradeTF, closedShift);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double minDistance = MathMax(
      (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL),
      (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL)) * _Point;
   if(tickSize <= 0.0 || barClose <= 0.0)
      return;
   for(int index = PositionsTotal() - 1; index >= 0; index--)
   {
      ulong ticket = PositionGetTicket(index);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      bool isBuy = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
      if((isBuy && !InpProtectLongs) || (!isBuy && !InpProtectShorts))
         continue;
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double tp = PositionGetDouble(POSITION_TP);
      double sl = PositionGetDouble(POSITION_SL);
      double initialRisk = MathAbs(tp - entry) / InpTakeProfitR;
      if(tp <= 0.0 || initialRisk <= 0.0)
         continue;
      double gain = isBuy ? barClose - entry : entry - barClose;
      if(gain < initialRisk * InpTrailStartR)
         continue;
      double newSL = isBuy ? barClose - initialRisk * InpTrailDistanceR
                           : barClose + initialRisk * InpTrailDistanceR;
      newSL = (isBuy ? MathFloor(newSL / tickSize) : MathCeil(newSL / tickSize)) * tickSize;
      double current = SymbolInfoDouble(_Symbol, isBuy ? SYMBOL_BID : SYMBOL_ASK);
      bool valid = isBuy ? newSL < current - minDistance && (sl == 0.0 || newSL > sl + tickSize / 2)
                         : newSL > current + minDistance && (sl == 0.0 || newSL < sl - tickSize / 2);
      if(!valid)
         continue;
      if(!trade.PositionModify(ticket, newSL, tp) ||
         (trade.ResultRetcode() != TRADE_RETCODE_DONE &&
          trade.ResultRetcode() != TRADE_RETCODE_NO_CHANGES))
         PrintFormat("Trailing stop failed: %u %s", trade.ResultRetcode(),
                     trade.ResultRetcodeDescription());
      else
         openStopWasTrailed = true;
   }
}

//+------------------------------------------------------------------+
void CutLosingAnchorReversals(int closedShift, int anchorTrend)
{
   if(!InpCutLosingAnchorReversals || InpTakeProfitR <= 0.0)
      return;
   double barClose = iClose(_Symbol, tradeTF, closedShift);
   for(int index = PositionsTotal() - 1; index >= 0; index--)
   {
      ulong ticket = PositionGetTicket(index);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      bool isBuy = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
      if(isBuy && !InpCutLosingLongs)
         continue;
      if(anchorTrend != (isBuy ? ANCHOR_TREND_BEAR : ANCHOR_TREND_BULL))
         continue;
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double tp = PositionGetDouble(POSITION_TP);
      double initialRisk = MathAbs(tp - entry) / InpTakeProfitR;
      if(tp <= 0.0 || initialRisk <= 0.0)
         continue;
      double loss = isBuy ? entry - barClose : barClose - entry;
      if(loss >= initialRisk * InpAnchorReversalLossR)
         CloseManagedPositions(isBuy ? POSITION_TYPE_BUY : POSITION_TYPE_SELL,
                               "losing position plus confirmed anchor reversal");
   }
}

//+------------------------------------------------------------------+
bool CloseManagedPositions(ENUM_POSITION_TYPE typeToClose,
                           string closureReason)
{
   // Read back by OnTradeTransaction to attribute the exit. A broker-side
   // stop or target leaves this empty and is classified from the result.
   if(HasManagedPosition(typeToClose))
      pendingCloseReason = closureReason;
   bool allClosed = true;
   for(int index = PositionsTotal() - 1; index >= 0; index--)
   {
      ulong ticket = PositionGetTicket(index);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber ||
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != typeToClose)
         continue;

      if(!trade.PositionClose(ticket, InpDeviationPoints) ||
         trade.ResultRetcode() != TRADE_RETCODE_DONE ||
         PositionSelectByTicket(ticket))
      {
         PrintFormat("Could not close position %I64u. Retcode %u: %s",
                     ticket, trade.ResultRetcode(),
                     trade.ResultRetcodeDescription());
         allClosed = false;
      }
      else
      {
         PrintFormat("Position %I64u closed by %s.", ticket, closureReason);
      }
   }
   return allClosed;
}

//+------------------------------------------------------------------+
void RecordExit(string reason, double rMultiple)
{
   int slot = -1;
   for(int index = 0; index < exitReasonsUsed; index++)
      if(exitReasonName[index] == reason)
      {
         slot = index;
         break;
      }
   if(slot < 0)
   {
      if(exitReasonsUsed >= ER_MAX_EXIT_REASONS)
         return;
      slot = exitReasonsUsed++;
      exitReasonName[slot] = reason;
   }
   exitReasonCount[slot]++;
   exitReasonSumR[slot] += rMultiple;
}

//+------------------------------------------------------------------+
// Every closing deal is attributed to an exit route and converted to R, so
// the journal shows where the money actually goes rather than only a total.
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal == 0)
      return;
   if(!HistoryDealSelect(trans.deal))
      return;
   if((ulong)HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagicNumber ||
      HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol ||
      (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY) !=
      DEAL_ENTRY_OUT)
      return;

   double netResult = HistoryDealGetDouble(trans.deal, DEAL_PROFIT) +
                      HistoryDealGetDouble(trans.deal, DEAL_SWAP) +
                      HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);

   string reason = pendingCloseReason;
   if(reason == "")
   {
      if(openStopWasTrailed && netResult > 0.0)
         reason = "trailing stop";
      else if(netResult > 0.0)
         reason = "take profit hit";
      else
         reason = "stop loss hit";
   }

   double rMultiple = openRiskMoney > 0.0 ? netResult / openRiskMoney : 0.0;
   RecordExit(reason, rMultiple);

   pendingCloseReason = "";
   openStopWasTrailed = false;
   openRiskMoney = 0.0;
}

//+------------------------------------------------------------------+
bool HasManagedPosition(ENUM_POSITION_TYPE positionType)
{
   for(int index = PositionsTotal() - 1; index >= 0; index--)
   {
      ulong ticket = PositionGetTicket(index);
      if(ticket != 0 && PositionGetString(POSITION_SYMBOL) == _Symbol &&
         (ulong)PositionGetInteger(POSITION_MAGIC) == InpMagicNumber &&
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == positionType)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
int ManagedPositionHeldBars(ENUM_POSITION_TYPE positionType,
                            datetime closedBarOpenTime)
{
   datetime evaluationTime = closedBarOpenTime + tradeTFSeconds;
   for(int index = PositionsTotal() - 1; index >= 0; index--)
   {
      ulong ticket = PositionGetTicket(index);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber ||
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != positionType)
         continue;

      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      if(opened <= 0 || evaluationTime <= opened)
         return 0;
      return (int)((evaluationTime - opened) / tradeTFSeconds);
   }
   return 0;
}

//+------------------------------------------------------------------+
bool IsPivotLow(int shift)
{
   double candidate = iLow(_Symbol, tradeTF, shift);
   if(candidate == 0.0)
      return false;

   for(int offset = 1; offset <= InpPivotLeftBars; offset++)
      if(candidate >= iLow(_Symbol, tradeTF, shift + offset))
         return false;

   for(int offset = 1; offset <= InpPivotRightBars; offset++)
      if(candidate >= iLow(_Symbol, tradeTF, shift - offset))
         return false;

   return true;
}

//+------------------------------------------------------------------+
bool IsPivotHigh(int shift)
{
   double candidate = iHigh(_Symbol, tradeTF, shift);
   if(candidate == 0.0)
      return false;

   for(int offset = 1; offset <= InpPivotLeftBars; offset++)
      if(candidate <= iHigh(_Symbol, tradeTF, shift + offset))
         return false;

   for(int offset = 1; offset <= InpPivotRightBars; offset++)
      if(candidate <= iHigh(_Symbol, tradeTF, shift - offset))
         return false;

   return true;
}

//+------------------------------------------------------------------+
bool BufferValue(int handle, int shift, double &value)
{
   double values[1];
   ResetLastError();
   if(CopyBuffer(handle, 0, shift, 1, values) != 1 ||
      values[0] == EMPTY_VALUE)
      return false;

   value = values[0];
   return true;
}

//+------------------------------------------------------------------+
double NormalizeTradeVolume(double requestedVolume,
                            bool allowMinimumClamp)
{
   double minimum = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maximum = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(minimum <= 0.0 || maximum < minimum || step <= 0.0)
      return 0.0;

   if(!allowMinimumClamp && requestedVolume + 1e-9 < minimum)
      return 0.0;

   double volume = MathMax(minimum, MathMin(maximum, requestedVolume));
   volume = MathFloor(volume / step + 1e-9) * step;

   int volumeDigits = 0;
   double scaledStep = step;
   while(volumeDigits < 8 &&
         MathAbs(scaledStep - MathRound(scaledStep)) > 1e-9)
   {
      scaledStep *= 10.0;
      volumeDigits++;
   }
   return NormalizeDouble(volume, volumeDigits);
}

//+------------------------------------------------------------------+
bool BarExists(ENUM_TIMEFRAMES timeframe, int shift)
{
   return shift >= 0 && iTime(_Symbol, timeframe, shift) != 0;
}

//+------------------------------------------------------------------+
int BarsBetween(datetime olderPivotTime, int newerPivotShift)
{
   if(olderPivotTime == 0)
      return -1;

   int olderShift = iBarShift(_Symbol, tradeTF, olderPivotTime, true);
   if(olderShift < 0)
      return -1;
   return olderShift - newerPivotShift;
}

//+------------------------------------------------------------------+
int BarsElapsed(datetime confirmationTime, int currentShift)
{
   if(confirmationTime == 0)
      return 0;

   int confirmationShift = iBarShift(_Symbol, tradeTF, confirmationTime, true);
   if(confirmationShift < 0)
      return 0;
   return confirmationShift - currentShift;
}

//+------------------------------------------------------------------+
bool BarIsAtOrAfter(datetime barTime, datetime referenceTime)
{
   return referenceTime != 0 && barTime >= referenceTime;
}

//+------------------------------------------------------------------+
void ConsumeBullSetup()
{
   bullSetupActive = false;
   bullSetupConfirmedTime = 0;
   bullSetupPivotPrice = 0.0;
   bullEntryWindowExpired = false;
   bullExitSignalIssued = false;
}

//+------------------------------------------------------------------+
void ConsumeBearSetup()
{
   bearSetupActive = false;
   bearSetupConfirmedTime = 0;
   bearSetupPivotPrice = 0.0;
   bearEntryWindowExpired = false;
   bearExitSignalIssued = false;
}
//+------------------------------------------------------------------+
