//+------------------------------------------------------------------+
//|                                              SnapScalp_MR_v2.mq5 |
//|   Mean-reversion intraday scalper for FX majors                  |
//|   GBPUSD / EURUSD / USDJPY  -  M5 (M15 acceptable)               |
//|                                                                  |
//|   v2 versus SnapScalp_MR v1                                      |
//|                                                                  |
//|   Profit side                                                    |
//|     * Second entry model: Bollinger band snap-back (a bar closes |
//|       outside the band, the next bar closes back inside).        |
//|       Raises signal frequency without touching the risk model.   |
//|     * Scale-out: part of the position is banked at a near        |
//|       target, the runner is worked to the mean with a trail.     |
//|     * Mean-revert target is re-anchored to the live EMA each     |
//|       bar, and the trade is closed at market if price crosses    |
//|       the mean between TP updates.                               |
//|     * Anti-martingale sizing: risk is cut after a loss streak    |
//|       and modestly raised after a win streak. Never the reverse. |
//|     * Symbol presets for the three pairs (spread cap, sessions). |
//|     * Lot ceiling raised so size can scale with equity.          |
//|                                                                  |
//|   Loss side                                                      |
//|     * Volatility band filter: no entries when ATR is far above   |
//|       or below its slow average (news spikes / dead tape).       |
//|     * Optional higher-timeframe bias filter.                     |
//|     * Time stop no longer dumps a working trade: it closes only  |
//|       trades that have gone nowhere, and hard-closes at 2x.      |
//|     * Break-even retries until the broker accepts it.            |
//|     * Position state survives a restart (terminal globals).      |
//|     * No re-entry on the bar a trade just closed on.             |
//|     * Calendar-correct daily / weekly guard anchors.             |
//|     * Partial closes are not mis-counted as finished trades.     |
//|                                                                  |
//|   Unchanged, deliberately                                        |
//|     * Hard stop on every trade. One position. No grid. No        |
//|       averaging. Size from stop distance, never from history.    |
//|     * Three independent equity guards that can flatten and       |
//|       disable the EA.                                            |
//+------------------------------------------------------------------+
#property copyright "Private use"
#property version   "2.00"
#property description "Session-filtered mean reversion scalper for GBPUSD / EURUSD / USDJPY."
#property description "Two bounded entry models, scale-out exits, anti-martingale sizing, layered equity guards."

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Enumerations                                                     |
//+------------------------------------------------------------------+
enum ENUM_REGIME_MODE
  {
   REGIME_RANGE_ONLY = 0,   // Only trade when ADX below threshold
   REGIME_ANY        = 1    // Ignore the regime filter
  };

enum ENUM_TP_MODE
  {
   TP_ATR_MULTIPLE = 0,     // Fixed ATR multiple
   TP_MEAN_REVERT  = 1      // Target the anchor EMA (re-anchored each bar)
  };

enum ENUM_ENTRY_MODEL
  {
   MODEL_STRETCH = 0,       // ATR stretch + RSI wash-out only
   MODEL_BAND    = 1,       // Bollinger snap-back only
   MODEL_BOTH    = 2        // Either model may fire
  };

enum ENUM_BIAS_MODE
  {
   BIAS_NONE       = 0,     // Trade both directions
   BIAS_WITH_TREND = 1      // Only fade dips in an uptrend / rallies in a downtrend
  };

enum ENUM_SYMBOL_PRESET
  {
   PRESET_AUTO   = 0,       // Detect from chart symbol
   PRESET_GBPUSD = 1,       // GBPUSD preset
   PRESET_EURUSD = 2,       // EURUSD preset
   PRESET_USDJPY = 3,       // USDJPY preset
   PRESET_MANUAL = 4        // Use the inputs below verbatim
  };

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "=== Identity ==="
input long               InpMagic               = 770201;      // Magic number (unique per chart)
input string             InpComment             = "SnapScalp2"; // Order comment

input group "=== Symbol preset ==="
input ENUM_SYMBOL_PRESET InpPreset              = PRESET_AUTO; // Preset (overrides spread cap and session 3)

input group "=== Signal: common ==="
input ENUM_TIMEFRAMES    InpSignalTF            = PERIOD_M5;   // Signal timeframe
input ENUM_TIMEFRAMES    InpRegimeTF            = PERIOD_H1;   // Regime / bias timeframe
input ENUM_ENTRY_MODEL   InpEntryModel          = MODEL_BOTH;  // Entry model(s)
input int                InpEmaPeriod           = 20;          // Anchor EMA period (signal TF)
input int                InpAtrPeriod           = 14;          // ATR period (signal TF)
input int                InpRsiPeriod           = 2;           // RSI period (fast, signal TF)
input ENUM_REGIME_MODE   InpRegimeMode          = REGIME_RANGE_ONLY; // Regime filter mode
input int                InpAdxPeriod           = 14;          // ADX period (regime TF)
input double             InpAdxMax              = 28.0;        // Max ADX to allow mean reversion
input ENUM_BIAS_MODE     InpBiasMode            = BIAS_NONE;   // Higher-timeframe bias filter
input int                InpBiasEmaPeriod       = 200;         // Bias EMA period (regime TF)

input group "=== Signal: model A (ATR stretch) ==="
input double             InpStretchATR          = 1.30;        // Stretch from EMA required, in ATR
input double             InpRsiBuyBelow         = 8.0;         // RSI must be below this to buy
input double             InpRsiSellAbove        = 92.0;        // RSI must be above this to sell
input bool               InpRequireRejection    = true;        // Signal bar must close back toward the mean

input group "=== Signal: model B (band snap-back) ==="
input int                InpBandPeriod          = 20;          // Bollinger period (signal TF)
input double             InpBandDeviation       = 2.0;         // Bollinger deviation
input double             InpBandRsiBuyBelow     = 25.0;        // RSI of the outside bar must be below this to buy
input double             InpBandRsiSellAbove    = 75.0;        // RSI of the outside bar must be above this to sell
input double             InpBandMinStretchATR   = 0.60;        // Re-entry close must still sit this far from the EMA

input group "=== Volatility filter ==="
input int                InpAtrSlowPeriod       = 100;         // Slow ATR period (signal TF)
input double             InpMinAtrRatio         = 0.50;        // ATR / slow ATR floor (0 = off)
input double             InpMaxAtrRatio         = 2.50;        // ATR / slow ATR ceiling (0 = off)

input group "=== Exits ==="
input double             InpSLatr               = 1.60;        // Stop loss, in ATR
input ENUM_TP_MODE       InpTPMode              = TP_MEAN_REVERT; // Final target style
input double             InpTPatr               = 1.10;        // Final target, in ATR (ATR mode)
input double             InpPartialPct          = 50.0;        // Scale-out share, % of entry size (0 = off)
input double             InpPartialATR          = 0.60;        // Scale-out target, in ATR
input double             InpBreakEvenAtR        = 0.40;        // Move to BE at this R (0 = off)
input double             InpBreakEvenOffsetR    = 0.10;        // BE lock-in, in R
input double             InpTrailStartR         = 0.80;        // Start ATR trail at this R (0 = off)
input double             InpTrailATR            = 1.00;        // Trailing distance, in ATR
input int                InpMaxBarsInTrade      = 18;          // Time stop, signal-TF bars (0 = off)
input double             InpTimeStopKeepR       = 0.30;        // At time stop keep trades at or above this R

input group "=== Costs & execution ==="
input double             InpCommissionRT        = 7.00;        // Commission per 1.00 lot, round turn, acct ccy
input int                InpMaxSpreadPts        = 20;          // Reject entry above this spread (points, manual preset)
input double             InpMinTPtoCostRatio    = 2.50;        // Final target must exceed cost by this multiple
input int                InpSlippagePts         = 10;          // Max deviation (points)

input group "=== Risk ==="
input double             InpRiskPercent         = 0.75;        // Base risk per trade, % of equity
input double             InpMaxLots             = 5.00;        // Hard lot ceiling
input int                InpLossesToReduce      = 2;           // Consecutive losses before risk is cut (0 = off)
input double             InpRiskMultAfterLoss   = 0.50;        // Risk multiplier while in a loss streak
input int                InpWinsToBoost         = 3;           // Consecutive wins before risk is raised (0 = off)
input double             InpRiskMultAfterWin    = 1.25;        // Risk multiplier while in a win streak
input int                InpMaxTradesPerDay     = 8;           // Max entries per day (0 = unlimited)
input int                InpMinBarsBetweenTrades = 1;          // Bars that must pass after an exit before re-entry
input double             InpDailyLossPct        = 2.00;        // Daily loss halt, % of day-start equity
input double             InpWeeklyLossPct       = 4.00;        // Weekly loss halt, % of week-start equity
input double             InpMaxEquityDDPct      = 10.00;       // Peak-to-trough equity halt, %
input bool               InpFlattenOnGuard      = true;        // Close open trade when a guard trips
input int                InpMaxConsecLosses     = 3;           // Losses in a row before cooldown
input int                InpCooldownBars        = 24;          // Cooldown length, signal-TF bars

input group "=== Sessions (broker server time) ==="
input bool               InpUseSessions         = true;        // Restrict to sessions
input int                InpSess1StartHour      = 7;           // Session 1 start hour (London)
input int                InpSess1EndHour        = 11;          // Session 1 end hour
input int                InpSess2StartHour      = 13;          // Session 2 start hour (New York)
input int                InpSess2EndHour        = 17;          // Session 2 end hour
input bool               InpUseSession3         = false;       // Session 3 (Tokyo) - forced on by USDJPY preset
input int                InpSess3StartHour      = 2;           // Session 3 start hour
input int                InpSess3EndHour        = 6;           // Session 3 end hour
input bool               InpAvoidRollover       = true;        // Skip the rollover window
input int                InpRolloverStart       = 23;          // Rollover start hour
input int                InpRolloverEnd         = 1;           // Rollover end hour
input bool               InpTradeMonday         = true;        // Trade Monday
input bool               InpTradeFriday         = true;        // Trade Friday
input int                InpFridayStopHour      = 19;          // No new trades Friday after this hour
input bool               InpFridayFlatten       = true;        // Flatten before the weekend
input int                InpFridayCloseHour     = 20;          // Friday flatten hour

input group "=== Diagnostics ==="
input bool               InpVerboseLog          = false;       // Log rejected signals
input int                InpMinTradesForTester  = 40;          // OnTester: minimum trades to score

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CTrade   trade;

int      hEma     = INVALID_HANDLE;
int      hAtr     = INVALID_HANDLE;
int      hAtrSlow = INVALID_HANDLE;
int      hRsi     = INVALID_HANDLE;
int      hAdx     = INVALID_HANDLE;
int      hBands   = INVALID_HANDLE;
int      hBiasEma = INVALID_HANDLE;

double   g_point       = 0.0;
int      g_digits      = 0;
double   g_tickValue   = 0.0;
double   g_tickSize    = 0.0;
double   g_volMin      = 0.0;
double   g_volMax      = 0.0;
double   g_volStep     = 0.0;
int      g_volDigits   = 2;
long     g_stopsLevel  = 0;

// preset-adjusted effective settings
int      g_maxSpreadPts = 0;
bool     g_useSess3     = false;
string   g_presetName   = "manual";

datetime g_lastBarTime = 0;

// risk state
double   g_dayStartEquity  = 0.0;
double   g_weekStartEquity = 0.0;
double   g_peakEquity      = 0.0;
int      g_curDayStamp     = -1;
int      g_curWeekStamp    = -1;
int      g_tradesToday     = 0;
int      g_consecLosses    = 0;
int      g_consecWins      = 0;
int      g_cooldownLeft    = 0;
bool     g_hardHalt        = false;
bool     g_dayHalt         = false;
bool     g_weekHalt        = false;
datetime g_lastExitBar     = 0;

// open position bookkeeping
ulong    g_posTicket       = 0;
long     g_posId           = 0;
int      g_posDir          = 0;
datetime g_posOpenBar      = 0;
double   g_posRiskPoints   = 0.0;     // initial SL distance in points
double   g_posInitLots     = 0.0;     // size at entry
double   g_posPartialPts   = 0.0;     // scale-out target in points (0 = no scale-out on this trade)
bool     g_posBEDone       = false;
bool     g_posPartialDone  = false;

// indicator snapshot, read once per tick
struct SInd
  {
   double   ema1;
   double   atr1;
   double   atrSlow;
   double   rsi1;
   double   rsi2;
   double   adx1;
   double   upper1;
   double   upper2;
   double   lower1;
   double   lower2;
   double   biasEma;
   double   biasClose;
   MqlRates r1;      // last closed bar
   MqlRates r2;      // bar before it
  };

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
double PointValuePerLot()
  {
// Monetary value of one point of price movement for one full lot,
// expressed in the account currency.
   if(g_tickSize <= 0.0)
      return(0.0);
   return(g_tickValue * (g_point / g_tickSize));
  }

double CommissionInPoints()
  {
   double pvl = PointValuePerLot();
   if(pvl <= 0.0)
      return(0.0);
   return(InpCommissionRT / pvl);
  }

int SpreadPoints()
  {
   return((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
  }

double NormPrice(const double p)
  {
   return(NormalizeDouble(p, g_digits));
  }

double NormVolume(const double v)
  {
   double lots = MathFloor(v / g_volStep + 1e-9) * g_volStep;
   return(NormalizeDouble(lots, g_volDigits));
  }

int MinStopPoints()
  {
// Broker minimum stop distance plus a small buffer for slippage on modify.
   return((int)g_stopsLevel + 2);
  }

bool IsNewBar()
  {
   datetime t = (datetime)SeriesInfoInteger(_Symbol, InpSignalTF, SERIES_LASTBAR_DATE);
   if(t == 0)
      return(false);
   if(t != g_lastBarTime)
     {
      g_lastBarTime = t;
      return(true);
     }
   return(false);
  }

int DayStamp(const MqlDateTime &dt)
  {
   return(dt.year * 1000 + dt.day_of_year);
  }

int WeekStamp(const datetime now, const MqlDateTime &dt)
  {
// Number of the Monday that starts the current trading week.
   int back = (dt.day_of_week == 0 ? 6 : dt.day_of_week - 1);
   return((int)((now - (datetime)(back * 86400)) / 86400));
  }

datetime BarTimeAt(const datetime t)
  {
   int shift = iBarShift(_Symbol, InpSignalTF, t, false);
   if(shift < 0)
      return(0);
   return(iTime(_Symbol, InpSignalTF, shift));
  }

//+------------------------------------------------------------------+
//| Position state persistence (survives a terminal restart)         |
//+------------------------------------------------------------------+
string GVKey(const string what)
  {
   return(StringFormat("SS2_%s_%s_%s", IntegerToString(InpMagic), _Symbol, what));
  }

void SaveState()
  {
   GlobalVariableSet(GVKey("ticket"),  (double)g_posTicket);
   GlobalVariableSet(GVKey("risk"),    g_posRiskPoints);
   GlobalVariableSet(GVKey("init"),    g_posInitLots);
   GlobalVariableSet(GVKey("partpts"), g_posPartialPts);
   GlobalVariableSet(GVKey("partial"), (g_posPartialDone ? 1.0 : 0.0));
   GlobalVariableSet(GVKey("be"),      (g_posBEDone ? 1.0 : 0.0));
   GlobalVariableSet(GVKey("openbar"), (double)g_posOpenBar);
  }

bool LoadState(const ulong ticket)
  {
   if(!GlobalVariableCheck(GVKey("ticket")))
      return(false);
   if((ulong)GlobalVariableGet(GVKey("ticket")) != ticket)
      return(false);

   g_posRiskPoints  = GlobalVariableGet(GVKey("risk"));
   g_posInitLots    = GlobalVariableGet(GVKey("init"));
   g_posPartialPts  = GlobalVariableGet(GVKey("partpts"));
   g_posPartialDone = (GlobalVariableGet(GVKey("partial")) > 0.5);
   g_posBEDone      = (GlobalVariableGet(GVKey("be")) > 0.5);
   g_posOpenBar     = (datetime)GlobalVariableGet(GVKey("openbar"));
   return(true);
  }

void ClearState()
  {
   GlobalVariableDel(GVKey("ticket"));
   GlobalVariableDel(GVKey("risk"));
   GlobalVariableDel(GVKey("init"));
   GlobalVariableDel(GVKey("partpts"));
   GlobalVariableDel(GVKey("partial"));
   GlobalVariableDel(GVKey("be"));
   GlobalVariableDel(GVKey("openbar"));
  }

void ResetPositionGlobals()
  {
   g_posTicket      = 0;
   g_posId          = 0;
   g_posDir         = 0;
   g_posOpenBar     = 0;
   g_posRiskPoints  = 0.0;
   g_posInitLots    = 0.0;
   g_posPartialPts  = 0.0;
   g_posBEDone      = false;
   g_posPartialDone = false;
  }

//+------------------------------------------------------------------+
//| Symbol presets                                                   |
//+------------------------------------------------------------------+
void ApplyPreset()
  {
   ENUM_SYMBOL_PRESET p = InpPreset;

   if(p == PRESET_AUTO)
     {
      string s = _Symbol;
      StringToUpper(s);
      if(StringFind(s, "GBPUSD") >= 0)      p = PRESET_GBPUSD;
      else if(StringFind(s, "EURUSD") >= 0) p = PRESET_EURUSD;
      else if(StringFind(s, "USDJPY") >= 0) p = PRESET_USDJPY;
      else                                  p = PRESET_MANUAL;
     }

// Presets are expressed in 5-digit (3-digit JPY) points. Scale down
// on a 4-digit / 2-digit feed so the cap means the same in pips.
   int scale = (g_digits == 5 || g_digits == 3) ? 1 : 10;

   g_maxSpreadPts = InpMaxSpreadPts;
   g_useSess3     = InpUseSession3;
   g_presetName   = "manual";

   switch(p)
     {
      case PRESET_GBPUSD:
         g_maxSpreadPts = MathMax(1, 20 / scale);   // 2.0 pips
         g_useSess3     = false;
         g_presetName   = "GBPUSD";
         break;
      case PRESET_EURUSD:
         g_maxSpreadPts = MathMax(1, 12 / scale);   // 1.2 pips
         g_useSess3     = false;
         g_presetName   = "EURUSD";
         break;
      case PRESET_USDJPY:
         g_maxSpreadPts = MathMax(1, 15 / scale);   // 1.5 pips
         g_useSess3     = true;                     // Tokyo range is tradeable
         g_presetName   = "USDJPY";
         break;
      default:
         break;
     }
  }

//+------------------------------------------------------------------+
//| Session / calendar gate                                          |
//+------------------------------------------------------------------+
bool InHourWindow(const int hour, const int startH, const int endH)
  {
   if(startH == endH)
      return(false);
   if(startH < endH)
      return(hour >= startH && hour < endH);
// wrapping window, e.g. 23 -> 1
   return(hour >= startH || hour < endH);
  }

bool SessionAllowsEntry(const MqlDateTime &dt)
  {
   if(dt.day_of_week == 0 || dt.day_of_week == 6)
      return(false);
   if(dt.day_of_week == 1 && !InpTradeMonday)
      return(false);
   if(dt.day_of_week == 5)
     {
      if(!InpTradeFriday)
         return(false);
      if(dt.hour >= InpFridayStopHour)
         return(false);
     }

   if(InpAvoidRollover && InHourWindow(dt.hour, InpRolloverStart, InpRolloverEnd))
      return(false);

   if(!InpUseSessions)
      return(true);

   if(InHourWindow(dt.hour, InpSess1StartHour, InpSess1EndHour))
      return(true);
   if(InHourWindow(dt.hour, InpSess2StartHour, InpSess2EndHour))
      return(true);
   if(g_useSess3 && InHourWindow(dt.hour, InpSess3StartHour, InpSess3EndHour))
      return(true);
   return(false);
  }

//+------------------------------------------------------------------+
//| Position lookup                                                  |
//+------------------------------------------------------------------+
bool SelectMyPosition(ulong &ticket)
  {
   ticket = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      ticket = t;
      return(true);
     }
   return(false);
  }

bool HasMyPosition()
  {
   ulong t = 0;
   return(SelectMyPosition(t));
  }

//+------------------------------------------------------------------+
//| Sizing                                                           |
//+------------------------------------------------------------------+
double EffectiveRiskPercent()
  {
// Anti-martingale: smaller after losses, modestly larger after wins.
   double mult = 1.0;
   if(InpLossesToReduce > 0 && g_consecLosses >= InpLossesToReduce)
      mult = InpRiskMultAfterLoss;
   else if(InpWinsToBoost > 0 && g_consecWins >= InpWinsToBoost)
      mult = InpRiskMultAfterWin;
   return(InpRiskPercent * mult);
  }

double LotsForRisk(const double slPoints)
  {
   if(slPoints <= 0.0)
      return(0.0);

   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskMoney = equity * EffectiveRiskPercent() / 100.0;
   double pvl       = PointValuePerLot();
   if(pvl <= 0.0 || riskMoney <= 0.0)
      return(0.0);

// Cost of being stopped out on one full lot, commission included.
   double lossPerLot = slPoints * pvl + InpCommissionRT;
   if(lossPerLot <= 0.0)
      return(0.0);

   double lots = riskMoney / lossPerLot;

// Snap to the broker's volume step, rounding DOWN so risk is never exceeded.
   lots = NormVolume(lots);
   lots = MathMin(lots, MathMin(g_volMax, InpMaxLots));

   if(lots < g_volMin)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: required lots %.4f below symbol minimum %.4f. "
                     "Stop distance %.0f pts would risk more than %.2f%% at min lot.",
                     lots, g_volMin, slPoints, EffectiveRiskPercent());
      return(0.0);
     }

   return(lots);
  }

bool MarginOK(const ENUM_ORDER_TYPE type, const double lots, const double price)
  {
   double need = 0.0;
   if(!OrderCalcMargin(type, _Symbol, lots, price, need))
      return(false);
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
// Require a 3x buffer so one trade can never consume the account's headroom.
   return(need * 3.0 <= freeMargin);
  }

//+------------------------------------------------------------------+
//| Risk guards                                                      |
//+------------------------------------------------------------------+
void UpdateRiskAnchors()
  {
   datetime   now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   int ds = DayStamp(dt);
   if(g_curDayStamp != ds)
     {
      g_curDayStamp    = ds;
      g_dayStartEquity = equity;
      g_tradesToday    = 0;
      g_dayHalt        = false;
     }

   int ws = WeekStamp(now, dt);
   if(g_curWeekStamp != ws)
     {
      g_curWeekStamp    = ws;
      g_weekStartEquity = equity;
      g_weekHalt        = false;
     }

   if(equity > g_peakEquity)
      g_peakEquity = equity;
  }

void EvaluateGuards()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   if(!g_dayHalt && g_dayStartEquity > 0.0 && InpDailyLossPct > 0.0)
     {
      double dd = (g_dayStartEquity - equity) / g_dayStartEquity * 100.0;
      if(dd >= InpDailyLossPct)
        {
         g_dayHalt = true;
         PrintFormat("GUARD: daily loss %.2f%% >= %.2f%%. No new entries today.",
                     dd, InpDailyLossPct);
        }
     }

   if(!g_weekHalt && g_weekStartEquity > 0.0 && InpWeeklyLossPct > 0.0)
     {
      double dd = (g_weekStartEquity - equity) / g_weekStartEquity * 100.0;
      if(dd >= InpWeeklyLossPct)
        {
         g_weekHalt = true;
         PrintFormat("GUARD: weekly loss %.2f%% >= %.2f%%. No new entries this week.",
                     dd, InpWeeklyLossPct);
        }
     }

   if(!g_hardHalt && g_peakEquity > 0.0 && InpMaxEquityDDPct > 0.0)
     {
      double dd = (g_peakEquity - equity) / g_peakEquity * 100.0;
      if(dd >= InpMaxEquityDDPct)
        {
         g_hardHalt = true;
         PrintFormat("GUARD: peak-to-trough equity drawdown %.2f%% >= %.2f%%. EA disabled.",
                     dd, InpMaxEquityDDPct);
        }
     }

   if(InpFlattenOnGuard && (g_hardHalt || g_dayHalt || g_weekHalt))
     {
      ulong t = 0;
      if(SelectMyPosition(t))
        {
         if(trade.PositionClose(t, (ulong)InpSlippagePts))
            Print("GUARD: open position flattened.");
        }
     }
  }

bool EntriesEnabled()
  {
   if(g_hardHalt || g_dayHalt || g_weekHalt)
      return(false);
   if(g_cooldownLeft > 0)
      return(false);
   if(InpMaxTradesPerDay > 0 && g_tradesToday >= InpMaxTradesPerDay)
      return(false);

   if(InpMinBarsBetweenTrades > 0 && g_lastExitBar > 0 && g_lastBarTime > 0)
     {
      int since = Bars(_Symbol, InpSignalTF, g_lastExitBar, g_lastBarTime) - 1;
      if(since < InpMinBarsBetweenTrades)
         return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Indicator reads (closed bars only, no intrabar peeking)          |
//+------------------------------------------------------------------+
bool ReadIndicators(SInd &x)
  {
   double buf[];

   if(CopyBuffer(hEma, 0, 1, 1, buf) != 1)
      return(false);
   x.ema1 = buf[0];

   if(CopyBuffer(hAtr, 0, 1, 1, buf) != 1)
      return(false);
   x.atr1 = buf[0];

   if(CopyBuffer(hAtrSlow, 0, 1, 1, buf) != 1)
      return(false);
   x.atrSlow = buf[0];

// two RSI values: buf[0] is the older bar (shift 2), buf[1] the last closed bar (shift 1)
   if(CopyBuffer(hRsi, 0, 1, 2, buf) != 2)
      return(false);
   x.rsi2 = buf[0];
   x.rsi1 = buf[1];

   x.adx1 = 0.0;
   if(InpRegimeMode == REGIME_RANGE_ONLY)
     {
      if(CopyBuffer(hAdx, 0, 1, 1, buf) != 1)
         return(false);
      x.adx1 = buf[0];
     }

   x.upper1 = x.upper2 = x.lower1 = x.lower2 = 0.0;
   if(InpEntryModel != MODEL_STRETCH)
     {
      if(CopyBuffer(hBands, 1, 1, 2, buf) != 2)   // UPPER_BAND
         return(false);
      x.upper2 = buf[0];
      x.upper1 = buf[1];
      if(CopyBuffer(hBands, 2, 1, 2, buf) != 2)   // LOWER_BAND
         return(false);
      x.lower2 = buf[0];
      x.lower1 = buf[1];
     }

   x.biasEma = x.biasClose = 0.0;
   if(InpBiasMode == BIAS_WITH_TREND)
     {
      if(CopyBuffer(hBiasEma, 0, 1, 1, buf) != 1)
         return(false);
      x.biasEma = buf[0];
      double c[];
      if(CopyClose(_Symbol, InpRegimeTF, 1, 1, c) != 1)
         return(false);
      x.biasClose = c[0];
     }

   MqlRates r[];
   if(CopyRates(_Symbol, InpSignalTF, 1, 2, r) != 2)
      return(false);
   x.r2 = r[0];
   x.r1 = r[1];

   return(x.atr1 > 0.0);
  }

//+------------------------------------------------------------------+
//| Signals                                                          |
//+------------------------------------------------------------------+
bool BiasAllows(const int dir, const SInd &x)
  {
   if(InpBiasMode == BIAS_NONE || x.biasEma <= 0.0)
      return(true);
   return(dir > 0 ? (x.biasClose > x.biasEma) : (x.biasClose < x.biasEma));
  }

bool VolatilityAllows(const SInd &x)
  {
   if(x.atrSlow <= 0.0)
      return(true);
   double ratio = x.atr1 / x.atrSlow;
   if(InpMinAtrRatio > 0.0 && ratio < InpMinAtrRatio)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: ATR ratio %.2f below floor %.2f (dead tape).", ratio, InpMinAtrRatio);
      return(false);
     }
   if(InpMaxAtrRatio > 0.0 && ratio > InpMaxAtrRatio)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: ATR ratio %.2f above ceiling %.2f (volatility spike).", ratio, InpMaxAtrRatio);
      return(false);
     }
   return(true);
  }

// Model A: bar closed far from the anchor, momentum washed out,
// and (optionally) the bar itself turned back toward the mean.
int SignalStretch(const SInd &x)
  {
   double stretch = InpStretchATR * x.atr1;

   if(x.r1.close < x.ema1 - stretch && x.rsi1 < InpRsiBuyBelow)
     {
      if(!InpRequireRejection || x.r1.close > x.r1.open)
         return(1);
     }
   if(x.r1.close > x.ema1 + stretch && x.rsi1 > InpRsiSellAbove)
     {
      if(!InpRequireRejection || x.r1.close < x.r1.open)
         return(-1);
     }
   return(0);
  }

// Model B: the bar before last closed OUTSIDE the band, the last bar
// closed back INSIDE it in the direction of the mean, the wash-out
// bar had an extreme fast RSI, and price still sits far enough from
// the EMA for the target to be worth taking.
int SignalBand(const SInd &x)
  {
   if(x.upper1 <= 0.0 || x.lower1 <= 0.0)
      return(0);

   double minStretch = InpBandMinStretchATR * x.atr1;

   if(x.r2.close < x.lower2 &&
      x.r1.close > x.lower1 &&
      x.r1.close > x.r1.open &&
      x.r1.close < x.ema1 - minStretch &&
      x.rsi2 < InpBandRsiBuyBelow)
      return(1);

   if(x.r2.close > x.upper2 &&
      x.r1.close < x.upper1 &&
      x.r1.close < x.r1.open &&
      x.r1.close > x.ema1 + minStretch &&
      x.rsi2 > InpBandRsiSellAbove)
      return(-1);

   return(0);
  }

// Returns  1 = long, -1 = short, 0 = none. `model` names the model that fired.
int Signal(const SInd &x, string &model)
  {
   model = "";

   if(InpRegimeMode == REGIME_RANGE_ONLY && x.adx1 > InpAdxMax)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: ADX %.1f above %.1f (trending regime).", x.adx1, InpAdxMax);
      return(0);
     }
   if(!VolatilityAllows(x))
      return(0);

   int dir = 0;
   if(InpEntryModel == MODEL_STRETCH || InpEntryModel == MODEL_BOTH)
     {
      dir = SignalStretch(x);
      if(dir != 0)
         model = "A-stretch";
     }
   if(dir == 0 && (InpEntryModel == MODEL_BAND || InpEntryModel == MODEL_BOTH))
     {
      dir = SignalBand(x);
      if(dir != 0)
         model = "B-band";
     }

   if(dir != 0 && !BiasAllows(dir, x))
     {
      if(InpVerboseLog)
         PrintFormat("Reject: %s signal against %s bias.", model, EnumToString(InpRegimeTF));
      return(0);
     }
   return(dir);
  }

//+------------------------------------------------------------------+
//| Entry                                                            |
//+------------------------------------------------------------------+
bool TryEnter(const int dir, const SInd &x, const string model)
  {
   int spread = SpreadPoints();
   if(spread > g_maxSpreadPts)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: spread %d pts above cap %d.", spread, g_maxSpreadPts);
      return(false);
     }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return(false);

   double entry    = (dir > 0 ? ask : bid);
   double slPoints = InpSLatr * x.atr1 / g_point;

   double tpPoints;
   if(InpTPMode == TP_MEAN_REVERT)
      tpPoints = (dir > 0 ? (x.ema1 - entry) : (entry - x.ema1)) / g_point;
   else
      tpPoints = InpTPatr * x.atr1 / g_point;

// Respect broker minimum stop distance on both legs.
   int minStop = MinStopPoints();
   if(slPoints < minStop)
      slPoints = minStop;
   if(tpPoints < minStop)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: target %.0f pts inside broker minimum %d.", tpPoints, minStop);
      return(false);
     }

// Cost gate: the final target has to be worth taking after spread and commission.
   double costPoints = (double)spread + CommissionInPoints();
   if(tpPoints < InpMinTPtoCostRatio * costPoints)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: target %.0f pts under cost gate (%.1f x %.1f = %.0f pts).",
                     tpPoints, InpMinTPtoCostRatio, costPoints,
                     InpMinTPtoCostRatio * costPoints);
      return(false);
     }

// Scale-out target: only used when it is clear of the minimum stop,
// clear of cost, and sits inside the final target.
   double partialPts = 0.0;
   if(InpPartialPct > 0.0 && InpPartialATR > 0.0)
     {
      double cand = InpPartialATR * x.atr1 / g_point;
      if(cand >= minStop && cand >= 1.5 * costPoints && cand < tpPoints)
         partialPts = cand;
     }

   double lots = LotsForRisk(slPoints);
   if(lots <= 0.0)
      return(false);

   ENUM_ORDER_TYPE type = (dir > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   if(!MarginOK(type, lots, entry))
     {
      if(InpVerboseLog)
         Print("Reject: insufficient free margin for the required size.");
      return(false);
     }

   double sl = (dir > 0 ? entry - slPoints * g_point : entry + slPoints * g_point);
   double tp = (dir > 0 ? entry + tpPoints * g_point : entry - tpPoints * g_point);
   sl = NormPrice(sl);
   tp = NormPrice(tp);

   bool ok = (dir > 0)
             ? trade.Buy(lots, _Symbol, 0.0, sl, tp, InpComment)
             : trade.Sell(lots, _Symbol, 0.0, sl, tp, InpComment);

   if(!ok)
     {
      PrintFormat("Entry failed. retcode=%d (%s)",
                  trade.ResultRetcode(), trade.ResultRetcodeDescription());
      return(false);
     }

   g_tradesToday++;

   ulong t = 0;
   if(!SelectMyPosition(t))
      return(true);   // filled asynchronously; ManagePosition will recover it

   PositionSelectByTicket(t);
   g_posTicket      = t;
   g_posId          = PositionGetInteger(POSITION_IDENTIFIER);
   g_posDir         = dir;
   g_posOpenBar     = g_lastBarTime;
   g_posRiskPoints  = slPoints;
   g_posInitLots    = PositionGetDouble(POSITION_VOLUME);
   g_posPartialPts  = partialPts;
   g_posBEDone      = false;
   g_posPartialDone = (partialPts <= 0.0);
   SaveState();

   PrintFormat("Entry %s %s %.2f lots @ %s | SL %.0f | TP %.0f | scale-out %.0f | spread %d | risk %.2f%%",
               model, (dir > 0 ? "BUY" : "SELL"), lots, DoubleToString(entry, g_digits),
               slPoints, tpPoints, partialPts, spread, EffectiveRiskPercent());
   return(true);
  }

//+------------------------------------------------------------------+
//| Recover bookkeeping for a position this EA did not see opened    |
//| in this process (restart, async fill).                           |
//+------------------------------------------------------------------+
void RecoverPositionState(const ulong t, const double atr1)
  {
   if(!PositionSelectByTicket(t))
      return;

   ResetPositionGlobals();
   g_posTicket = t;
   g_posId     = PositionGetInteger(POSITION_IDENTIFIER);
   g_posDir    = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1);

   if(LoadState(t))
     {
      Print("Position state restored from terminal globals.");
      return;
     }

   double open = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl   = PositionGetDouble(POSITION_SL);
   double vol  = PositionGetDouble(POSITION_VOLUME);

   g_posRiskPoints  = (sl > 0.0 ? MathAbs(open - sl) / g_point : 0.0);
   g_posInitLots    = vol;
   g_posOpenBar     = BarTimeAt((datetime)PositionGetInteger(POSITION_TIME));
   g_posBEDone      = (g_posDir > 0 ? (sl >= open) : (sl > 0.0 && sl <= open));
   g_posPartialDone = false;
   g_posPartialPts  = (InpPartialPct > 0.0 && atr1 > 0.0 ? InpPartialATR * atr1 / g_point : 0.0);
   if(g_posPartialPts <= 0.0)
      g_posPartialDone = true;

   SaveState();
   Print("Position state rebuilt from the live position.");
  }

//+------------------------------------------------------------------+
//| A position we were tracking is gone: settle it into the streak   |
//| counters. Net P/L comes from history so partial closes and the   |
//| final close are counted as ONE trade.                            |
//+------------------------------------------------------------------+
void FinalizeClosedTrade()
  {
   double   net      = 0.0;
   datetime lastDeal = 0;
   bool     found    = false;

   if(g_posId > 0 && HistorySelectByPosition(g_posId))
     {
      int n = HistoryDealsTotal();
      for(int i = 0; i < n; i++)
        {
         ulong d = HistoryDealGetTicket(i);
         if(d == 0)
            continue;
         found = true;
         net += HistoryDealGetDouble(d, DEAL_PROFIT)
                + HistoryDealGetDouble(d, DEAL_COMMISSION)
                + HistoryDealGetDouble(d, DEAL_SWAP);
         datetime dt = (datetime)HistoryDealGetInteger(d, DEAL_TIME);
         if(dt > lastDeal)
            lastDeal = dt;
        }
     }

   if(found)
     {
      if(net < 0.0)
        {
         g_consecLosses++;
         g_consecWins = 0;
         if(InpMaxConsecLosses > 0 && g_consecLosses >= InpMaxConsecLosses)
           {
            g_cooldownLeft = InpCooldownBars;
            PrintFormat("%d consecutive losses. Cooldown for %d bars.",
                        g_consecLosses, InpCooldownBars);
           }
        }
      else
        {
         g_consecWins++;
         g_consecLosses = 0;
        }
      PrintFormat("Trade closed: net %.2f | streak W%d / L%d | next risk %.2f%%",
                  net, g_consecWins, g_consecLosses, EffectiveRiskPercent());
     }

   g_lastExitBar = (lastDeal > 0 ? BarTimeAt(lastDeal) : g_lastBarTime);

   ClearState();
   ResetPositionGlobals();
  }

//+------------------------------------------------------------------+
//| Open position management                                         |
//+------------------------------------------------------------------+
void ManagePosition(const SInd &x, const bool newBar)
  {
   ulong t = 0;
   if(!SelectMyPosition(t))
      return;

   if(t != g_posTicket)
      RecoverPositionState(t, x.atr1);

   if(!PositionSelectByTicket(t))
      return;

   long   ptype  = PositionGetInteger(POSITION_TYPE);
   double open   = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl     = PositionGetDouble(POSITION_SL);
   double tp     = PositionGetDouble(POSITION_TP);
   double cur    = PositionGetDouble(POSITION_PRICE_CURRENT);
   double vol    = PositionGetDouble(POSITION_VOLUME);
   bool   isLong = (ptype == POSITION_TYPE_BUY);

   if(g_posRiskPoints <= 0.0)
     {
      if(sl > 0.0)
         g_posRiskPoints = MathAbs(open - sl) / g_point;
      if(g_posRiskPoints <= 0.0)
         return;
     }

   double movePoints = (isLong ? (cur - open) : (open - cur)) / g_point;
   double rMultiple  = movePoints / g_posRiskPoints;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

// --- weekend flatten (hard) ---
   if(InpFridayFlatten && dt.day_of_week == 5 && dt.hour >= InpFridayCloseHour)
     {
      if(trade.PositionClose(t, (ulong)InpSlippagePts))
         Print("Weekend flatten.");
      return;
     }

// --- time stop ---
// Closes trades that have gone nowhere. A trade that is working
// (at or above InpTimeStopKeepR) is left to its stop, target or
// trail, but is hard-closed at twice the bar budget.
   if(InpMaxBarsInTrade > 0 && g_posOpenBar > 0)
     {
      int bars = Bars(_Symbol, InpSignalTF, g_posOpenBar, g_lastBarTime);
      bool stale = (bars >= InpMaxBarsInTrade && rMultiple < InpTimeStopKeepR);
      bool hard  = (bars >= 2 * InpMaxBarsInTrade);
      if(stale || hard)
        {
         if(trade.PositionClose(t, (ulong)InpSlippagePts))
            PrintFormat("Time stop after %d bars at %.2fR%s.", bars, rMultiple,
                        (hard ? " (hard limit)" : ""));
         return;
        }
     }

// --- mean reached between TP updates (mean-revert mode) ---
   if(InpTPMode == TP_MEAN_REVERT && x.ema1 > 0.0)
     {
      bool reached = (isLong ? (cur >= x.ema1) : (cur <= x.ema1));
      if(reached)
        {
         if(trade.PositionClose(t, (ulong)InpSlippagePts))
            PrintFormat("Mean reached at %.2fR.", rMultiple);
         return;
        }
     }

// --- scale-out ---
   if(!g_posPartialDone && g_posPartialPts > 0.0 && movePoints >= g_posPartialPts)
     {
      double closeVol = NormVolume(g_posInitLots * InpPartialPct / 100.0);
      double remain   = NormVolume(vol - closeVol);
      if(closeVol >= g_volMin && remain >= g_volMin)
        {
         if(trade.PositionClosePartial(t, closeVol, (ulong)InpSlippagePts))
           {
            g_posPartialDone = true;
            SaveState();
            PrintFormat("Scale-out: %.2f lots banked at %.2fR, %.2f lots run to the mean.",
                        closeVol, rMultiple, remain);
            if(!PositionSelectByTicket(t))
               return;
            sl  = PositionGetDouble(POSITION_SL);
            tp  = PositionGetDouble(POSITION_TP);
            cur = PositionGetDouble(POSITION_PRICE_CURRENT);
           }
        }
      else
        {
         // Too small to split: treat the whole position as the runner.
         g_posPartialDone = true;
         SaveState();
        }
     }

   double newSL = sl;

// --- break even ---
// Armed by R, or immediately once the scale-out has banked. Not marked
// done until the broker has actually accepted the new stop.
   bool   wantBE = false;
   double beCand = 0.0;
   if(!g_posBEDone && ((InpBreakEvenAtR > 0.0 && rMultiple >= InpBreakEvenAtR) || g_posPartialDone))
     {
      wantBE = true;
      double lock = InpBreakEvenOffsetR * g_posRiskPoints * g_point;
      beCand = (isLong ? open + lock : open - lock);
      if(isLong  && (sl <= 0.0 || beCand > sl))
         newSL = beCand;
      if(!isLong && (sl <= 0.0 || beCand < sl))
         newSL = beCand;
     }

// --- ATR trail ---
   if(InpTrailStartR > 0.0 && InpTrailATR > 0.0 && rMultiple >= InpTrailStartR)
     {
      double dist = InpTrailATR * x.atr1;
      double cand = (isLong ? cur - dist : cur + dist);
      if(isLong  && cand > newSL)
         newSL = cand;
      if(!isLong && (newSL <= 0.0 || cand < newSL))
         newSL = cand;
     }

// --- validate the stop against the broker's minimum distance ---
   double minDist = MinStopPoints() * g_point;
   double slFinal = sl;
   if(newSL != sl && newSL > 0.0)
     {
      newSL = NormPrice(newSL);
      bool valid = (isLong ? (cur - newSL >= minDist) : (newSL - cur >= minDist));
      if(valid && newSL != sl)
         slFinal = newSL;
     }

// --- re-anchor the target to the live mean once per bar ---
   double tpFinal = tp;
   if(newBar && InpTPMode == TP_MEAN_REVERT && x.ema1 > 0.0)
     {
      double cand  = NormPrice(x.ema1);
      bool   valid = (isLong ? (cand - cur >= minDist) : (cur - cand >= minDist));
      if(valid && MathAbs(cand - tp) >= 2.0 * g_point)
         tpFinal = cand;
     }

// --- apply ---
   if(slFinal != sl || tpFinal != tp)
     {
      if(trade.PositionModify(t, slFinal, tpFinal))
        {
         if(wantBE && slFinal != sl)
           {
            g_posBEDone = (isLong ? (slFinal >= beCand - g_point * 0.5)
                                  : (slFinal <= beCand + g_point * 0.5));
            if(g_posBEDone)
               SaveState();
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Lifecycle                                                        |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_point      = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   g_digits     = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   g_tickValue  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   g_tickSize   = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   g_volMin     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   g_volMax     = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   g_volStep    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   g_stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);

   if(g_point <= 0.0 || g_tickSize <= 0.0 || g_volStep <= 0.0)
     {
      Print("Init failed: incomplete symbol specification.");
      return(INIT_FAILED);
     }

// volume precision from the step (0.01 -> 2, 0.001 -> 3)
   g_volDigits = 0;
   for(double s = g_volStep; s < 1.0 - 1e-12 && g_volDigits < 8; s *= 10.0)
      g_volDigits++;

   if(InpRiskPercent <= 0.0 || InpRiskPercent > 5.0)
     {
      Print("Init failed: InpRiskPercent must be between 0 and 5.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpRiskPercent * MathMax(1.0, InpRiskMultAfterWin) > 5.0)
     {
      Print("Init failed: risk after a win streak exceeds 5%. Lower InpRiskMultAfterWin.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpRiskMultAfterLoss <= 0.0 || InpRiskMultAfterLoss > 1.0)
     {
      Print("Init failed: InpRiskMultAfterLoss must be in (0, 1]. Losses never raise size.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpSLatr <= 0.0)
     {
      Print("Init failed: InpSLatr must be positive. This EA will not run without a hard stop.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpPartialPct < 0.0 || InpPartialPct >= 100.0)
     {
      Print("Init failed: InpPartialPct must be in [0, 100).");
      return(INIT_PARAMETERS_INCORRECT);
     }

   ApplyPreset();

   hEma     = iMA(_Symbol, InpSignalTF, InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   hAtr     = iATR(_Symbol, InpSignalTF, InpAtrPeriod);
   hAtrSlow = iATR(_Symbol, InpSignalTF, InpAtrSlowPeriod);
   hRsi     = iRSI(_Symbol, InpSignalTF, InpRsiPeriod, PRICE_CLOSE);
   hAdx     = iADX(_Symbol, InpRegimeTF, InpAdxPeriod);
   hBands   = iBands(_Symbol, InpSignalTF, InpBandPeriod, 0, InpBandDeviation, PRICE_CLOSE);
   hBiasEma = iMA(_Symbol, InpRegimeTF, InpBiasEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);

   if(hEma == INVALID_HANDLE || hAtr == INVALID_HANDLE || hAtrSlow == INVALID_HANDLE ||
      hRsi == INVALID_HANDLE || hAdx == INVALID_HANDLE || hBands == INVALID_HANDLE ||
      hBiasEma == INVALID_HANDLE)
     {
      Print("Init failed: could not create indicator handles.");
      return(INIT_FAILED);
     }

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints((ulong)InpSlippagePts);
   trade.SetTypeFillingBySymbol(_Symbol);
   trade.SetAsyncMode(false);
   trade.LogLevel(LOG_LEVEL_ERRORS);

   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   g_dayStartEquity  = eq;
   g_weekStartEquity = eq;
   g_peakEquity      = eq;

// An open position from a previous run is picked up on the first tick.
   ResetPositionGlobals();

   PrintFormat("SnapScalp_MR_v2 init OK. %s %s | preset %s | spread cap %d pts | session3 %s | "
               "point %.*f | min lot %.2f | commission %.1f/lot RT = %.1f pts | stops level %d",
               _Symbol, EnumToString(InpSignalTF), g_presetName, g_maxSpreadPts,
               (g_useSess3 ? "on" : "off"), g_digits, g_point, g_volMin,
               InpCommissionRT, CommissionInPoints(), (int)g_stopsLevel);

   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   if(hEma     != INVALID_HANDLE) IndicatorRelease(hEma);
   if(hAtr     != INVALID_HANDLE) IndicatorRelease(hAtr);
   if(hAtrSlow != INVALID_HANDLE) IndicatorRelease(hAtrSlow);
   if(hRsi     != INVALID_HANDLE) IndicatorRelease(hRsi);
   if(hAdx     != INVALID_HANDLE) IndicatorRelease(hAdx);
   if(hBands   != INVALID_HANDLE) IndicatorRelease(hBands);
   if(hBiasEma != INVALID_HANDLE) IndicatorRelease(hBiasEma);
// Terminal globals are left in place on purpose so a restart can recover.
  }

void OnTick()
  {
   UpdateRiskAnchors();
   EvaluateGuards();

   bool newBar = IsNewBar();

   SInd x;
   bool haveInd = ReadIndicators(x);

// Manage any live position on every tick so stops react promptly.
   if(haveInd)
      ManagePosition(x, newBar);

// A tracked position that is no longer open has to be settled before
// anything else can happen.
   if(g_posTicket != 0 && !HasMyPosition())
      FinalizeClosedTrade();

// Entries only on a fresh closed bar.
   if(!newBar)
      return;

   if(g_cooldownLeft > 0)
     {
      g_cooldownLeft--;
      if(g_cooldownLeft == 0)
         Print("Cooldown complete. Entries re-enabled.");
      return;
     }

   if(!haveInd)
      return;
   if(HasMyPosition())
      return;
   if(!EntriesEnabled())
      return;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(!SessionAllowsEntry(dt))
      return;

   string model = "";
   int dir = Signal(x, model);
   if(dir != 0)
      TryEnter(dir, x, model);
  }

//+------------------------------------------------------------------+
//| Optimisation objective                                           |
//|                                                                   |
//| Net profit alone rewards fragile parameter sets. This scores      |
//| return per unit of equity drawdown, scaled by trade count so a    |
//| lucky 12-trade run cannot outrank a stable 300-trade one, and     |
//| returns 0 below a minimum sample size or a profit factor of 1.    |
//+------------------------------------------------------------------+
double OnTester()
  {
   double trades = TesterStatistics(STAT_TRADES);
   if(trades < InpMinTradesForTester)
      return(0.0);

   double net = TesterStatistics(STAT_PROFIT);
   if(net <= 0.0)
      return(0.0);

   double ddPct = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   if(ddPct < 0.1)
      ddPct = 0.1;

   double pf = TesterStatistics(STAT_PROFIT_FACTOR);
   if(pf <= 1.0)
      return(0.0);

   return((net / ddPct) * MathSqrt(trades / 100.0));
  }
//+------------------------------------------------------------------+
