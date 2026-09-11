//+------------------------------------------------------------------+
//|                                                 SnapScalp_MR.mq5 |
//|   ATR-stretch mean-reversion intraday scalper for FX majors       |
//|   GBPUSD / EURUSD / USDJPY  -  M5 or M15                          |
//|                                                                   |
//|   Design constraints (deliberate, do not "improve" them away):    |
//|     * Every position carries a hard stop loss at broker level.    |
//|     * One position at a time. No grid. No averaging. No recovery  |
//|       lot multiplication of any kind.                             |
//|     * Position size derived from stop distance, never from        |
//|       previous trade outcomes.                                    |
//|     * Trade is rejected unless the target clears modelled         |
//|       transaction cost by a stated multiple.                      |
//|     * Three independent equity guards (day / week / peak-to-      |
//|       trough) that can flatten and disable the EA.                |
//+------------------------------------------------------------------+
#property copyright "Private use"
#property version   "1.00"
#property description "Session-filtered ATR stretch mean reversion for FX majors."
#property description "Fixed-fractional risk, hard SL on every trade, layered equity guards."

#include <Trade\Trade.mqh>

//--- regime handling -----------------------------------------------
enum ENUM_REGIME_MODE
  {
   REGIME_RANGE_ONLY = 0,   // Only trade when ADX below threshold
   REGIME_ANY        = 1    // Ignore the regime filter
  };

//--- exit target style ---------------------------------------------
enum ENUM_TP_MODE
  {
   TP_ATR_MULTIPLE = 0,     // Fixed ATR multiple
   TP_MEAN_REVERT  = 1      // Target the anchor EMA itself
  };

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input group "=== Identity ==="
input long              InpMagic              = 770101;   // Magic number (unique per chart)
input string            InpComment            = "SnapScalp"; // Order comment

input group "=== Signal ==="
input ENUM_TIMEFRAMES   InpSignalTF           = PERIOD_M5;  // Signal timeframe
input ENUM_TIMEFRAMES   InpRegimeTF           = PERIOD_H1;  // Regime timeframe
input int               InpEmaPeriod          = 20;         // Anchor EMA period (signal TF)
input int               InpAtrPeriod          = 14;         // ATR period (signal TF)
input int               InpRsiPeriod          = 2;          // RSI period (fast, signal TF)
input double            InpStretchATR         = 1.30;       // Stretch from EMA required, in ATR
input double            InpRsiBuyBelow        = 8.0;        // RSI must be below this to buy
input double            InpRsiSellAbove       = 92.0;       // RSI must be above this to sell
input bool              InpRequireRejection   = true;       // Require signal bar to close back toward mean
input ENUM_REGIME_MODE  InpRegimeMode         = REGIME_RANGE_ONLY; // Regime filter mode
input int               InpAdxPeriod          = 14;         // ADX period (regime TF)
input double            InpAdxMax             = 28.0;       // Max ADX to allow mean reversion

input group "=== Exits ==="
input double            InpSLatr              = 1.60;       // Stop loss, in ATR
input ENUM_TP_MODE      InpTPMode             = TP_ATR_MULTIPLE; // Take profit style
input double            InpTPatr              = 1.10;       // Take profit, in ATR (if ATR mode)
input double            InpBreakEvenAtR       = 0.60;       // Move to BE at this R (0 = off)
input double            InpBreakEvenOffsetR   = 0.10;       // BE lock-in, in R
input double            InpTrailStartR        = 1.00;       // Start ATR trail at this R (0 = off)
input double            InpTrailATR           = 1.20;       // Trailing distance, in ATR
input int               InpMaxBarsInTrade     = 18;         // Time stop, signal-TF bars (0 = off)

input group "=== Costs & execution ==="
input double            InpCommissionRT       = 7.00;       // Commission per 1.00 lot, round turn, acct ccy
input int               InpMaxSpreadPts       = 25;         // Reject entry above this spread (points)
input double            InpMinTPtoCostRatio   = 3.00;       // Target must exceed cost by this multiple
input int               InpSlippagePts        = 10;         // Max deviation (points)

input group "=== Risk ==="
input double            InpRiskPercent        = 0.50;       // Risk per trade, % of equity
input double            InpMaxLots            = 0.30;       // Hard lot ceiling
input int               InpMaxTradesPerDay    = 6;          // Max entries per day (0 = unlimited)
input double            InpDailyLossPct       = 2.00;       // Daily loss halt, % of day-start equity
input double            InpWeeklyLossPct      = 4.00;       // Weekly loss halt, % of week-start equity
input double            InpMaxEquityDDPct     = 10.00;      // Peak-to-trough equity halt, %
input bool              InpFlattenOnGuard     = true;       // Close open trade when a guard trips
input int               InpMaxConsecLosses    = 3;          // Losses in a row before cooldown
input int               InpCooldownBars       = 24;         // Cooldown length, signal-TF bars

input group "=== Sessions (broker server time) ==="
input bool              InpUseSessions        = true;       // Restrict to sessions
input int               InpSess1StartHour     = 7;          // Session 1 start hour
input int               InpSess1EndHour       = 11;         // Session 1 end hour
input int               InpSess2StartHour     = 13;         // Session 2 start hour
input int               InpSess2EndHour       = 17;         // Session 2 end hour
input bool              InpAvoidRollover      = true;       // Skip the rollover window
input int               InpRolloverStart      = 23;         // Rollover start hour
input int               InpRolloverEnd        = 1;          // Rollover end hour
input bool              InpTradeMonday        = true;       // Trade Monday
input bool              InpTradeFriday        = true;       // Trade Friday
input int               InpFridayStopHour     = 19;         // No new trades Friday after this hour
input bool              InpFridayFlattenHour  = true;       // Flatten before the weekend
input int               InpFridayCloseHour    = 20;         // Friday flatten hour

input group "=== Diagnostics ==="
input bool              InpVerboseLog         = false;      // Log rejected signals
input int               InpMinTradesForTester = 40;         // OnTester: minimum trades to score

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CTrade   trade;

int      hEma   = INVALID_HANDLE;
int      hAtr   = INVALID_HANDLE;
int      hRsi   = INVALID_HANDLE;
int      hAdx   = INVALID_HANDLE;

double   g_point       = 0.0;
int      g_digits      = 0;
double   g_tickValue   = 0.0;
double   g_tickSize    = 0.0;
double   g_volMin      = 0.0;
double   g_volMax      = 0.0;
double   g_volStep     = 0.0;
long     g_stopsLevel  = 0;

datetime g_lastBarTime = 0;

// risk state
double   g_dayStartEquity  = 0.0;
double   g_weekStartEquity = 0.0;
double   g_peakEquity      = 0.0;
int      g_curDay          = -1;
int      g_curWeekStamp    = -1;
int      g_tradesToday     = 0;
int      g_consecLosses    = 0;
int      g_cooldownLeft    = 0;
bool     g_hardHalt        = false;   // peak-to-trough guard: permanent for the session
bool     g_dayHalt         = false;
bool     g_weekHalt        = false;

// open position bookkeeping
ulong    g_posTicket       = 0;
datetime g_posOpenBar      = 0;
double   g_posRiskPoints   = 0.0;     // initial SL distance in points
bool     g_posBEDone       = false;

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

int WeekStamp(const MqlDateTime &dt)
  {
// Cheap ISO-ish week identifier: year * 100 + week-of-year approximation.
   return(dt.year * 100 + (dt.day_of_year / 7));
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
double LotsForRisk(const double slPoints)
  {
   if(slPoints <= 0.0)
      return(0.0);

   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskMoney = equity * InpRiskPercent / 100.0;
   double pvl       = PointValuePerLot();
   if(pvl <= 0.0 || riskMoney <= 0.0)
      return(0.0);

// Cost of being stopped out on one full lot, commission included.
   double lossPerLot = slPoints * pvl + InpCommissionRT;
   if(lossPerLot <= 0.0)
      return(0.0);

   double lots = riskMoney / lossPerLot;

// Snap to the broker's volume step, rounding DOWN so risk is never exceeded.
   lots = MathFloor(lots / g_volStep) * g_volStep;
   lots = MathMin(lots, MathMin(g_volMax, InpMaxLots));

   if(lots < g_volMin)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: required lots %.4f below symbol minimum %.4f. "
                     "Stop distance %.0f pts would risk more than %.2f%% at min lot.",
                     lots, g_volMin, slPoints, InpRiskPercent);
      return(0.0);
     }

   return(NormalizeDouble(lots, 2));
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
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   if(g_curDay != dt.day)
     {
      g_curDay          = dt.day;
      g_dayStartEquity  = equity;
      g_tradesToday     = 0;
      g_dayHalt         = false;
     }

   int ws = WeekStamp(dt);
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
   return(true);
  }

//+------------------------------------------------------------------+
//| Indicator reads                                                  |
//+------------------------------------------------------------------+
bool ReadIndicators(double &ema1, double &atr1, double &rsi1, double &adx1)
  {
   double buf[];

   if(CopyBuffer(hEma, 0, 1, 1, buf) != 1)
      return(false);
   ema1 = buf[0];

   if(CopyBuffer(hAtr, 0, 1, 1, buf) != 1)
      return(false);
   atr1 = buf[0];

   if(CopyBuffer(hRsi, 0, 1, 1, buf) != 1)
      return(false);
   rsi1 = buf[0];

   adx1 = 0.0;
   if(InpRegimeMode == REGIME_RANGE_ONLY)
     {
      if(CopyBuffer(hAdx, 0, 1, 1, buf) != 1)
         return(false);
      adx1 = buf[0];
     }

   return(atr1 > 0.0);
  }

//+------------------------------------------------------------------+
//| Signal                                                           |
//+------------------------------------------------------------------+
// Returns  1 = long, -1 = short, 0 = none.
// Signal is read from the CLOSED bar (shift 1) only. No intrabar peeking.
int Signal(const double ema1, const double atr1, const double rsi1, const double adx1)
  {
   if(InpRegimeMode == REGIME_RANGE_ONLY && adx1 > InpAdxMax)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: ADX %.1f above %.1f (trending regime).", adx1, InpAdxMax);
      return(0);
     }

   MqlRates r[];
   if(CopyRates(_Symbol, InpSignalTF, 1, 1, r) != 1)
      return(0);

   double stretch = InpStretchATR * atr1;

// Long: bar closed well below the anchor, momentum washed out,
// and (optionally) the bar itself turned back up.
   if(r[0].close < ema1 - stretch && rsi1 < InpRsiBuyBelow)
     {
      if(!InpRequireRejection || r[0].close > r[0].open)
         return(1);
     }

// Short: mirror image.
   if(r[0].close > ema1 + stretch && rsi1 > InpRsiSellAbove)
     {
      if(!InpRequireRejection || r[0].close < r[0].open)
         return(-1);
     }

   return(0);
  }

//+------------------------------------------------------------------+
//| Entry                                                            |
//+------------------------------------------------------------------+
bool TryEnter(const int dir, const double ema1, const double atr1)
  {
   int spread = SpreadPoints();
   if(spread > InpMaxSpreadPts)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: spread %d pts above cap %d.", spread, InpMaxSpreadPts);
      return(false);
     }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return(false);

   double entry    = (dir > 0 ? ask : bid);
   double slPoints = InpSLatr * atr1 / g_point;

   double tpPoints;
   if(InpTPMode == TP_MEAN_REVERT)
      tpPoints = MathAbs(ema1 - entry) / g_point;
   else
      tpPoints = InpTPatr * atr1 / g_point;

// Respect broker minimum stop distance on both legs.
   int minStop = MinStopPoints();
   if(slPoints < minStop)
      slPoints = minStop;
   if(tpPoints < minStop)
      tpPoints = minStop;

// Cost gate: the target has to be worth taking after spread and commission.
   double costPoints = (double)spread + CommissionInPoints();
   if(tpPoints < InpMinTPtoCostRatio * costPoints)
     {
      if(InpVerboseLog)
         PrintFormat("Reject: target %.0f pts under cost gate (%.1f x %.1f = %.0f pts).",
                     tpPoints, InpMinTPtoCostRatio, costPoints,
                     InpMinTPtoCostRatio * costPoints);
      return(false);
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

   ulong t = 0;
   SelectMyPosition(t);
   g_posTicket     = t;
   g_posOpenBar    = g_lastBarTime;
   g_posRiskPoints = slPoints;
   g_posBEDone     = false;
   g_tradesToday++;

   PrintFormat("Entry %s %.2f lots @ %s | SL %.0f pts | TP %.0f pts | spread %d | risk %.2f%%",
               (dir > 0 ? "BUY" : "SELL"), lots, DoubleToString(entry, g_digits),
               slPoints, tpPoints, spread, InpRiskPercent);
   return(true);
  }

//+------------------------------------------------------------------+
//| Open position management                                         |
//+------------------------------------------------------------------+
void ManagePosition(const double atr1)
  {
   ulong t = 0;
   if(!SelectMyPosition(t))
      return;

   if(!PositionSelectByTicket(t))
      return;

   long   ptype   = PositionGetInteger(POSITION_TYPE);
   double open    = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl      = PositionGetDouble(POSITION_SL);
   double tp      = PositionGetDouble(POSITION_TP);
   double cur     = PositionGetDouble(POSITION_PRICE_CURRENT);
   bool   isLong  = (ptype == POSITION_TYPE_BUY);

   if(g_posRiskPoints <= 0.0)
     {
      // Recovered position from a restart: rebuild R from the live stop.
      if(sl > 0.0)
         g_posRiskPoints = MathAbs(open - sl) / g_point;
      if(g_posRiskPoints <= 0.0)
         return;
     }

   double movePoints = (isLong ? (cur - open) : (open - cur)) / g_point;
   double rMultiple  = movePoints / g_posRiskPoints;

   double newSL = sl;

// --- break even ---
   if(!g_posBEDone && InpBreakEvenAtR > 0.0 && rMultiple >= InpBreakEvenAtR)
     {
      double lock = InpBreakEvenOffsetR * g_posRiskPoints * g_point;
      double cand = (isLong ? open + lock : open - lock);
      if(isLong  && (sl <= 0.0 || cand > sl))
         newSL = cand;
      if(!isLong && (sl <= 0.0 || cand < sl))
         newSL = cand;
      g_posBEDone = true;
     }

// --- ATR trail ---
   if(InpTrailStartR > 0.0 && InpTrailATR > 0.0 && rMultiple >= InpTrailStartR)
     {
      double dist = InpTrailATR * atr1;
      double cand = (isLong ? cur - dist : cur + dist);
      if(isLong  && cand > newSL)
         newSL = cand;
      if(!isLong && (newSL <= 0.0 || cand < newSL))
         newSL = cand;
     }

// --- apply, respecting the broker's minimum distance ---
   if(newSL != sl && newSL > 0.0)
     {
      double minDist = MinStopPoints() * g_point;
      bool   valid   = isLong ? (cur - newSL >= minDist) : (newSL - cur >= minDist);
      if(valid)
        {
         newSL = NormPrice(newSL);
         if(newSL != sl)
            trade.PositionModify(t, newSL, tp);
        }
     }

// --- time stop ---
   if(InpMaxBarsInTrade > 0 && g_posOpenBar > 0)
     {
      int bars = Bars(_Symbol, InpSignalTF, g_posOpenBar, g_lastBarTime);
      if(bars >= InpMaxBarsInTrade)
        {
         if(trade.PositionClose(t, (ulong)InpSlippagePts))
            PrintFormat("Time stop after %d bars at %.2fR.", bars, rMultiple);
         return;
        }
     }

// --- Friday flatten ---
   if(InpFridayFlattenHour)
     {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      if(dt.day_of_week == 5 && dt.hour >= InpFridayCloseHour)
        {
         if(trade.PositionClose(t, (ulong)InpSlippagePts))
            Print("Weekend flatten.");
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

   if(InpRiskPercent <= 0.0 || InpRiskPercent > 5.0)
     {
      Print("Init failed: InpRiskPercent must be between 0 and 5.");
      return(INIT_PARAMETERS_INCORRECT);
     }
   if(InpSLatr <= 0.0)
     {
      Print("Init failed: InpSLatr must be positive. This EA will not run without a hard stop.");
      return(INIT_PARAMETERS_INCORRECT);
     }

   hEma = iMA(_Symbol, InpSignalTF, InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   hAtr = iATR(_Symbol, InpSignalTF, InpAtrPeriod);
   hRsi = iRSI(_Symbol, InpSignalTF, InpRsiPeriod, PRICE_CLOSE);
   hAdx = iADX(_Symbol, InpRegimeTF, InpAdxPeriod);

   if(hEma == INVALID_HANDLE || hAtr == INVALID_HANDLE ||
      hRsi == INVALID_HANDLE || hAdx == INVALID_HANDLE)
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

   PrintFormat("SnapScalp_MR init OK. %s %s | point %.*f | min lot %.2f | "
               "commission %.1f/lot RT = %.1f pts | stops level %d",
               _Symbol, EnumToString(InpSignalTF), g_digits, g_point, g_volMin,
               InpCommissionRT, CommissionInPoints(), (int)g_stopsLevel);

   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   if(hEma != INVALID_HANDLE) IndicatorRelease(hEma);
   if(hAtr != INVALID_HANDLE) IndicatorRelease(hAtr);
   if(hRsi != INVALID_HANDLE) IndicatorRelease(hRsi);
   if(hAdx != INVALID_HANDLE) IndicatorRelease(hAdx);
  }

void OnTick()
  {
   UpdateRiskAnchors();
   EvaluateGuards();

   double ema1, atr1, rsi1, adx1;
   bool   haveInd = ReadIndicators(ema1, atr1, rsi1, adx1);

// Manage any live position on every tick so stops react promptly.
   if(haveInd)
      ManagePosition(atr1);

// Entries only on a fresh closed bar.
   if(!IsNewBar())
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

   int dir = Signal(ema1, atr1, rsi1, adx1);
   if(dir != 0)
      TryEnter(dir, ema1, atr1);
  }

//+------------------------------------------------------------------+
//| Track closed trades for the consecutive-loss cooldown            |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest     &request,
                        const MqlTradeResult      &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;
   if(!HistoryDealSelect(trans.deal))
      return;
   if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol)
      return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagic)
      return;

   long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT)
      return;

   double net = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION)
                + HistoryDealGetDouble(trans.deal, DEAL_SWAP);

   if(net < 0.0)
     {
      g_consecLosses++;
      if(InpMaxConsecLosses > 0 && g_consecLosses >= InpMaxConsecLosses)
        {
         g_cooldownLeft  = InpCooldownBars;
         g_consecLosses  = 0;
         PrintFormat("%d consecutive losses. Cooldown for %d bars.",
                     InpMaxConsecLosses, InpCooldownBars);
        }
     }
   else
      g_consecLosses = 0;

// Clear position bookkeeping.
   g_posTicket     = 0;
   g_posOpenBar    = 0;
   g_posRiskPoints = 0.0;
   g_posBEDone     = false;
  }

//+------------------------------------------------------------------+
//| Optimisation objective                                           |
//|                                                                   |
//| Net profit alone rewards fragile parameter sets. This scores      |
//| return per unit of equity drawdown, scaled by trade count so a    |
//| lucky 12-trade run cannot outrank a stable 300-trade one, and     |
//| returns 0 below a minimum sample size.                            |
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
