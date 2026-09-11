//+------------------------------------------------------------------+
//|                                      DarkVenus_Reconstruction.mq5 |
//|                                                                   |
//|  BEHAVIOURAL RECONSTRUCTION - NOT THE ORIGINAL SOURCE             |
//|                                                                   |
//|  This is an independent implementation of the strategy that the   |
//|  "Dark Venus" EA (Marco Solito, MQL5 Market) appears to run,      |
//|  inferred from:                                                   |
//|    - the published input parameter names and value ranges         |
//|    - the product notes (Bollinger Bands counter-trend scalper     |
//|      with grid / cumulative lot sizing)                           |
//|    - the statistical signature of a one-month M30 backtest        |
//|                                                                   |
//|  It is written to be BEHAVIOURALLY comparable, not identical.     |
//|  Trade-for-trade agreement with the real EA is not expected and   |
//|  not the point. The point is to have open, instrumentable code    |
//|  in which the tail risk can be measured directly.                 |
//|                                                                   |
//|  Every inference is marked // ASSUMPTION. Where the real EA's     |
//|  behaviour is unknown, the most common implementation of that     |
//|  idiom has been used.                                             |
//|                                                                   |
//|  WARNING: as configured by default this system has no stop loss   |
//|  and multiplies position size against an adverse move. It is      |
//|  reproduced here for measurement. Do not run it on live funds.    |
//+------------------------------------------------------------------+
#property copyright "Independent reconstruction for analysis"
#property version   "1.00"
#property description "Reconstruction of a Bollinger-band counter-trend grid EA."
#property description "For backtest analysis and tail-risk measurement only."

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Enumerations mirroring the observed dropdown options              |
//+------------------------------------------------------------------+
enum ENUM_BB_STRATEGY
  {
   BB_SELL_ABOVE_BUY_BELOW = 0,  // Sell Above and Buy Below
   BB_BUY_ABOVE_SELL_BELOW = 1,  // Buy Above and Sell Below
   BB_CROSS_UP_CENTRAL     = 2,  // Cross Up Central Band
   BB_CROSS_DOWN_CENTRAL   = 3   // Cross Down Central Band
  };

enum ENUM_SIGNAL_MOMENT
  {
   MOMENT_CURRENT_PRICE = 0,     // Current Price
   MOMENT_BAR_CLOSE     = 1      // Bar Close Price
  };

enum ENUM_GRID_MGMT
  {
   GRID_LOTS_SUM = 0,            // Lots Sum
   GRID_FIX      = 1             // Fix
  };

enum ENUM_CLOSE_MODE
  {
   CLOSE_AVERAGE_POINT          = 0, // Average Point
   CLOSE_AVERAGE_POINT_WEIGHTED = 1, // Average Point Weighted
   CLOSE_FIX_POINT              = 2  // Fix Point
  };

enum ENUM_STOP_MODE
  {
   STOP_DEFAULT  = 0,            // Default
   STOP_DISABLED = 1             // Disabled
  };

enum ENUM_CLOSURE_TYPE
  {
   CLOSURE_OPPOSITE_SIGNAL = 0,  // Close on Opposite Signal
   CLOSURE_CENTRAL_BAND    = 1   // Close on Central Band
  };

//+------------------------------------------------------------------+
//| Inputs - names and defaults mirror the screenshots               |
//+------------------------------------------------------------------+
input group "=== General ==="
input long   InpMagic                 = 8398;    // Magic Number
input int    InpMaxSpread              = 500;     // Max Spread (for open a trade)

input group "=== Money management ==="
input double InpLots                   = 0.01;    // Lots
input bool   InpMoneyManagement        = false;   // Money Management
input double InpRiskPercent            = 5.0;     // Risk Percent
input double InpMaxLotAmount           = 0.0;     // Max Lot Amount (0 = disabled)

input group "=== Bollinger Bands ==="
input bool             InpEnableBB     = true;                    // Enable Bollinger Bands
input ENUM_BB_STRATEGY InpBBStrategy   = BB_SELL_ABOVE_BUY_BELOW; // Bollinger Bands Strategies
input ENUM_SIGNAL_MOMENT InpMoment     = MOMENT_BAR_CLOSE;        // Moment of the Signal
input int              InpBBPeriod     = 20;                      // Bollinger Bands Period
input double           InpBBDeviations = 2.0;                     // Bollinger Bands Deviations
input ENUM_APPLIED_PRICE InpBBPrice    = PRICE_CLOSE;             // Bollinger Bands Price
input ENUM_TIMEFRAMES  InpBBTimeframe  = PERIOD_CURRENT;          // Bollinger Bands Timeframe

input group "=== Trading hour (broker server time) ==="
input bool   InpEnableTimeFilter       = false;   // Enable Time Filter
input int    InpStartHour              = 0;       // Trading Start Hour
input int    InpStartMinute            = 0;       // Trading Start Minute
input int    InpStopHour               = 9;       // Trading Stop Hour
input int    InpStopMinute             = 0;       // Trading Stop Minute
input bool   InpCloseOutOfHours        = false;   // Close everything Out of hours

input group "=== Trading days ==="
input bool   InpMonday                 = true;    // Monday
input bool   InpTuesday                = true;    // Tuesday
input bool   InpWednesday              = true;    // Wednesday
input bool   InpThursday               = true;    // Thursday
input bool   InpFriday                 = true;    // Friday
input bool   InpSaturday               = false;   // Saturday
input bool   InpSunday                 = false;   // Sunday

input group "=== Trading directions ==="
input bool   InpAllowBuy               = true;    // Allow Buy
input bool   InpAllowSell              = true;    // Allow Sell
input bool   InpAllowBothSides         = true;    // Allow Buy and Sell at the same time
input bool   InpAllowOtherCharts       = true;    // Allow Order From Others Charts

input group "=== Trade settings ==="
input int    InpMaxBuyOrders           = 50;      // Max Buy Orders
input int    InpMaxSellOrders          = 50;      // Max Sell Orders
input bool   InpOneTradeBar            = true;    // One Trade Bar
input bool   InpNoOpenIfClosedOnBar    = true;    // Do Not Open if Closed Order On Current Bar
input ENUM_TIMEFRAMES InpOrderTimeframe = PERIOD_CURRENT; // Order Timeframe

input group "=== Grid ==="
input bool   InpEnableGrid             = true;    // Enable Grid
input bool   InpGridComplySpread       = true;    // Grid Orders Comply Max Spread Conditions
input bool   InpGridComplyIndicators   = false;   // Grid Orders Comply Indicators Conditions
input bool   InpGridComplyHours        = false;   // Grid Orders Comply Hours Conditions
input bool   InpGridComplyDays         = false;   // Grid Orders Comply Week Days Conditions
input ENUM_GRID_MGMT InpGridMgmt       = GRID_LOTS_SUM; // Grid Management
input double InpGridCoefficient        = 1.0;     // Coefficient Grid Management
input int    InpMinDistance            = 50;      // Min Distance (for the grid orders)
input double InpMinDistanceMultiplier  = 1.0;     // Min Distance Multiplier
input bool   InpOneTradeBarGrid        = true;    // One Trade Bar Grid
input ENUM_TIMEFRAMES InpGridOrderTF   = PERIOD_CURRENT; // Grid Order Timeframe
input bool   InpMinDistanceOnAtr       = false;   // Min Distance On Atr
input int    InpAtrPeriod              = 9;       // Atr Period
input double InpAtrMultiplier          = 2.0;     // Atr Multiplier
input ENUM_TIMEFRAMES InpAtrTimeframe  = PERIOD_CURRENT; // Atr Timeframe

input group "=== Closure on indicator ==="
input bool              InpEnableCloseOnBB = false;                  // Enable Closing on Bollinger Bands
input ENUM_CLOSURE_TYPE InpClosureType     = CLOSURE_OPPOSITE_SIGNAL; // Closure Type
input bool              InpRespectSpreadOnClose = true;              // Respect Spread While Closing On Indicator

input group "=== Target settings ==="
input int             InpTakeTarget        = 50;      // Take Target (points)
input bool            InpDiffTargetFirst   = false;   // Different Take Target For First Order
input int             InpTakeTargetFirst   = 50;      // Take Target First Order (points)
input int             InpStopTarget        = 500;     // Stop Target (points)
input ENUM_STOP_MODE  InpStopTargetMode    = STOP_DISABLED; // Stop Target Mode
input bool            InpCloseOnlyEndOfBar = false;   // Close Trades Only at End Of Bar
input ENUM_CLOSE_MODE InpCloseMode         = CLOSE_AVERAGE_POINT_WEIGHTED; // Close Mode

input group "=== Monetary loss ==="
input bool   InpEnableMonetarySL       = false;   // Enable Monetary Stop Loss
input bool   InpMultiplyMonetarySL     = false;   // Multiply Monetary SL * Start Lot * 100
input bool   InpStopEAAfterMonetary    = false;   // Stop EA after Monetary loss
input double InpMonetarySLAmount       = 3000.0;  // Monetary Stop Loss Amount

input group "=== Percentage loss ==="
input bool   InpEnablePercentLoss      = false;   // Enable Close In Percentage Loss
input bool   InpStopEAAfterPercent     = false;   // Stop EA After Percentage Loss
input double InpLossPercent            = 30.0;    // Loss Amount in Percentage

input group "=== Average virtual trailing stop ==="
input bool   InpEnableAvgTrailing      = false;   // Enable Average Trailing Stop
input bool   InpTrailOnlyInProfit      = true;    // Only In Profit
input double InpBrokerCommission       = 7.0;     // Broker Commission (per lot, round turn)
input int    InpAvgTrailStopValue      = 50;      // Average Trailing Stop Value (points)
input int    InpAvgTrailStepValue      = 10;      // Average Trailing Step Value (points)

input group "=== Close trade settings ==="
input bool   InpFreezesAllFriday       = false;   // Freezes All Friday
input int    InpFreezesHour            = 18;      // Freezes Hour
input bool   InpCloseFridayNight       = false;   // Close Friday Night
input int    InpCloseFridayHour        = 18;      // Close Friday Hour
input bool   InpForcedCloseFridayNight = false;   // Forced Close Friday Night
input int    InpForcedCloseFridayHour  = 23;      // Forced Close Friday Hour

input group "=== Misc ==="
input string InpComment                = "Dark Venus"; // Custom Comment
input int    InpMinTradesOnTester      = 100;     // Minimum Trades for OnTester result
input bool   InpAllowNetting           = false;   // Allow Trading on Netting Account

//+------------------------------------------------------------------+
//| Globals                                                          |
//+------------------------------------------------------------------+
CTrade  trade;

int     hBB  = INVALID_HANDLE;
int     hAtr = INVALID_HANDLE;

double  g_point      = 0.0;
int     g_digits     = 0;
double  g_tickValue  = 0.0;
double  g_tickSize   = 0.0;
double  g_volMin     = 0.0;
double  g_volMax     = 0.0;
double  g_volStep    = 0.0;
long    g_stopsLevel = 0;

bool    g_eaStopped  = false;   // set by the "Stop EA after ..." switches

// one-trade-per-bar bookkeeping
datetime g_lastEntryBar      = 0;
datetime g_lastGridBarBuy    = 0;
datetime g_lastGridBarSell   = 0;
datetime g_lastExitBarBuy    = 0;
datetime g_lastExitBarSell   = 0;

// virtual trailing state, per side
bool    g_trailActiveBuy  = false;
bool    g_trailActiveSell = false;
double  g_trailPeakBuy    = 0.0;   // best basket profit in points seen
double  g_trailPeakSell   = 0.0;

// instrumentation: the numbers the original report hides
double  g_worstFloating   = 0.0;   // worst basket floating loss, account ccy
double  g_maxBasketLots   = 0.0;   // largest aggregate volume held
int     g_maxGridDepth    = 0;     // deepest grid reached
double  g_minMarginLevel  = 1e12;  // lowest margin level touched

//+------------------------------------------------------------------+
//| Basket description for one direction                             |
//+------------------------------------------------------------------+
struct SBasket
  {
   int      count;          // number of open orders on this side
   double   volume;         // aggregate lots
   double   avgPrice;       // simple average entry
   double   wavgPrice;      // volume-weighted average entry
   double   extremePrice;   // worst entry (lowest for buys, highest for sells)
   double   profitMoney;    // floating P/L including commission and swap
   datetime lastOpenTime;
  };

void ZeroBasket(SBasket &b)
  {
   b.count        = 0;
   b.volume       = 0.0;
   b.avgPrice     = 0.0;
   b.wavgPrice    = 0.0;
   b.extremePrice = 0.0;
   b.profitMoney  = 0.0;
   b.lastOpenTime = 0;
  }

//+------------------------------------------------------------------+
//| Utility                                                          |
//+------------------------------------------------------------------+
double PointValuePerLot()
  {
   if(g_tickSize <= 0.0)
      return(0.0);
   return(g_tickValue * (g_point / g_tickSize));
  }

// The Broker Commission input is expressed in account currency per lot.
// To fold it into a points-based average profit calculation it has to be
// converted into an equivalent number of points. Because commission and
// point value both scale linearly with volume, the volume terms cancel:
// the result is the same regardless of basket size.
// ASSUMPTION: round-turn cost per 1.00 lot.
double CommissionPoints()
  {
   double pvl = PointValuePerLot();
   if(pvl <= 0.0)
      return(0.0);
   return(InpBrokerCommission / pvl);
  }

double NormPrice(const double p)
  {
   return(NormalizeDouble(p, g_digits));
  }

int SpreadPoints()
  {
   return((int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
  }

datetime BarTime(const ENUM_TIMEFRAMES tf)
  {
   datetime t[];
   if(CopyTime(_Symbol, tf, 0, 1, t) != 1)
      return(0);
   return(t[0]);
  }

//+------------------------------------------------------------------+
//| Filters                                                          |
//+------------------------------------------------------------------+
bool DayAllowed()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   switch(dt.day_of_week)
     {
      case 0: return(InpSunday);
      case 1: return(InpMonday);
      case 2: return(InpTuesday);
      case 3: return(InpWednesday);
      case 4: return(InpThursday);
      case 5: return(InpFriday);
      case 6: return(InpSaturday);
     }
   return(false);
  }

// ASSUMPTION: the window is inclusive of start, exclusive of stop, and
// wraps across midnight when stop < start.
bool HourAllowed()
  {
   if(!InpEnableTimeFilter)
      return(true);

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int now   = dt.hour * 60 + dt.min;
   int start = InpStartHour * 60 + InpStartMinute;
   int stop  = InpStopHour  * 60 + InpStopMinute;

   if(start == stop)
      return(true);
   if(start < stop)
      return(now >= start && now < stop);
   return(now >= start || now < stop);
  }

bool SpreadAllowed()
  {
   return(SpreadPoints() <= InpMaxSpread);
  }

// "Allow Order From Others Charts, At Same Time": when false, the EA
// refuses to open if positions exist whose magic sits within +/-10 of
// ours on this symbol - i.e. a sibling instance on another chart.
bool OtherChartsAllowed()
  {
   if(InpAllowOtherCharts)
      return(true);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i) == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      long m = PositionGetInteger(POSITION_MAGIC);
      if(m == InpMagic)
         continue;
      if(MathAbs(m - InpMagic) <= 10)
         return(false);
     }
   return(true);
  }

// "Do Not Open Orders if there is Closed Order On Current Bar"
bool NoRecentCloseOnBar()
  {
   if(!InpNoOpenIfClosedOnBar)
      return(true);

   datetime barOpen = BarTime(InpOrderTimeframe);
   if(barOpen == 0)
      return(true);

   if(!HistorySelect(barOpen, TimeCurrent()))
      return(true);

   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
     {
      ulong d = HistoryDealGetTicket(i);
      if(d == 0)
         continue;
      if(HistoryDealGetString(d, DEAL_SYMBOL) != _Symbol)
         continue;
      if(HistoryDealGetInteger(d, DEAL_MAGIC) != InpMagic)
         continue;
      long entry = HistoryDealGetInteger(d, DEAL_ENTRY);
      if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_INOUT)
         return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Basket state                                                     |
//+------------------------------------------------------------------+
void ReadBasket(const ENUM_POSITION_TYPE side, SBasket &b)
  {
   ZeroBasket(b);
   double sumPrice = 0.0, sumPV = 0.0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i) == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != side)
         continue;

      double vol   = PositionGetDouble(POSITION_VOLUME);
      double price = PositionGetDouble(POSITION_PRICE_OPEN);
      datetime tm  = (datetime)PositionGetInteger(POSITION_TIME);

      b.count++;
      b.volume      += vol;
      sumPrice      += price;
      sumPV         += price * vol;
      b.profitMoney += PositionGetDouble(POSITION_PROFIT)
                       + PositionGetDouble(POSITION_SWAP);

      if(b.extremePrice == 0.0)
         b.extremePrice = price;
      else if(side == POSITION_TYPE_BUY)
         b.extremePrice = MathMin(b.extremePrice, price);
      else
         b.extremePrice = MathMax(b.extremePrice, price);

      if(tm > b.lastOpenTime)
         b.lastOpenTime = tm;
     }

   if(b.count > 0)
     {
      b.avgPrice  = sumPrice / b.count;
      b.wavgPrice = (b.volume > 0.0 ? sumPV / b.volume : b.avgPrice);
      // Commission is charged per deal; approximate the round-turn cost
      // so the basket target is net of it.
      b.profitMoney -= InpBrokerCommission * b.volume;
     }
  }

// Basket profit measured in points against whichever reference price
// the Close Mode selects.
double BasketProfitPoints(const ENUM_POSITION_TYPE side, const SBasket &b)
  {
   if(b.count == 0)
      return(0.0);

   double ref = (InpCloseMode == CLOSE_AVERAGE_POINT ? b.avgPrice : b.wavgPrice);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   if(side == POSITION_TYPE_BUY)
      return((bid - ref) / g_point);
   return((ref - ask) / g_point);
  }

//+------------------------------------------------------------------+
//| Sizing                                                           |
//+------------------------------------------------------------------+
// ASSUMPTION: the real EA's Money Management scaling is undocumented.
// The convention used here is the common one: Risk Percent of equity
// divided by a 1000-unit notional step. Do not read anything into the
// exact scaling - it exists so the input is not inert.
double BaseLot()
  {
   double lot = InpLots;

   if(InpMoneyManagement)
     {
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      lot = (equity * InpRiskPercent / 100.0) / 1000.0;
     }

   if(InpMaxLotAmount > 0.0)
      lot = MathMin(lot, InpMaxLotAmount);

   return(NormalizeLot(lot));
  }

double NormalizeLot(const double raw)
  {
   double lot = MathFloor(raw / g_volStep) * g_volStep;
   lot = MathMax(lot, g_volMin);
   lot = MathMin(lot, g_volMax);
   if(InpMaxLotAmount > 0.0)
      lot = MathMin(lot, InpMaxLotAmount);
   return(NormalizeDouble(lot, 2));
  }

// This is the mechanism that dominates the risk profile.
//   Lots Sum : next lot = (sum of open lots on this side) * coefficient
//              At coefficient 1.0 this is a strict doubling sequence.
//   Fix      : next lot = base lot * coefficient
double NextGridLot(const SBasket &b)
  {
   double lot;
   if(InpGridMgmt == GRID_LOTS_SUM)
      lot = b.volume * InpGridCoefficient;
   else
      lot = BaseLot() * InpGridCoefficient;

   if(lot < g_volMin)
      lot = g_volMin;
   return(NormalizeLot(lot));
  }

// Required adverse distance before the next grid order, in points.
// Geometric widening via Min Distance Multiplier, or ATR-scaled.
double RequiredGridDistance(const int depth)
  {
   double dist;

   if(InpMinDistanceOnAtr)
     {
      double atr[];
      if(CopyBuffer(hAtr, 0, 1, 1, atr) != 1 || atr[0] <= 0.0)
         return(0.0);
      dist = (atr[0] * InpAtrMultiplier) / g_point;
     }
   else
      dist = (double)InpMinDistance;

   if(InpMinDistanceMultiplier > 0.0 && depth > 1)
      dist *= MathPow(InpMinDistanceMultiplier, depth - 1);

   return(dist);
  }

//+------------------------------------------------------------------+
//| Signal                                                           |
//+------------------------------------------------------------------+
// Returns 1 = buy, -1 = sell, 0 = none.
// Moment of the Signal selects whether the live price or the previous
// bar's close is compared against the bands.
int Signal()
  {
   if(!InpEnableBB)
      return(0);

   double upper[], lower[], middle[];
   int shift = (InpMoment == MOMENT_BAR_CLOSE ? 1 : 0);

   if(CopyBuffer(hBB, 1, shift, 2, upper)  < 2) return(0);  // UPPER_BAND
   if(CopyBuffer(hBB, 2, shift, 2, lower)  < 2) return(0);  // LOWER_BAND
   if(CopyBuffer(hBB, 0, shift, 2, middle) < 2) return(0);  // BASE_LINE

   double px, pxPrev;
   if(InpMoment == MOMENT_BAR_CLOSE)
     {
      double c[];
      if(CopyClose(_Symbol, InpBBTimeframe, 1, 2, c) < 2)
         return(0);
      px     = c[1];   // last closed bar
      pxPrev = c[0];
     }
   else
     {
      px     = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double c[];
      if(CopyClose(_Symbol, InpBBTimeframe, 1, 1, c) < 1)
         return(0);
      pxPrev = c[0];
     }

   switch(InpBBStrategy)
     {
      case BB_SELL_ABOVE_BUY_BELOW:
         // Counter-trend: this is the observed default.
         if(px < lower[0]) return(1);
         if(px > upper[0]) return(-1);
         break;

      case BB_BUY_ABOVE_SELL_BELOW:
         if(px > upper[0]) return(1);
         if(px < lower[0]) return(-1);
         break;

      case BB_CROSS_UP_CENTRAL:
         if(pxPrev <= middle[1] && px > middle[0]) return(1);
         if(pxPrev >= middle[1] && px < middle[0]) return(-1);
         break;

      case BB_CROSS_DOWN_CENTRAL:
         if(pxPrev >= middle[1] && px < middle[0]) return(1);
         if(pxPrev <= middle[1] && px > middle[0]) return(-1);
         break;
     }

   return(0);
  }

double CentralBand()
  {
   double m[];
   if(CopyBuffer(hBB, 0, 0, 1, m) != 1)
      return(0.0);
   return(m[0]);
  }

//+------------------------------------------------------------------+
//| Order placement                                                  |
//+------------------------------------------------------------------+
bool OpenOrder(const int dir, const double lots, const bool isFirst)
  {
   if(lots <= 0.0)
      return(false);

   double sl = 0.0, tp = 0.0;

   // Fix Point close mode gives each order its own take profit rather
   // than closing the basket on an average. All other modes manage the
   // basket in code and leave TP unset.
   if(InpCloseMode == CLOSE_FIX_POINT)
     {
      int target = (isFirst && InpDiffTargetFirst ? InpTakeTargetFirst : InpTakeTarget);
      double px  = (dir > 0 ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                            : SymbolInfoDouble(_Symbol, SYMBOL_BID));
      int minStop = (int)g_stopsLevel + 2;
      if(target < minStop)
         target = minStop;
      tp = NormPrice(dir > 0 ? px + target * g_point : px - target * g_point);
     }

   bool ok = (dir > 0)
             ? trade.Buy(lots, _Symbol, 0.0, sl, tp, InpComment)
             : trade.Sell(lots, _Symbol, 0.0, sl, tp, InpComment);

   if(!ok)
     {
      // A failed grid addition is the single most important event in this
      // system: the basket is then held without further averaging and
      // without a stop. Log it loudly.
      PrintFormat("OPEN FAILED %s %.2f lots. retcode=%d (%s) | free margin %.2f",
                  (dir > 0 ? "BUY" : "SELL"), lots,
                  trade.ResultRetcode(), trade.ResultRetcodeDescription(),
                  AccountInfoDouble(ACCOUNT_MARGIN_FREE));
     }
   return(ok);
  }

void CloseBasket(const ENUM_POSITION_TYPE side, const string reason)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != side)
         continue;
      trade.PositionClose(t, 50);
     }
   if(reason != "")
      PrintFormat("Basket %s closed: %s",
                  (side == POSITION_TYPE_BUY ? "BUY" : "SELL"), reason);
  }

void CloseEverything(const string reason)
  {
   CloseBasket(POSITION_TYPE_BUY,  reason);
   CloseBasket(POSITION_TYPE_SELL, reason);
  }

//+------------------------------------------------------------------+
//| Basket management for one side                                   |
//+------------------------------------------------------------------+
void ManageSide(const int dir)
  {
   ENUM_POSITION_TYPE side = (dir > 0 ? POSITION_TYPE_BUY : POSITION_TYPE_SELL);
   SBasket b;
   ReadBasket(side, b);
   if(b.count == 0)
     {
      if(dir > 0) { g_trailActiveBuy  = false; g_trailPeakBuy  = 0.0; }
      else        { g_trailActiveSell = false; g_trailPeakSell = 0.0; }
      return;
     }

   // ---- instrumentation -------------------------------------------
   if(b.profitMoney < g_worstFloating)
      g_worstFloating = b.profitMoney;
   if(b.volume > g_maxBasketLots)
      g_maxBasketLots = b.volume;
   if(b.count > g_maxGridDepth)
      g_maxGridDepth = b.count;

   double profitPoints = BasketProfitPoints(side, b);
   double commPoints   = CommissionPoints();

   // "Close Trades Only at End Of Bar" gates every basket exit: an exit
   // is only evaluated on the first tick of a new bar. Tracked per side.
   bool exitAllowed = true;
   if(InpCloseOnlyEndOfBar)
     {
      datetime bt = BarTime(InpOrderTimeframe);
      if(dir > 0)
        {
         exitAllowed = (bt != g_lastExitBarBuy);
         if(exitAllowed)
            g_lastExitBarBuy = bt;
        }
      else
        {
         exitAllowed = (bt != g_lastExitBarSell);
         if(exitAllowed)
            g_lastExitBarSell = bt;
        }
     }

   // ---- take target ------------------------------------------------
   if(InpCloseMode != CLOSE_FIX_POINT && exitAllowed)
     {
      int target = (b.count == 1 && InpDiffTargetFirst
                    ? InpTakeTargetFirst : InpTakeTarget);
      if(profitPoints >= target + commPoints)
        {
         CloseBasket(side, StringFormat("take target %d pts reached (%.0f net)",
                                        target, profitPoints));
         return;
        }
     }

   // ---- basket stop target ----------------------------------------
   // Observed default is Disabled. That is the whole risk story.
   if(InpStopTargetMode == STOP_DEFAULT && exitAllowed)
     {
      if(profitPoints <= -(double)InpStopTarget)
        {
         CloseBasket(side, StringFormat("stop target %d pts hit", InpStopTarget));
         return;
        }
     }

   // ---- average virtual trailing stop ------------------------------
   if(InpEnableAvgTrailing)
     {
      bool   active = (dir > 0 ? g_trailActiveBuy : g_trailActiveSell);
      double peak   = (dir > 0 ? g_trailPeakBuy   : g_trailPeakSell);
      double net    = profitPoints - commPoints;
      bool   doClose = false;

      if(!active && net >= InpAvgTrailStopValue)
        {
         active = true;
         peak   = net;
        }
      else if(active)
        {
         if(net > peak + InpAvgTrailStepValue)
            peak = net;
         double trigger = peak - InpAvgTrailStopValue;
         if(InpTrailOnlyInProfit && trigger < commPoints)
            trigger = commPoints;
         if(net <= trigger)
            doClose = true;
        }

      if(dir > 0) { g_trailActiveBuy  = active; g_trailPeakBuy  = peak; }
      else        { g_trailActiveSell = active; g_trailPeakSell = peak; }

      if(doClose)
        {
         CloseBasket(side, StringFormat("virtual trail: %.0f pts from peak %.0f",
                                        net, peak));
         return;
        }
     }

   // ---- closure on indicator ---------------------------------------
   if(InpEnableCloseOnBB)
     {
      if(InpRespectSpreadOnClose && !SpreadAllowed())
        { /* wait for a tighter spread */ }
      else if(InpClosureType == CLOSURE_CENTRAL_BAND)
        {
         double mid = CentralBand();
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(mid > 0.0)
           {
            if(dir > 0 && bid >= mid) { CloseBasket(side, "central band"); return; }
            if(dir < 0 && bid <= mid) { CloseBasket(side, "central band"); return; }
           }
        }
      else
        {
         int sig = Signal();
         if((dir > 0 && sig == -1) || (dir < 0 && sig == 1))
           { CloseBasket(side, "opposite signal"); return; }
        }
     }

   // ---- grid addition ----------------------------------------------
   if(!InpEnableGrid)
      return;

   int maxOrders = (dir > 0 ? InpMaxBuyOrders : InpMaxSellOrders);
   if(b.count >= maxOrders)
      return;

   if(InpGridComplySpread && !SpreadAllowed())
      return;
   if(InpGridComplyHours && !HourAllowed())
      return;
   if(InpGridComplyDays && !DayAllowed())
      return;
   if(InpGridComplyIndicators)
     {
      int sig = Signal();
      if(sig != dir)
         return;
     }

   if(InpOneTradeBarGrid)
     {
      datetime bt      = BarTime(InpGridOrderTF);
      datetime lastBar = (dir > 0 ? g_lastGridBarBuy : g_lastGridBarSell);
      if(bt == lastBar)
         return;
     }

   double needPts = RequiredGridDistance(b.count);
   if(needPts <= 0.0)
      return;

   double px      = (dir > 0 ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                             : SymbolInfoDouble(_Symbol, SYMBOL_BID));
   double adverse = (dir > 0 ? (b.extremePrice - px) : (px - b.extremePrice)) / g_point;

   if(adverse < needPts)
      return;

   double lot = NextGridLot(b);
   if(OpenOrder(dir, lot, false))
     {
      if(dir > 0) g_lastGridBarBuy  = BarTime(InpGridOrderTF);
      else        g_lastGridBarSell = BarTime(InpGridOrderTF);

      PrintFormat("GRID %s level %d: %.2f lots (basket now %.2f lots, %.0f pts adverse)",
                  (dir > 0 ? "BUY" : "SELL"), b.count + 1, lot, b.volume + lot, adverse);
     }
  }

//+------------------------------------------------------------------+
//| Account-level loss guards                                        |
//+------------------------------------------------------------------+
void CheckLossGuards()
  {
   SBasket bb, bs;
   ReadBasket(POSITION_TYPE_BUY,  bb);
   ReadBasket(POSITION_TYPE_SELL, bs);
   double floating = bb.profitMoney + bs.profitMoney;

   if(InpEnableMonetarySL)
     {
      double limit = InpMonetarySLAmount;
      if(InpMultiplyMonetarySL)
         limit = InpMonetarySLAmount * InpLots * 100.0;
      if(floating <= -limit)
        {
         CloseEverything(StringFormat("monetary stop loss %.2f", limit));
         if(InpStopEAAfterMonetary)
           {
            g_eaStopped = true;
            Print("EA stopped by monetary loss switch.");
           }
         return;
        }
     }

   if(InpEnablePercentLoss)
     {
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      if(balance > 0.0)
        {
         double lossPct = -floating / balance * 100.0;
         if(lossPct >= InpLossPercent)
           {
            CloseEverything(StringFormat("percentage loss %.2f%%", lossPct));
            if(InpStopEAAfterPercent)
              {
               g_eaStopped = true;
               Print("EA stopped by percentage loss switch.");
              }
           }
        }
     }
  }

void CheckCalendarClosures()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);

   if(InpCloseOutOfHours && !HourAllowed())
     {
      CloseEverything("out of hours");
      return;
     }

   if(dt.day_of_week != 5)
      return;

   if(InpForcedCloseFridayNight && dt.hour >= InpForcedCloseFridayHour)
     {
      CloseEverything("forced Friday close");
      return;
     }

   if(InpCloseFridayNight && dt.hour >= InpCloseFridayHour)
     {
      // ASSUMPTION: closes only baskets that are in profit, leaving
      // losing baskets open. This matches the distinction between
      // "Close Friday Night" and "Forced Close Friday Night".
      SBasket b;
      ReadBasket(POSITION_TYPE_BUY, b);
      if(b.count > 0 && b.profitMoney > 0.0)
         CloseBasket(POSITION_TYPE_BUY, "Friday close in profit");
      ReadBasket(POSITION_TYPE_SELL, b);
      if(b.count > 0 && b.profitMoney > 0.0)
         CloseBasket(POSITION_TYPE_SELL, "Friday close in profit");
     }
  }

bool FridayFreeze()
  {
   if(!InpFreezesAllFriday)
      return(false);
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   return(dt.day_of_week == 5 && dt.hour >= InpFreezesHour);
  }

//+------------------------------------------------------------------+
//| Entry                                                            |
//+------------------------------------------------------------------+
void TryNewEntry()
  {
   if(!DayAllowed() || !HourAllowed() || !SpreadAllowed())
      return;
   if(!OtherChartsAllowed() || !NoRecentCloseOnBar())
      return;
   if(FridayFreeze())
      return;

   if(InpOneTradeBar)
     {
      datetime bt = BarTime(InpOrderTimeframe);
      if(bt == g_lastEntryBar)
         return;
     }

   int sig = Signal();
   if(sig == 0)
      return;
   if(sig > 0 && !InpAllowBuy)
      return;
   if(sig < 0 && !InpAllowSell)
      return;

   SBasket bb, bs;
   ReadBasket(POSITION_TYPE_BUY,  bb);
   ReadBasket(POSITION_TYPE_SELL, bs);

   // A basket already exists on this side: grid logic owns it, not entry.
   if(sig > 0 && bb.count > 0)
      return;
   if(sig < 0 && bs.count > 0)
      return;

   if(!InpAllowBothSides)
     {
      if(sig > 0 && bs.count > 0)
         return;
      if(sig < 0 && bb.count > 0)
         return;
     }

   if(OpenOrder(sig, BaseLot(), true))
      g_lastEntryBar = BarTime(InpOrderTimeframe);
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

   if(g_point <= 0.0 || g_volStep <= 0.0 || g_tickSize <= 0.0)
     {
      Print("Init failed: incomplete symbol specification.");
      return(INIT_FAILED);
     }

   long marginMode = AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(marginMode == ACCOUNT_MARGIN_MODE_RETAIL_NETTING && !InpAllowNetting)
     {
      Print("Init failed: netting account. This logic requires hedging, "
            "or set Allow Trading on Netting Account.");
      return(INIT_FAILED);
     }

   hBB  = iBands(_Symbol, InpBBTimeframe, InpBBPeriod, 0, InpBBDeviations, InpBBPrice);
   hAtr = iATR(_Symbol, InpAtrTimeframe, InpAtrPeriod);
   if(hBB == INVALID_HANDLE || hAtr == INVALID_HANDLE)
     {
      Print("Init failed: indicator handles.");
      return(INIT_FAILED);
     }

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(50);
   trade.SetTypeFillingBySymbol(_Symbol);
   trade.SetAsyncMode(false);
   trade.LogLevel(LOG_LEVEL_ERRORS);

   if(InpStopTargetMode == STOP_DISABLED && !InpEnableMonetarySL && !InpEnablePercentLoss)
      Print("NOTE: no basket stop, no monetary stop, no percentage stop. "
            "Loss is bounded only by margin. This is the default configuration.");

   if(InpEnableGrid && InpGridMgmt == GRID_LOTS_SUM && InpGridCoefficient >= 1.0)
      PrintFormat("NOTE: Lots Sum at coefficient %.2f doubles or more at every "
                  "grid level. Level 10 from %.2f lots is %.2f lots aggregate.",
                  InpGridCoefficient, InpLots, InpLots * MathPow(2.0, 9));

   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   if(hBB  != INVALID_HANDLE) IndicatorRelease(hBB);
   if(hAtr != INVALID_HANDLE) IndicatorRelease(hAtr);

   PrintFormat("=== Reconstruction diagnostics ===");
   PrintFormat("Worst basket floating loss : %.2f %s",
               g_worstFloating, AccountInfoString(ACCOUNT_CURRENCY));
   PrintFormat("Largest aggregate volume   : %.2f lots", g_maxBasketLots);
   PrintFormat("Deepest grid reached       : %d orders", g_maxGridDepth);
   PrintFormat("Lowest margin level        : %.2f%%",
               (g_minMarginLevel > 1e11 ? 0.0 : g_minMarginLevel));
  }

void OnTick()
  {
   double ml = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   if(ml > 0.0 && ml < g_minMarginLevel)
      g_minMarginLevel = ml;

   CheckLossGuards();
   CheckCalendarClosures();

   ManageSide(1);
   ManageSide(-1);

   if(g_eaStopped)
      return;

   TryNewEntry();
  }

//+------------------------------------------------------------------+
//| Optimisation objective                                           |
//|                                                                   |
//| Deliberately penalises the failure mode this strategy hides:      |
//| a high profit factor sitting on top of an unbounded floating      |
//| loss. Scores net profit per unit of equity drawdown.              |
//+------------------------------------------------------------------+
double OnTester()
  {
   double trades = TesterStatistics(STAT_TRADES);
   if(trades < InpMinTradesOnTester)
      return(0.0);

   double net = TesterStatistics(STAT_PROFIT);
   if(net <= 0.0)
      return(0.0);

   double ddPct = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   if(ddPct < 0.1)
      ddPct = 0.1;

   return(net / ddPct);
  }
//+------------------------------------------------------------------+
