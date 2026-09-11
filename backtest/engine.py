"""Event-driven backtest engine and strategy replicas.

Execution model
---------------
Bars are M1. Quoted OHLC is treated as BID, exactly as MetaTrader 5 exports it.
Ask is bid + spread for that bar.

  * A buy fills at ask, and its stop loss and take profit trigger on bid.
  * A sell fills at bid, and its stop loss and take profit trigger on ask.
  * If a bar's range covers both the stop and the target, the STOP is taken.
    That is the conservative choice and it is the single biggest reason a
    replay like this reads worse than an optimistic tester.
  * Market orders pay `slippage_pts` against the trader, every time.
  * Commission is charged per lot round turn, pro-rated on partial closes.
  * Margin is tracked and a stop-out closes everything below the stop-out
    level. That matters only for the grid strategy, which is the point.

Signals are evaluated on the first M1 bar of each new signal-timeframe bar,
using indicator values from the LAST CLOSED signal bar. There is no intrabar
peeking anywhere.
"""

import numpy as np

import indicators as ind


# ----------------------------------------------------------------------------
# Broker
# ----------------------------------------------------------------------------
class Position:
    __slots__ = ("ticket", "dir", "lots", "init_lots", "open_price", "sl", "tp",
                 "open_i", "open_sig_bar", "risk_points", "partial_done",
                 "be_done", "realised", "comm_paid", "cost_spread", "cost_slip",
                 "tag")

    def __init__(self, ticket, direction, lots, price, sl, tp, i, sig_bar, tag=""):
        self.ticket = ticket
        self.dir = direction
        self.lots = lots
        self.init_lots = lots
        self.open_price = price
        self.sl = sl
        self.tp = tp
        self.open_i = i
        self.open_sig_bar = sig_bar
        self.risk_points = abs(price - sl) if sl > 0 else 0.0
        self.partial_done = False
        self.be_done = False
        self.realised = 0.0        # banked P/L from partial closes
        self.comm_paid = 0.0
        self.cost_spread = 0.0
        self.cost_slip = 0.0
        self.tag = tag


class Broker:
    def __init__(self, tape, balance=10_000.0, leverage=100, commission_rt=7.0,
                 slippage_pts=2, stopout_level=50.0, vol_min=0.01, vol_step=0.01,
                 vol_max=100.0, stops_level=0):
        self.tape = tape
        self.spec = tape.spec
        self.point = tape.point
        self.balance = balance
        self.start_balance = balance
        self.leverage = leverage
        self.commission_rt = commission_rt
        self.slip = slippage_pts * tape.point
        self.stopout_level = stopout_level
        self.vol_min, self.vol_step, self.vol_max = vol_min, vol_step, vol_max
        self.stops_level = stops_level

        self.positions = []
        self.cur_spread = 0.0          # spread in points for the bar being processed
        self.slip_pts = slippage_pts
        self._next_ticket = 1
        self.trades = []           # one record per closed position
        self.stopouts = 0

        self.equity_curve = np.empty(len(tape))
        self.peak_equity = balance
        self.max_dd_pct = 0.0
        self.worst_floating = 0.0
        self.max_open_lots = 0.0
        self.max_basket_n = 0
        self.min_margin_level = float("inf")

    # --- valuation -------------------------------------------------------
    def point_value_per_lot(self, price):
        """Account-currency value of one point of one lot."""
        c = self.spec["contract"]
        if self.spec["quote"] == "USD":
            return c * self.point
        return c * self.point / price          # USD/JPY and friends

    def notional_usd_per_lot(self, price):
        c = self.spec["contract"]
        return c * price if self.spec["quote"] == "USD" else c

    def floating(self, bid, ask):
        total = 0.0
        for p in self.positions:
            px = bid if p.dir > 0 else ask
            pv = self.point_value_per_lot(px)
            total += (px - p.open_price) * p.dir / self.point * pv * p.lots
        return total

    def used_margin(self, price):
        lots = sum(p.lots for p in self.positions)
        if lots <= 0:
            return 0.0
        return self.notional_usd_per_lot(price) * lots / self.leverage

    def equity(self, bid, ask):
        return self.balance + self.floating(bid, ask)

    def normalise_lots(self, lots):
        lots = np.floor(lots / self.vol_step + 1e-9) * self.vol_step
        return round(min(lots, self.vol_max), 8)

    # --- orders ----------------------------------------------------------
    def open(self, direction, lots, bid, ask, i, sig_bar, sl=0.0, tp=0.0, tag=""):
        lots = self.normalise_lots(lots)
        if lots < self.vol_min - 1e-12:
            return None
        price = (ask + self.slip) if direction > 0 else (bid - self.slip)

        need = self.notional_usd_per_lot(price) * lots / self.leverage
        free = self.equity(bid, ask) - self.used_margin(price)
        if need > free:
            return None

        p = Position(self._next_ticket, direction, lots, price, sl, tp, i, sig_bar, tag)
        pv = self.point_value_per_lot(price)
        p.cost_spread += lots * self.cur_spread * pv * 0.5
        p.cost_slip += lots * self.slip_pts * pv
        self._next_ticket += 1
        self.positions.append(p)
        return p

    def _book(self, p, close_price, lots, reason, i, slipped=True):
        pv = self.point_value_per_lot(close_price)
        p.cost_spread += lots * self.cur_spread * pv * 0.5
        if slipped:
            p.cost_slip += lots * self.slip_pts * pv
        gross = (close_price - p.open_price) * p.dir / self.point * pv * lots
        comm = self.commission_rt * lots
        self.balance += gross - comm
        p.realised += gross - comm
        p.comm_paid += comm
        p.lots = round(p.lots - lots, 8)

        if p.lots <= 1e-9:
            self.positions.remove(p)
            self.trades.append(dict(
                ticket=p.ticket, dir=p.dir, lots=p.init_lots,
                open_i=p.open_i, close_i=i, open_price=p.open_price,
                close_price=close_price, net=p.realised, comm=p.comm_paid,
                spread_cost=p.cost_spread, slip_cost=p.cost_slip,
                reason=reason, tag=p.tag,
                bars=i - p.open_i,
            ))
            return True
        return False

    def close(self, p, bid, ask, reason, i, lots=None):
        px = (bid - self.slip) if p.dir > 0 else (ask + self.slip)
        return self._book(p, px, p.lots if lots is None else lots, reason, i)

    def close_all(self, bid, ask, reason, i):
        for p in list(self.positions):
            self.close(p, bid, ask, reason, i)

    # --- per-bar processing ---------------------------------------------
    def process_bar(self, i, o, h, l, c, spread):
        """Fill stops and targets inside bar i. Stop wins ties."""
        sp = spread * self.point
        for p in list(self.positions):
            if p.dir > 0:
                hit_sl = p.sl > 0 and l <= p.sl
                hit_tp = p.tp > 0 and h >= p.tp
                if hit_sl:
                    self._book(p, min(p.sl, o) - self.slip, p.lots, "sl", i)
                elif hit_tp:
                    self._book(p, max(p.tp, o), p.lots, "tp", i, slipped=False)
            else:
                ah, al, ao = h + sp, l + sp, o + sp
                hit_sl = p.sl > 0 and ah >= p.sl
                hit_tp = p.tp > 0 and al <= p.tp
                if hit_sl:
                    self._book(p, max(p.sl, ao) + self.slip, p.lots, "sl", i)
                elif hit_tp:
                    self._book(p, min(p.tp, ao), p.lots, "tp", i, slipped=False)

    def mark(self, i, bid, ask):
        eq = self.equity(bid, ask)
        self.equity_curve[i] = eq
        if eq > self.peak_equity:
            self.peak_equity = eq
        dd = (self.peak_equity - eq) / self.peak_equity * 100.0 if self.peak_equity > 0 else 0.0
        if dd > self.max_dd_pct:
            self.max_dd_pct = dd

        fl = eq - self.balance
        if fl < self.worst_floating:
            self.worst_floating = fl
        lots = sum(p.lots for p in self.positions)
        if lots > self.max_open_lots:
            self.max_open_lots = lots
        if len(self.positions) > self.max_basket_n:
            self.max_basket_n = len(self.positions)

        if self.positions:
            um = self.used_margin(bid)
            if um > 0:
                ml = eq / um * 100.0
                if ml < self.min_margin_level:
                    self.min_margin_level = ml
                if ml < self.stopout_level:
                    self.close_all(bid, ask, "stopout", i)
                    self.stopouts += 1
        return eq


# ----------------------------------------------------------------------------
# Shared context: indicators on every timeframe the strategies need
# ----------------------------------------------------------------------------
class Context:
    def __init__(self, tape, sig_tf=5, regime_tf=60, cfg=None):
        self.tape = tape
        self.sig_tf = sig_tf
        cfg = cfg or {}

        self.sig_id, so, sh, sl, sc, sstart = tape.resample(sig_tf)
        self.so, self.sh, self.sl, self.sc = so, sh, sl, sc
        self.sig_start = np.zeros(len(tape), dtype=bool)
        self.sig_start[sstart] = True

        self.reg_id, ro, rh, rl, rc, _ = tape.resample(regime_tf)

        self.ema = ind.ema(sc, cfg.get("ema_period", 20))
        self.atr = ind.atr(sh, sl, sc, cfg.get("atr_period", 14))
        self.atr_slow = ind.atr(sh, sl, sc, cfg.get("atr_slow_period", 100))
        self.rsi = ind.rsi(sc, cfg.get("rsi_period", 2))
        self.adx = ind.adx(rh, rl, rc, cfg.get("adx_period", 14))
        self.bias_ema = ind.ema(rc, cfg.get("bias_ema_period", 200))
        self.reg_close = rc
        mid, up, lo = ind.bollinger(sc, cfg.get("band_period", 20),
                                    cfg.get("band_dev", 2.0))
        self.bb_mid, self.bb_up, self.bb_lo = mid, up, lo

        self.hour = ((tape.t // 60) % 24).astype(np.int32)
        self.dow = (((tape.t // 1440) + 3) % 7).astype(np.int32)   # 0=Sun..6=Sat
        self.day_id = (tape.t // 1440).astype(np.int64)
        self.week_id = ((tape.t // 1440) - ((self.dow + 6) % 7)).astype(np.int64)


# ----------------------------------------------------------------------------
# SnapScalp v2
# ----------------------------------------------------------------------------
V2 = dict(
    stretch_atr=1.30, rsi_buy_below=8.0, rsi_sell_above=92.0, require_rejection=True,
    entry_model="both",
    band_rsi_buy_below=25.0, band_rsi_sell_above=75.0, band_min_stretch_atr=0.60,
    adx_max=28.0, use_regime=True, bias_mode="none",
    min_atr_ratio=0.50, max_atr_ratio=2.50,
    sl_atr=1.60, tp_mode="mean", tp_atr=1.10,
    partial_pct=50.0, partial_atr=0.60,
    be_at_r=0.40, be_offset_r=0.10, trail_start_r=0.80, trail_atr=1.00,
    max_bars=18, time_stop_keep_r=0.30,
    max_spread_pts=20, min_tp_cost_ratio=2.50,
    risk_pct=0.75, max_lots=5.0,
    losses_to_reduce=2, risk_mult_loss=0.50, wins_to_boost=3, risk_mult_win=1.25,
    max_trades_day=8, min_bars_between=1,
    daily_loss_pct=2.0, weekly_loss_pct=4.0, max_equity_dd_pct=10.0,
    flatten_on_guard=True, max_consec_losses=3, cooldown_bars=24,
    use_sessions=True, sess=((7, 11), (13, 17)), use_sess3=False, sess3=(2, 6),
    avoid_rollover=True, rollover=(23, 1),
    trade_monday=True, trade_friday=True, friday_stop_hour=19,
    friday_flatten=True, friday_close_hour=20,
)

# SnapScalp v1: the original, for a like-for-like comparison.
V1 = dict(
    V2,
    entry_model="stretch", tp_mode="atr", tp_atr=1.10,
    partial_pct=0.0, be_at_r=0.60, trail_start_r=1.00, trail_atr=1.20,
    max_bars=18, time_stop_keep_r=-1e9,        # v1 closes regardless of P/L
    max_spread_pts=25, min_tp_cost_ratio=3.00,
    risk_pct=0.50, max_lots=0.30,
    losses_to_reduce=0, wins_to_boost=0,
    max_trades_day=6, min_bars_between=0,
    min_atr_ratio=0.0, max_atr_ratio=0.0,      # no volatility filter
    use_sess3=False,
)

PRESET_SPREAD = {"EURUSD": 12, "GBPUSD": 20, "USDJPY": 15}


def _in_window(hour, start, end):
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


class SnapScalp:
    """Replica of SnapScalp_MR (v1 and v2 share this code path)."""

    def __init__(self, broker, ctx, cfg, version=2, use_preset=True):
        self.b = broker
        self.x = ctx
        self.p = dict(cfg)
        self.version = version
        if use_preset and version == 2:
            self.p["max_spread_pts"] = PRESET_SPREAD.get(broker.tape.symbol,
                                                         self.p["max_spread_pts"])
            if broker.tape.symbol == "USDJPY":
                self.p["use_sess3"] = True

        self.day_start_eq = broker.balance
        self.week_start_eq = broker.balance
        self.peak_eq = broker.balance
        self.cur_day = None
        self.cur_week = None
        self.trades_today = 0
        self.consec_losses = 0
        self.consec_wins = 0
        self.cooldown = 0
        self.day_halt = self.week_halt = self.hard_halt = False
        self.last_exit_sig_bar = -10**9
        self.pos = None
        self.rejects = dict(spread=0, cost=0, lots=0, minstop=0, adx_bars=0, vol=0,
                            bias=0, session=0, guard=0)
        self.guard_days = 0
        self.signals = 0

    # --- helpers ---------------------------------------------------------
    def comm_points(self, price):
        pv = self.b.point_value_per_lot(price)
        return self.b.commission_rt / pv if pv > 0 else 0.0

    def eff_risk(self):
        p = self.p
        if p["losses_to_reduce"] > 0 and self.consec_losses >= p["losses_to_reduce"]:
            return p["risk_pct"] * p["risk_mult_loss"]
        if p["wins_to_boost"] > 0 and self.consec_wins >= p["wins_to_boost"]:
            return p["risk_pct"] * p["risk_mult_win"]
        return p["risk_pct"]

    def lots_for_risk(self, sl_points, price, equity):
        pv = self.b.point_value_per_lot(price)
        risk_money = equity * self.eff_risk() / 100.0
        loss_per_lot = sl_points * pv + self.b.commission_rt
        if loss_per_lot <= 0:
            return 0.0
        lots = self.b.normalise_lots(min(risk_money / loss_per_lot, self.p["max_lots"]))
        return lots if lots >= self.b.vol_min - 1e-12 else 0.0

    def session_ok(self, i):
        x, p = self.x, self.p
        dow, hour = x.dow[i], x.hour[i]
        if dow in (0, 6):
            return False
        if dow == 1 and not p["trade_monday"]:
            return False
        if dow == 5:
            if not p["trade_friday"] or hour >= p["friday_stop_hour"]:
                return False
        if p["avoid_rollover"] and _in_window(hour, *p["rollover"]):
            return False
        if not p["use_sessions"]:
            return True
        for s, e in p["sess"]:
            if _in_window(hour, s, e):
                return True
        if p["use_sess3"] and _in_window(hour, *p["sess3"]):
            return True
        return False

    # --- guards ----------------------------------------------------------
    def update_guards(self, i, equity, bid, ask):
        x, p = self.x, self.p
        if self.cur_day != x.day_id[i]:
            self.cur_day = x.day_id[i]
            self.day_start_eq = equity
            self.trades_today = 0
            self.day_halt = False
        if self.cur_week != x.week_id[i]:
            self.cur_week = x.week_id[i]
            self.week_start_eq = equity
            self.week_halt = False
        if equity > self.peak_eq:
            self.peak_eq = equity

        tripped = False
        if not self.day_halt and p["daily_loss_pct"] > 0 and self.day_start_eq > 0:
            if (self.day_start_eq - equity) / self.day_start_eq * 100.0 >= p["daily_loss_pct"]:
                self.day_halt = True
                self.guard_days += 1
                tripped = True
        if not self.week_halt and p["weekly_loss_pct"] > 0 and self.week_start_eq > 0:
            if (self.week_start_eq - equity) / self.week_start_eq * 100.0 >= p["weekly_loss_pct"]:
                self.week_halt = True
                tripped = True
        if not self.hard_halt and p["max_equity_dd_pct"] > 0 and self.peak_eq > 0:
            if (self.peak_eq - equity) / self.peak_eq * 100.0 >= p["max_equity_dd_pct"]:
                self.hard_halt = True
                tripped = True

        if p["flatten_on_guard"] and (self.day_halt or self.week_halt or self.hard_halt):
            if self.pos is not None and self.pos in self.b.positions:
                self.b.close(self.pos, bid, ask, "guard", i)
        return tripped

    def entries_enabled(self, sig_bar):
        p = self.p
        if self.hard_halt or self.day_halt or self.week_halt or self.cooldown > 0:
            return False
        if p["max_trades_day"] > 0 and self.trades_today >= p["max_trades_day"]:
            return False
        if p["min_bars_between"] > 0 and sig_bar - self.last_exit_sig_bar < p["min_bars_between"]:
            return False
        return True

    # --- signal ----------------------------------------------------------
    def signal(self, sb):
        """sb = index of the last CLOSED signal bar."""
        x, p = self.x, self.p
        if sb < 2:
            return 0, ""
        atr1, ema1 = x.atr[sb], x.ema[sb]
        if not np.isfinite(atr1) or atr1 <= 0 or not np.isfinite(ema1):
            return 0, ""

        if p["use_regime"]:
            adx = self.cur_adx
            if not np.isfinite(adx):
                return 0, ""
            if adx > p["adx_max"]:
                self.rejects["adx_bars"] += 1
                return 0, ""

        if p["min_atr_ratio"] > 0 or p["max_atr_ratio"] > 0:
            slow = x.atr_slow[sb]
            if np.isfinite(slow) and slow > 0:
                ratio = atr1 / slow
                if p["min_atr_ratio"] > 0 and ratio < p["min_atr_ratio"]:
                    self.rejects["vol"] += 1
                    return 0, ""
                if p["max_atr_ratio"] > 0 and ratio > p["max_atr_ratio"]:
                    self.rejects["vol"] += 1
                    return 0, ""

        c1, o1 = x.sc[sb], x.so[sb]
        rsi1 = x.rsi[sb]
        d, model = 0, ""

        if p["entry_model"] in ("stretch", "both"):
            stretch = p["stretch_atr"] * atr1
            if c1 < ema1 - stretch and rsi1 < p["rsi_buy_below"]:
                if not p["require_rejection"] or c1 > o1:
                    d, model = 1, "A"
            elif c1 > ema1 + stretch and rsi1 > p["rsi_sell_above"]:
                if not p["require_rejection"] or c1 < o1:
                    d, model = -1, "A"

        if d == 0 and p["entry_model"] in ("band", "both"):
            up1, up2 = x.bb_up[sb], x.bb_up[sb - 1]
            lo1, lo2 = x.bb_lo[sb], x.bb_lo[sb - 1]
            c2 = x.sc[sb - 1]
            rsi2 = x.rsi[sb - 1]
            ms = p["band_min_stretch_atr"] * atr1
            if np.isfinite(lo1) and np.isfinite(lo2):
                if (c2 < lo2 and c1 > lo1 and c1 > o1 and c1 < ema1 - ms
                        and rsi2 < p["band_rsi_buy_below"]):
                    d, model = 1, "B"
                elif (c2 > up2 and c1 < up1 and c1 < o1 and c1 > ema1 + ms
                      and rsi2 > p["band_rsi_sell_above"]):
                    d, model = -1, "B"

        if d != 0 and p["bias_mode"] == "with_trend":
            be, bc = self.cur_bias_ema, self.cur_bias_close
            if np.isfinite(be):
                ok = (bc > be) if d > 0 else (bc < be)
                if not ok:
                    self.rejects["bias"] += 1
                    return 0, ""
        return d, model

    # --- entry -----------------------------------------------------------
    def try_enter(self, i, sb, d, model, bid, ask, spread, equity):
        x, p, b = self.x, self.p, self.b
        if spread > p["max_spread_pts"]:
            self.rejects["spread"] += 1
            return
        atr1, ema1 = x.atr[sb], x.ema[sb]
        entry = ask if d > 0 else bid
        sl_pts = p["sl_atr"] * atr1 / b.point

        if p["tp_mode"] == "mean":
            tp_pts = (ema1 - entry) * d / b.point
        else:
            tp_pts = p["tp_atr"] * atr1 / b.point

        min_stop = b.stops_level + 2
        sl_pts = max(sl_pts, min_stop)
        if tp_pts < min_stop:
            self.rejects["minstop"] += 1
            return

        cost_pts = spread + self.comm_points(entry)
        if tp_pts < p["min_tp_cost_ratio"] * cost_pts:
            self.rejects["cost"] += 1
            return

        partial_pts = 0.0
        if p["partial_pct"] > 0 and p["partial_atr"] > 0:
            cand = p["partial_atr"] * atr1 / b.point
            if cand >= min_stop and cand >= 1.5 * cost_pts and cand < tp_pts:
                partial_pts = cand

        lots = self.lots_for_risk(sl_pts, entry, equity)
        if lots <= 0:
            self.rejects["lots"] += 1
            return

        sl = entry - d * sl_pts * b.point
        tp = entry + d * tp_pts * b.point
        pos = b.open(d, lots, bid, ask, i, sb, sl, tp, tag=model)
        if pos is None:
            self.rejects["lots"] += 1
            return
        pos.partial_done = (partial_pts <= 0)
        pos.risk_points = sl_pts
        self.partial_pts = partial_pts
        self.pos = pos
        self.trades_today += 1

    # --- management ------------------------------------------------------
    def manage(self, i, sb, bid, ask, new_sig_bar):
        x, p, b = self.x, self.p, self.b
        pos = self.pos
        if pos is None or pos not in b.positions or sb < 1:
            return
        atr1, ema1 = x.atr[sb], x.ema[sb]
        if not np.isfinite(atr1) or atr1 <= 0:
            return

        is_long = pos.dir > 0
        cur = bid if is_long else ask
        if pos.risk_points <= 0:
            return
        move_pts = (cur - pos.open_price) * pos.dir / b.point
        r = move_pts / pos.risk_points

        # weekend flatten
        if p["friday_flatten"] and x.dow[i] == 5 and x.hour[i] >= p["friday_close_hour"]:
            b.close(pos, bid, ask, "friday", i)
            return

        # time stop
        if p["max_bars"] > 0:
            bars = sb - pos.open_sig_bar
            if (bars >= p["max_bars"] and r < p["time_stop_keep_r"]) or bars >= 2 * p["max_bars"]:
                b.close(pos, bid, ask, "time", i)
                return

        # mean reached
        if p["tp_mode"] == "mean" and np.isfinite(ema1):
            if (is_long and cur >= ema1) or (not is_long and cur <= ema1):
                b.close(pos, bid, ask, "mean", i)
                return

        # scale-out
        if not pos.partial_done and self.partial_pts > 0 and move_pts >= self.partial_pts:
            cv = b.normalise_lots(pos.init_lots * p["partial_pct"] / 100.0)
            rem = round(pos.lots - cv, 8)
            if cv >= b.vol_min - 1e-12 and rem >= b.vol_min - 1e-12:
                b.close(pos, bid, ask, "partial", i, lots=cv)
                pos.partial_done = True
                if pos not in b.positions:
                    return
            else:
                pos.partial_done = True

        new_sl = pos.sl
        want_be = False
        be_cand = 0.0
        if not pos.be_done and ((p["be_at_r"] > 0 and r >= p["be_at_r"]) or pos.partial_done):
            want_be = True
            lock = p["be_offset_r"] * pos.risk_points * b.point
            be_cand = pos.open_price + pos.dir * lock
            if (is_long and (pos.sl <= 0 or be_cand > pos.sl)) or \
               (not is_long and (pos.sl <= 0 or be_cand < pos.sl)):
                new_sl = be_cand

        if p["trail_start_r"] > 0 and p["trail_atr"] > 0 and r >= p["trail_start_r"]:
            dist = p["trail_atr"] * atr1
            cand = cur - pos.dir * dist
            if (is_long and cand > new_sl) or (not is_long and (new_sl <= 0 or cand < new_sl)):
                new_sl = cand

        min_dist = (b.stops_level + 2) * b.point
        if new_sl != pos.sl and new_sl > 0:
            ok = (cur - new_sl >= min_dist) if is_long else (new_sl - cur >= min_dist)
            if ok:
                pos.sl = new_sl
                if want_be:
                    pos.be_done = (new_sl >= be_cand - b.point * 0.5) if is_long \
                        else (new_sl <= be_cand + b.point * 0.5)

        if new_sig_bar and p["tp_mode"] == "mean" and np.isfinite(ema1):
            cand = ema1
            ok = (cand - cur >= min_dist) if is_long else (cur - cand >= min_dist)
            if ok:
                pos.tp = cand

    # --- settle ----------------------------------------------------------
    def settle(self, sb):
        tr = self.b.trades[-1]
        if tr["net"] < 0:
            self.consec_losses += 1
            self.consec_wins = 0
            if self.p["max_consec_losses"] > 0 and self.consec_losses >= self.p["max_consec_losses"]:
                self.cooldown = self.p["cooldown_bars"]
                self.consec_losses = 0
        else:
            self.consec_wins += 1
            self.consec_losses = 0
        self.last_exit_sig_bar = sb
        self.pos = None

    # --- main loop -------------------------------------------------------
    def run(self):
        x, b, p = self.x, self.b, self.p
        tape = b.tape
        n = len(tape)
        o, h, l, c = tape.o, tape.h, tape.l, tape.c
        spr = tape.spread_pts
        pt = b.point
        self.partial_pts = 0.0
        sig_id, sig_start, reg_id = x.sig_id, x.sig_start, x.reg_id
        n_closed_before = 0

        for i in range(n):
            sp = spr[i] * pt
            b.cur_spread = spr[i]
            bid_o, ask_o = o[i], o[i] + sp
            self.cur_adx = x.adx[reg_id[i] - 1] if reg_id[i] >= 1 else np.nan
            self.cur_bias_ema = x.bias_ema[reg_id[i] - 1] if reg_id[i] >= 1 else np.nan
            self.cur_bias_close = x.reg_close[reg_id[i] - 1] if reg_id[i] >= 1 else np.nan

            sb = sig_id[i] - 1                       # last closed signal bar
            equity = b.equity(bid_o, ask_o)
            self.update_guards(i, equity, bid_o, ask_o)

            b.process_bar(i, o[i], h[i], l[i], c[i], spr[i])

            if len(b.trades) > n_closed_before:
                n_closed_before = len(b.trades)
                if self.pos is not None and self.pos not in b.positions:
                    self.settle(sb)

            bid_c, ask_c = c[i], c[i] + sp
            if self.pos is not None:
                self.manage(i, sb, bid_c, ask_c, sig_start[i])
                if len(b.trades) > n_closed_before:
                    n_closed_before = len(b.trades)
                    if self.pos not in b.positions:
                        self.settle(sb)

            b.mark(i, bid_c, ask_c)

            if not sig_start[i]:
                continue
            if self.cooldown > 0:
                self.cooldown -= 1
                continue
            if self.pos is not None or b.positions:
                continue
            if not self.entries_enabled(sb):
                continue
            if not self.session_ok(i):
                continue

            d, model = self.signal(sb)
            if d != 0:
                self.signals += 1
                self.try_enter(i, sb, d, model, bid_o, ask_o, spr[i],
                               b.equity(bid_o, ask_o))
        return b


# ----------------------------------------------------------------------------
# Dark Venus: Bollinger counter-trend grid, default configuration
# ----------------------------------------------------------------------------
DV = dict(lots=0.01, bb_period=20, bb_dev=2.0, take_target=50, min_distance=50,
          grid_coeff=1.0, max_orders=50, max_spread=500, both_sides=True)


class DarkVenus:
    """Bollinger counter-trend grid with lot-sum sizing and NO stop loss.

    This is the default configuration described in the reconstruction: stop
    target disabled, monetary and percentage stops off. The only thing that
    ends a losing basket is the target coming back to it, or a margin call.
    """

    def __init__(self, broker, ctx, cfg=None):
        self.b = broker
        self.x = ctx
        self.p = dict(DV, **(cfg or {}))
        self.last_grid_bar = {1: -1, -1: -1}
        self.last_entry_bar = -1

    def basket(self, d):
        ps = [q for q in self.b.positions if q.dir == d]
        if not ps:
            return None
        vol = sum(q.lots for q in ps)
        wavg = sum(q.open_price * q.lots for q in ps) / vol
        worst = min(q.open_price for q in ps) if d > 0 else max(q.open_price for q in ps)
        return dict(n=len(ps), vol=vol, wavg=wavg, worst=worst, ps=ps)

    def run(self):
        x, b, p = self.x, self.b, self.p
        tape = b.tape
        o, h, l, c = tape.o, tape.h, tape.l, tape.c
        spr, pt = tape.spread_pts, b.point
        sig_id, sig_start = x.sig_id, x.sig_start

        for i in range(len(tape)):
            sp = spr[i] * pt
            b.cur_spread = spr[i]
            bid, ask = c[i], c[i] + sp
            sb = sig_id[i] - 1

            b.mark(i, bid, ask)
            if not b.positions and b.balance <= 0:
                break

            # basket targets, checked every bar
            comm_pts = b.commission_rt / b.point_value_per_lot(bid) if b.point_value_per_lot(bid) > 0 else 0
            for d in (1, -1):
                bk = self.basket(d)
                if not bk:
                    continue
                ref = bk["wavg"]
                gain = ((bid - ref) if d > 0 else (ref - ask)) / pt
                if gain >= p["take_target"] + comm_pts:
                    for q in list(bk["ps"]):
                        b.close(q, bid, ask, "basket_tp", i)

            if not sig_start[i] or sb < 1:
                continue
            if spr[i] > p["max_spread"]:
                continue

            up, lo = x.bb_up[sb], x.bb_lo[sb]
            px = x.sc[sb]
            if not (np.isfinite(up) and np.isfinite(lo)):
                continue
            sig = 1 if px < lo else (-1 if px > up else 0)

            # grid additions
            for d in (1, -1):
                bk = self.basket(d)
                if not bk or bk["n"] >= p["max_orders"]:
                    continue
                if self.last_grid_bar[d] == sb:
                    continue
                ref = ask if d > 0 else bid
                adverse = ((bk["worst"] - ref) if d > 0 else (ref - bk["worst"])) / pt
                if adverse >= p["min_distance"]:
                    lot = bk["vol"] * p["grid_coeff"]
                    if b.open(d, lot, bid, ask, i, sb) is not None:
                        self.last_grid_bar[d] = sb

            # new basket
            if sig != 0 and self.basket(sig) is None and self.last_entry_bar != sb:
                if p["both_sides"] or self.basket(-sig) is None:
                    if b.open(sig, p["lots"], bid, ask, i, sb) is not None:
                        self.last_entry_bar = sb
        return b
