"""Diagnostics: where the edge is lost, and what actually moves the needle.

Runs with the equity guards switched OFF and risk held constant, so what is
measured is the strategy's raw expectancy per trade rather than how quickly it
trips its own halt.

  python3 backtest/diag.py
"""

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import engine as eng
import run as R
import tape as tp

# Guards off, risk fixed: measure expectancy, not halt behaviour.
NO_GUARDS = dict(daily_loss_pct=0, weekly_loss_pct=0, max_equity_dd_pct=0,
                 max_trades_day=0, max_consec_losses=0, min_bars_between=0,
                 risk_pct=0.25, max_lots=5.0,
                 losses_to_reduce=0, wins_to_boost=0)

COSTS = {
    # label            spread multiplier, commission per lot round turn
    "retail 1.0pip+$7": (1.00, 7.0),
    "raw 0.3pip+$6":    (0.30, 6.0),
}


def scaled_spread(t, mult):
    out = tp.Tape(t.symbol, t.t, t.o, t.h, t.l, t.c,
                  np.maximum(1, np.round(t.spread_pts * mult)).astype(np.int32),
                  t.spec)
    return out


def job(spec):
    pair, share, seed, sig_tf, cost_label, weeks = spec
    mult, comm = COSTS[cost_label]
    t = scaled_spread(tp.synthetic(pair, weeks=weeks, transient_share=share,
                                   seed=seed), mult)
    cfg = dict(eng.V2, **NO_GUARDS)
    ctx = eng.Context(t, sig_tf=sig_tf, regime_tf=60)
    b = eng.Broker(t, commission_rt=comm)
    s = eng.SnapScalp(b, ctx, cfg, version=2)
    s.run()
    m = R.metrics(b)
    cost = sum(x["comm"] + x.get("spread_cost", 0) + x.get("slip_cost", 0)
               for x in b.trades)
    lots = sum(x["lots"] for x in b.trades)
    n = max(m["trades"], 1)
    pv = b.point_value_per_lot(t.c[-1])
    return dict(pair=pair, share=share, seed=seed, sig_tf=sig_tf, cost=cost_label,
                trades=m["trades"], win=m["win_pct"], pf=m["pf"], net=m["net"],
                exp_per_trade=m["net"] / n,
                cost_pts=cost / max(lots, 1e-9) / pv,
                gross_pts=(m["net"] + cost) / max(lots, 1e-9) / pv,
                ret=m["ret_pct"])


def main():
    pairs = ["EURUSD", "GBPUSD"]
    shares = [0.15, 0.45]
    tfs = [5, 15, 30, 60]
    seeds = [3, 4]
    weeks = 104

    specs = [(p, sh, sd, tf, cl, weeks)
             for p in pairs for sh in shares for sd in seeds
             for tf in tfs for cl in COSTS]

    with Pool(processes=min(4, os.cpu_count() or 2)) as pool:
        rows = pool.map(job, specs)

    print("\n" + "=" * 100)
    print("Expectancy per trade in POINTS, guards off, risk fixed at 0.25%")
    print("gross = edge before costs,  cost = what the round turn takes,")
    print("net   = gross - cost. Net must be positive for the EA to be worth running.")
    print("=" * 100)
    for share in shares:
        print(f"\n--- tape: {share:.0%} of M5 variance mean reverting ---")
        print(f"{'cost model':<18} {'TF':>4} {'pair':<8} {'trades':>7} {'win%':>6} "
              f"{'gross pts':>10} {'cost pts':>9} {'net pts':>8} {'PF':>6} {'ann ret%':>9}")
        for cl in COSTS:
            for tf in tfs:
                for p in pairs:
                    rs = [r for r in rows if r["share"] == share and r["cost"] == cl
                          and r["sig_tf"] == tf and r["pair"] == p]
                    if not rs:
                        continue
                    tr = np.mean([r["trades"] for r in rs])
                    if tr < 5:
                        continue
                    g = np.mean([r["gross_pts"] for r in rs])
                    c = np.mean([r["cost_pts"] for r in rs])
                    pf = np.mean([r["pf"] for r in rs if np.isfinite(r["pf"])] or [0])
                    print(f"{cl:<18} {tf:>4} {p:<8} {tr:>7.0f} "
                          f"{np.mean([r['win'] for r in rs]):>6.1f} {g:>10.1f} "
                          f"{c:>9.1f} {g-c:>8.1f} {pf:>6.2f} "
                          f"{np.mean([r['ret'] for r in rs])/2:>9.1f}")
    return rows


if __name__ == "__main__":
    main()
