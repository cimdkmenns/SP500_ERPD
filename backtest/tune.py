"""Pick structural defaults for v2 on the evidence from diag.py.

This is NOT parameter optimisation for live trading - fitting numbers to a
generated tape would be meaningless. It compares STRUCTURAL choices whose
mechanism is understood from the cost arithmetic:

  * how many entry models to run (trade count vs cost drag)
  * the take-profit geometry (reward-to-risk vs required win rate)
  * whether the scale-out pays for its second commission

Guards off, risk fixed, so what is compared is expectancy per trade.
"""

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import diag
import engine as eng
import run as R
import tape as tp

VARIANTS = []
for model in ("stretch", "both"):
    for tp_name, tp_mode, tp_atr in (("mean", "mean", None),
                                     ("atr1.1", "atr", 1.1),
                                     ("atr1.6", "atr", 1.6),
                                     ("atr2.2", "atr", 2.2)):
        for part in (0.0, 50.0):
            VARIANTS.append(dict(
                name=f"{model:<7} tp={tp_name:<6} part={int(part)}%",
                entry_model=model, tp_mode=tp_mode,
                tp_atr=(tp_atr or 1.1), partial_pct=part))


def job(spec):
    v, pair, share, seed, cost_label, sig_tf, weeks = spec
    mult, comm = diag.COSTS[cost_label]
    t = diag.scaled_spread(tp.synthetic(pair, weeks=weeks,
                                        transient_share=share, seed=seed), mult)
    cfg = dict(eng.V2, **diag.NO_GUARDS)
    cfg.update({k: v[k] for k in ("entry_model", "tp_mode", "tp_atr", "partial_pct")})
    ctx = eng.Context(t, sig_tf=sig_tf, regime_tf=60)
    b = eng.Broker(t, commission_rt=comm)
    eng.SnapScalp(b, ctx, cfg, version=2).run()
    m = R.metrics(b)
    cost = sum(x["comm"] + x.get("spread_cost", 0) + x.get("slip_cost", 0)
               for x in b.trades)
    lots = max(sum(x["lots"] for x in b.trades), 1e-9)
    pv = b.point_value_per_lot(t.c[-1])
    return dict(name=v["name"], pair=pair, share=share, cost=cost_label,
                trades=m["trades"], win=m["win_pct"], pf=m["pf"],
                net_pts=m["net"] / lots / pv, ret=m["ret_pct"],
                dd=m["max_dd_pct"])


def main():
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    seeds = [3, 4]
    share = 0.45
    weeks = 104
    sig_tf = 30

    specs = [(v, p, share, sd, cl, sig_tf, weeks)
             for v in VARIANTS for p in pairs for sd in seeds
             for cl in diag.COSTS]
    with Pool(processes=min(4, os.cpu_count() or 2)) as pool:
        rows = pool.map(job, specs)

    print("\n" + "=" * 92)
    print(f"Structural variants at M{sig_tf}, tape = {share:.0%} mean reverting, "
          f"guards off, 3 pairs x {len(seeds)} seeds")
    print("=" * 92)
    for cl in diag.COSTS:
        print(f"\n--- cost model: {cl} ---")
        print(f"{'variant':<30} {'trades':>7} {'win%':>6} {'net pts/lot':>12} "
              f"{'PF':>6} {'ann ret%':>9}")
        agg = []
        for v in VARIANTS:
            rs = [r for r in rows if r["name"] == v["name"] and r["cost"] == cl]
            if not rs:
                continue
            agg.append((np.mean([r["net_pts"] for r in rs]), v["name"], rs))
        for net, name, rs in sorted(agg, reverse=True):
            pf = np.mean([r["pf"] for r in rs if np.isfinite(r["pf"])] or [0])
            print(f"{name:<30} {np.mean([r['trades'] for r in rs]):>7.0f} "
                  f"{np.mean([r['win'] for r in rs]):>6.1f} {net:>12.1f} {pf:>6.2f} "
                  f"{np.mean([r['ret'] for r in rs])/2:>9.1f}")
    return rows


if __name__ == "__main__":
    main()
