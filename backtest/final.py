"""Final evidence run.

  A. Null validation of the engine (tight, many seeds).
  B. The recommended v2.1 configuration at M30, guards ON, all three pairs,
     both cost models, weak and strong mean-reversion tapes.
  C. Dark Venus grid ruin study over many seeds.

  python3 backtest/final.py
"""

import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import diag
import engine as eng
import run as R
import tape as tp

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

# The configuration the evidence points to: M30, both entry models for trade
# count, scale-out kept, target at the mean, risk trimmed.
V21 = dict(eng.V2, entry_model="both", tp_mode="mean", partial_pct=50.0,
           risk_pct=0.50, max_trades_day=4, min_tp_cost_ratio=2.5)


def job_null(spec):
    pair, seed, weeks = spec
    t = tp.synthetic(pair, weeks=weeks, transient_share=0.0, seed=seed)
    b, _ = R.run_one(pair, "v2", t)
    m = R.metrics(b)
    cost = sum(x["comm"] + x.get("spread_cost", 0) + x.get("slip_cost", 0)
               for x in b.trades)
    nets = [x["net"] for x in b.trades]
    return dict(pair=pair, trades=m["trades"], net=m["net"], cost=cost,
                sd=float(np.std(nets)) if nets else 0.0,
                lots=sum(x["lots"] for x in b.trades),
                pv=b.point_value_per_lot(t.c[-1]))


def job_final(spec):
    pair, share, seed, cost_label, sig_tf, weeks = spec
    mult, comm = diag.COSTS[cost_label]
    t = diag.scaled_spread(tp.synthetic(pair, weeks=weeks, transient_share=share,
                                        seed=seed), mult)
    ctx = eng.Context(t, sig_tf=sig_tf, regime_tf=60)
    b = eng.Broker(t, commission_rt=comm)
    s = eng.SnapScalp(b, ctx, dict(V21), version=2)
    s.run()
    m = R.metrics(b)
    return dict(pair=pair, share=share, seed=seed, cost=cost_label,
                trades=m["trades"], win=m["win_pct"], pf=m["pf"],
                ret=m["ret_pct"], dd=m["max_dd_pct"],
                guard_days=s.guard_days, reasons=R.reason_mix(b),
                years=weeks / 52.0)


def job_dv(spec):
    pair, share, seed, weeks = spec
    t = tp.synthetic(pair, weeks=weeks, transient_share=share, seed=seed)
    b, _ = R.run_one(pair, "dv", t)
    m = R.metrics(b)
    return dict(pair=pair, share=share, seed=seed, ret=m["ret_pct"],
                dd=m["max_dd_pct"], max_lots=m["max_lots"],
                max_basket=m["max_basket"], worst_float=m["worst_floating"],
                min_margin=m["min_margin_lvl"], stopouts=m["stopouts"],
                ruined=m["ruined"], trades=m["trades"], win=m["win_pct"])


def main():
    out = {}
    with Pool(processes=min(4, os.cpu_count() or 2)) as pool:
        # ---------------- A ----------------
        print("=" * 92)
        print("A. ENGINE VALIDATION - null tape (pure random walk), v2 as shipped, M5")
        print("=" * 92)
        rows = pool.map(job_null, [(p, sd, 52) for p in PAIRS
                                   for sd in range(1, 13)])
        print(f"{'pair':<8} {'trades':>7} {'net $':>9} {'costs $':>9} {'gross $':>9} "
              f"{'gross/trade':>12} {'+-2 SE':>8}  verdict")
        out["validation"] = {}
        for p in PAIRS:
            rs = [r for r in rows if r["pair"] == p]
            n = sum(r["trades"] for r in rs)
            net = sum(r["net"] for r in rs)
            cost = sum(r["cost"] for r in rs)
            gross = net + cost
            sd = np.sqrt(sum(r["sd"] ** 2 * r["trades"] for r in rs) / max(n, 1))
            se = sd / np.sqrt(max(n, 1))
            ok = "clean" if abs(gross / max(n, 1)) <= 2 * se else "CHECK"
            print(f"{p:<8} {n:>7} {net:>9.0f} {cost:>9.0f} {gross:>9.0f} "
                  f"{gross/max(n,1):>12.2f} {2*se:>8.2f}  {ok}")
            out["validation"][p] = dict(trades=n, net=net, cost=cost, gross=gross,
                                        per_trade=gross / max(n, 1), two_se=2 * se)
        print("\nGross edge within +-2 SE of zero on a driftless walk means the replay")
        print("has no lookahead and no structural leak. It loses exactly its costs.\n")

        print("Cost hurdle implied by those runs:")
        print(f"{'pair':<8} {'avg lots':>9} {'cost/trade $':>13} {'cost/round turn':>17}")
        out["cost"] = {}
        for p in PAIRS:
            rs = [r for r in rows if r["pair"] == p]
            n = sum(r["trades"] for r in rs)
            lots = sum(r["lots"] for r in rs)
            cost = sum(r["cost"] for r in rs)
            pv = rs[0]["pv"]
            pts = cost / max(lots, 1e-9) / pv
            print(f"{p:<8} {lots/max(n,1):>9.2f} {cost/max(n,1):>13.2f} "
                  f"{pts:>13.1f} pts")
            out["cost"][p] = dict(cost_per_trade=cost / max(n, 1), points=pts)

        # ---------------- B ----------------
        print("\n" + "=" * 92)
        print("B. RECOMMENDED CONFIGURATION - M30, both entry models, scale-out, "
              "guards ON, risk 0.5%")
        print("=" * 92)
        specs = [(p, sh, sd, cl, 30, 104)
                 for p in PAIRS for sh in (0.15, 0.45)
                 for sd in range(1, 5) for cl in diag.COSTS]
        rows_f = pool.map(job_final, specs)
        print(f"{'tape':<22} {'cost model':<18} {'pair':<8} {'trades/yr':>10} "
              f"{'win%':>6} {'PF':>6} {'ann ret%':>9} {'maxDD%':>8} {'guard trips':>12}")
        out["final"] = []
        for sh in (0.15, 0.45):
            for cl in diag.COSTS:
                for p in PAIRS:
                    rs = [r for r in rows_f if r["pair"] == p and r["share"] == sh
                          and r["cost"] == cl]
                    yrs = rs[0]["years"]
                    pf = np.mean([r["pf"] for r in rs if np.isfinite(r["pf"])] or [0])
                    rec = dict(share=sh, cost=cl, pair=p,
                               trades_yr=float(np.mean([r["trades"] for r in rs]) / yrs),
                               win=float(np.mean([r["win"] for r in rs])),
                               pf=float(pf),
                               ann_ret=float(np.mean([r["ret"] for r in rs]) / yrs),
                               dd=float(np.mean([r["dd"] for r in rs])),
                               guard=float(np.mean([r["guard_days"] for r in rs])))
                    out["final"].append(rec)
                    print(f"{f'{sh:.0%} mean reverting':<22} {cl:<18} {p:<8} "
                          f"{rec['trades_yr']:>10.0f} {rec['win']:>6.1f} "
                          f"{rec['pf']:>6.2f} {rec['ann_ret']:>9.1f} "
                          f"{rec['dd']:>8.1f} {rec['guard']:>12.1f}")
            print()

        # ---------------- C ----------------
        print("=" * 92)
        print("C. DARK VENUS GRID - 2 years per run, $10k account, 1:100, stop-out 50%")
        print("=" * 92)
        specs = [(p, sh, sd, 104) for p in PAIRS for sh in (0.0, 0.25)
                 for sd in range(1, 13)]
        rows_dv = pool.map(job_dv, specs)
        print(f"{'pair':<8} {'runs':>5} {'win%':>6} {'ret% med':>9} {'ret% p10':>9} "
              f"{'ret% worst':>11} {'maxDD% med':>11} {'maxDD% worst':>13} "
              f"{'peak lots':>10} {'blowups':>8}")
        out["grid"] = {}
        for p in PAIRS:
            rs = [r for r in rows_dv if r["pair"] == p]
            rets = [r["ret"] for r in rs]
            dds = [r["dd"] for r in rs]
            blow = sum(1 for r in rs if r["ruined"])
            rec = dict(runs=len(rs), win=float(np.mean([r["win"] for r in rs])),
                       med=float(np.median(rets)), p10=float(np.percentile(rets, 10)),
                       worst=float(min(rets)), dd_med=float(np.median(dds)),
                       dd_worst=float(max(dds)),
                       peak_lots=float(max(r["max_lots"] for r in rs)),
                       blowups=blow)
            out["grid"][p] = rec
            print(f"{p:<8} {rec['runs']:>5} {rec['win']:>6.1f} {rec['med']:>9.1f} "
                  f"{rec['p10']:>9.1f} {rec['worst']:>11.1f} {rec['dd_med']:>11.1f} "
                  f"{rec['dd_worst']:>13.1f} {rec['peak_lots']:>10.2f} {blow:>8}")
        allr = [r["ret"] for r in rows_dv]
        nblow = sum(1 for r in rows_dv if r["ruined"])
        out["grid"]["ALL"] = dict(runs=len(rows_dv), blowups=nblow,
                                  med=float(np.median(allr)),
                                  worst=float(min(allr)), best=float(max(allr)))
        print(f"\nAll {len(rows_dv)} grid runs: median {np.median(allr):+.1f}%, "
              f"best {max(allr):+.1f}%, worst {min(allr):+.1f}%, "
              f"account destroyed in {nblow} of {len(rows_dv)} "
              f"({nblow/len(rows_dv)*100:.0f}%) within two years.")

    with open("backtest/final_results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nWritten to backtest/final_results.json")


if __name__ == "__main__":
    main()
