"""Full assessment suite.

Runs every experiment used in the written assessment and dumps the raw
numbers to backtest/assessment.json. Parallel over available cores.

  python3 backtest/assess.py               # full run (~15 min on 4 cores)
  python3 backtest/assess.py --fast        # fewer seeds / shorter tapes
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import run as R
import tape as tp

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]


def _job(spec):
    """One backtest. Returns a flat dict of results."""
    pair, strat, share, hl, seed, weeks = spec
    t = tp.synthetic(pair, weeks=weeks, transient_share=share,
                     half_life_min=hl, seed=seed)
    b, s = R.run_one(pair, strat, t)
    m = R.metrics(b, f"{pair}/{strat}")
    m.update(pair=pair, strategy=strat, share=share, seed=seed, weeks=weeks,
             reasons=R.reason_mix(b))
    m["commission"] = float(sum(x["comm"] for x in b.trades))
    m["spread_cost"] = float(sum(x.get("spread_cost", 0.0) for x in b.trades))
    m["slip_cost"] = float(sum(x.get("slip_cost", 0.0) for x in b.trades))
    m["total_cost"] = m["commission"] + m["spread_cost"] + m["slip_cost"]
    m["gross"] = m["net"] + m["total_cost"]
    m["lots_traded"] = float(sum(x["lots"] for x in b.trades))
    nets = [x["net"] for x in b.trades]
    m["net_sd"] = float(np.std(nets)) if nets else 0.0
    if s is not None:
        m["signals"] = s.signals
        m["rejects"] = s.rejects
        m["guard_days"] = s.guard_days
    return m


def run_jobs(specs, pool):
    return pool.map(_job, specs)


def sect(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def part0(pool, seeds, weeks):
    """Null test: on a driftless random walk, net must equal minus costs."""
    sect("PART 0 - Engine validation. Null tape (pure random walk), SnapScalp v2.")
    specs = [(p, "v2", 0.0, 45.0, sd, weeks) for p in PAIRS for sd in seeds]
    rows = run_jobs(specs, pool)
    print(f"{'pair':<8} {'runs':>5} {'trades':>7} {'net$':>9} {'costs$':>9} "
          f"{'gross$':>9} {'gross/trade':>12} {'+-2SE':>9}")
    out = {}
    for p in PAIRS:
        rs = [r for r in rows if r["pair"] == p]
        n = sum(r["trades"] for r in rs)
        net = sum(r["net"] for r in rs)
        cost = sum(r["total_cost"] for r in rs)
        gross = net + cost
        sd = np.sqrt(sum(r["net_sd"] ** 2 * r["trades"] for r in rs) / max(n, 1))
        se = sd / np.sqrt(max(n, 1))
        print(f"{p:<8} {len(rs):>5} {n:>7} {net:>9.0f} {cost:>9.0f} {gross:>9.0f} "
              f"{gross/max(n,1):>12.2f} {2*se:>9.2f}")
        out[p] = dict(runs=len(rs), trades=n, net=net, cost=cost, gross=gross,
                      gross_per_trade=gross / max(n, 1), two_se=2 * se)
    print("\nA gross edge inside +-2SE of zero means the replay has no lookahead")
    print("and no structural leak: the strategy loses exactly its costs, as it must.")
    return out, rows


def part1(rows):
    """Cost hurdle implied by the null runs."""
    sect("PART 1 - The cost hurdle per pair (from the null runs above)")
    print(f"{'pair':<8} {'trades':>7} {'avg lots':>9} {'cost/trade$':>12} "
          f"{'avg |P/L|$':>11} {'cost as % of':>13} {'breakeven':>10}")
    print(f"{'':<8} {'':>7} {'':>9} {'':>12} {'':>11} {'trade size':>13} {'win rate':>10}")
    out = {}
    for p in PAIRS:
        rs = [r for r in rows if r["pair"] == p]
        n = sum(r["trades"] for r in rs)
        lots = sum(r["lots_traded"] for r in rs)
        cost = sum(r["total_cost"] for r in rs)
        avg_abs = np.mean([r["net_sd"] for r in rs])
        cpt = cost / max(n, 1)
        # Symmetric-payoff breakeven: with average win W and average loss L
        # both equal to avg_abs, the win rate that pays the cost is
        #   w*W - (1-w)*L = cost  ->  w = 0.5 + cost/(2*avg)
        be = 0.5 + cpt / (2 * max(avg_abs, 1e-9))
        print(f"{p:<8} {n:>7} {lots/max(n,1):>9.2f} {cpt:>12.2f} {avg_abs:>11.1f} "
              f"{cpt/max(avg_abs,1e-9)*100:>12.0f}% {be*100:>9.1f}%")
        out[p] = dict(cost_per_trade=cpt, avg_abs=float(avg_abs),
                      breakeven_win=float(be))
    return out


def part2(pool, seeds, weeks):
    sect("PART 2 - SnapScalp v2 vs v1, identical tapes and costs")
    scen = [("null   0%", 0.00), ("mild  25%", 0.25), ("strong 45%", 0.45)]
    specs = [(p, st, sh, 45.0, sd, weeks)
             for _, sh in scen for p in PAIRS for st in ("v1", "v2") for sd in seeds]
    rows = run_jobs(specs, pool)
    print(f"{'scenario':<11} {'pair':<8} {'ver':<4} {'trades':>7} {'ret%':>8} "
          f"{'PF':>6} {'win%':>7} {'maxDD%':>8} {'ret/DD':>8}")
    out = []
    for name, sh in scen:
        for p in PAIRS:
            for st in ("v1", "v2"):
                rs = [r for r in rows if r["pair"] == p and r["strategy"] == st
                      and r["share"] == sh]
                tr = np.mean([r["trades"] for r in rs])
                ret = np.mean([r["ret_pct"] for r in rs])
                pf = np.mean([r["pf"] for r in rs if np.isfinite(r["pf"])] or [0])
                win = np.mean([r["win_pct"] for r in rs])
                dd = np.mean([r["max_dd_pct"] for r in rs])
                rdd = ret / dd if dd > 0.05 else 0.0
                print(f"{name:<11} {p:<8} {st:<4} {tr:>7.0f} {ret:>8.1f} {pf:>6.2f} "
                      f"{win:>7.1f} {dd:>8.1f} {rdd:>8.2f}")
                out.append(dict(scenario=name, pair=p, version=st, trades=tr,
                                ret=ret, pf=pf, win=win, dd=dd, ret_dd=rdd))
        print()
    return out, rows


def part3(pool, seeds, weeks):
    sect("PART 3 - How much mean reversion v2 needs before it clears costs")
    shares = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    specs = [(p, "v2", sh, 45.0, sd, weeks)
             for p in PAIRS for sh in shares for sd in seeds]
    rows = run_jobs(specs, pool)
    print("annualised return %, by share of M5 variance that is mean reverting\n")
    print(f"{'pair':<8} " + " ".join(f"{s:>9.0%}" for s in shares))
    out = {}
    years = weeks / 52.0
    for p in PAIRS:
        vals = []
        for sh in shares:
            rs = [r for r in rows if r["pair"] == p and r["share"] == sh]
            vals.append(float(np.mean([r["ret_pct"] for r in rs]) / years))
        out[p] = dict(shares=shares, ann_ret=vals)
        print(f"{p:<8} " + " ".join(f"{v:>9.1f}" for v in vals))
    print("\nThe crossing point is the amount of genuine mean reversion the real")
    print("tape must contain for this EA to be worth running at these costs.")
    return out, rows


def part4(pool, seeds, weeks):
    sect("PART 4 - Dark Venus grid: the tail the equity curve hides")
    specs = [(p, "dv", sh, 45.0, sd, weeks)
             for p in PAIRS for sh in (0.0, 0.25) for sd in seeds]
    rows = run_jobs(specs, pool)
    print(f"{'pair':<8} {'runs':>5} {'ret% med':>9} {'ret% worst':>11} "
          f"{'maxDD% med':>11} {'maxDD% worst':>13} {'max lots':>9} "
          f"{'minMargin%':>11} {'blowups':>8}")
    out = {}
    for p in PAIRS:
        rs = [r for r in rows if r["pair"] == p]
        rets = [r["ret_pct"] for r in rs]
        dds = [r["max_dd_pct"] for r in rs]
        ml = [r["min_margin_lvl"] for r in rs if r["min_margin_lvl"] is not None]
        blow = sum(1 for r in rs if r["ruined"])
        print(f"{p:<8} {len(rs):>5} {np.median(rets):>9.1f} {min(rets):>11.1f} "
              f"{np.median(dds):>11.1f} {max(dds):>13.1f} "
              f"{max(r['max_lots'] for r in rs):>9.2f} "
              f"{(min(ml) if ml else float('nan')):>11.0f} {blow:>8}")
        out[p] = dict(runs=len(rs), ret_med=float(np.median(rets)),
                      ret_worst=float(min(rets)), dd_med=float(np.median(dds)),
                      dd_worst=float(max(dds)),
                      max_lots=float(max(r["max_lots"] for r in rs)),
                      min_margin=(float(min(ml)) if ml else None), blowups=blow)
    allr = [r["ret_pct"] for r in rows]
    print(f"\nacross all {len(rows)} grid runs: median {np.median(allr):.1f}%, "
          f"worst {min(allr):.1f}%, best {max(allr):.1f}%")
    print("A grid with no stop posts a high win rate and a smooth curve until the")
    print("one basket that does not come back. Judge it on the worst column.")
    return out, rows


def part5(rows_v2):
    sect("PART 5 - Where v2 trades end, and what the guards do")
    agg = {}
    for r in rows_v2:
        if r["strategy"] != "v2":
            continue
        for k, v in r.get("reasons", {}).items():
            agg[k] = agg.get(k, 0) + v
    tot = sum(agg.values()) or 1
    print("exit reason mix across all v2 runs:")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<10} {v:>7}  {v/tot*100:>5.1f}%")
    gd = [r.get("guard_days", 0) for r in rows_v2 if r["strategy"] == "v2"]
    if gd:
        print(f"\ndaily-loss guard trips per run: mean {np.mean(gd):.1f}, max {max(gd)}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default="backtest/assessment.json")
    a = ap.parse_args()

    if a.fast:
        seeds0, w0 = list(range(1, 5)), 52
        seeds2, w2 = list(range(1, 3)), 52
        seeds3, w3 = list(range(1, 3)), 52
        seeds4, w4 = list(range(1, 5)), 52
    else:
        seeds0, w0 = list(range(1, 13)), 52
        seeds2, w2 = list(range(1, 5)), 104
        seeds3, w3 = list(range(1, 5)), 78
        seeds4, w4 = list(range(1, 11)), 104

    with Pool(processes=min(4, os.cpu_count() or 2)) as pool:
        p0, rows0 = part0(pool, seeds0, w0)
        p1 = part1(rows0)
        p2, rows2 = part2(pool, seeds2, w2)
        p3, _ = part3(pool, seeds3, w3)
        p4, _ = part4(pool, seeds4, w4)
        p5 = part5(rows2)

    with open(a.out, "w") as fh:
        json.dump(dict(validation=p0, costs=p1, head_to_head=p2, sweep=p3,
                       grid=p4, exits=p5), fh, indent=1, default=float)
    print(f"\nRaw numbers written to {a.out}")


if __name__ == "__main__":
    main()
