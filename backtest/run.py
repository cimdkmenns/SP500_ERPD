"""Backtest runner.

Examples
--------
  python3 backtest/run.py --suite            # full assessment suite
  python3 backtest/run.py --quick            # short smoke run
  python3 backtest/run.py --mt5 GBPUSD=/path/GBPUSD_M1.csv --strategy v2

The --mt5 form is the one to use once you have exported real M1 bars from
your own terminal. Everything else runs on generated tapes, which test the
machinery and the cost hurdle but say nothing about whether a real edge exists.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import engine as eng
import tape as tp


def metrics(broker, label=""):
    tr = broker.trades
    n = len(tr)
    nets = np.array([t["net"] for t in tr]) if n else np.array([])
    wins = nets[nets > 0]
    losses = nets[nets <= 0]
    gross_w = wins.sum() if len(wins) else 0.0
    gross_l = -losses.sum() if len(losses) else 0.0
    eq = broker.equity_curve

    return dict(
        label=label,
        trades=n,
        net=float(broker.balance - broker.start_balance),
        ret_pct=float((broker.balance / broker.start_balance - 1.0) * 100.0),
        pf=float(gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0),
        win_pct=float(len(wins) / n * 100.0) if n else 0.0,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        expectancy=float(nets.mean()) if n else 0.0,
        max_dd_pct=float(broker.max_dd_pct),
        worst_floating=float(broker.worst_floating),
        max_lots=float(broker.max_open_lots),
        max_basket=int(broker.max_basket_n),
        min_margin_lvl=(None if broker.min_margin_level == float("inf")
                        else float(broker.min_margin_level)),
        stopouts=int(broker.stopouts),
        final_equity=float(eq[-1]) if len(eq) else float(broker.balance),
        ruined=bool(broker.stopouts > 0 or broker.balance <= 0.35 * broker.start_balance),
    )


def reason_mix(broker):
    mix = {}
    for t in broker.trades:
        mix[t["reason"]] = mix.get(t["reason"], 0) + 1
    return mix


def run_one(symbol, strategy, tape_obj, balance=10_000.0, commission=7.0,
            slippage=2, leverage=100, cfg_over=None, sig_tf=5):
    if strategy == "dv":
        ctx = eng.Context(tape_obj, sig_tf=30, regime_tf=60,
                          cfg=dict(band_period=20, band_dev=2.0))
        b = eng.Broker(tape_obj, balance=balance, leverage=leverage,
                       commission_rt=commission, slippage_pts=slippage)
        eng.DarkVenus(b, ctx).run()
        return b, None

    base = eng.V2 if strategy == "v2" else eng.V1
    cfg = dict(base)
    if cfg_over:
        cfg.update(cfg_over)
    ctx = eng.Context(tape_obj, sig_tf=sig_tf, regime_tf=60)
    b = eng.Broker(tape_obj, balance=balance, leverage=leverage,
                   commission_rt=commission, slippage_pts=slippage)
    s = eng.SnapScalp(b, ctx, cfg, version=(2 if strategy == "v2" else 1))
    s.run()
    return b, s


def fmt_row(m, extra=""):
    pf = "inf" if m["pf"] == float("inf") else f"{m['pf']:.2f}"
    return (f"{m['label']:<26} {m['trades']:>6} {m['net']:>10.0f} "
            f"{m['ret_pct']:>8.1f} {pf:>6} {m['win_pct']:>7.1f} "
            f"{m['max_dd_pct']:>8.1f} {extra}")


HEADER = (f"{'run':<26} {'trades':>6} {'net':>10} {'ret%':>8} {'PF':>6} "
          f"{'win%':>7} {'maxDD%':>8}")


def suite(weeks, seeds, out_path, quick=False):
    pairs = ["EURUSD", "GBPUSD", "USDJPY"]
    scenarios = [
        ("null  (random walk)", 0.00, 45.0),
        ("mild  (25% transient)", 0.25, 45.0),
        ("strong(45% transient)", 0.45, 45.0),
    ]
    if quick:
        pairs = ["EURUSD"]
        scenarios = scenarios[:2]

    results = []

    print("\n" + "=" * 100)
    print("PART 1 - SnapScalp v2 vs v1, same tape, same costs")
    print("=" * 100)
    for sname, share, hl in scenarios:
        print(f"\n--- scenario: {sname} ---")
        print(HEADER)
        for pair in pairs:
            for strat in ("v1", "v2"):
                agg = []
                for sd in seeds:
                    t = tp.synthetic(pair, weeks=weeks, transient_share=share,
                                     half_life_min=hl, seed=sd)
                    b, s = run_one(pair, strat, t)
                    m = metrics(b, f"{pair} {strat}")
                    m.update(scenario=sname, seed=sd, strategy=strat, pair=pair,
                             reasons=reason_mix(b),
                             signals=(s.signals if s else 0),
                             rejects=(s.rejects if s else {}),
                             guard_days=(s.guard_days if s else 0))
                    agg.append(m)
                    results.append(m)
                mm = _mean_metrics(agg, f"{pair} {strat}")
                print(fmt_row(mm))

    print("\n" + "=" * 100)
    print("PART 2 - Dark Venus grid on the same tapes (no stop loss, lot-sum grid)")
    print("=" * 100)
    print(HEADER + "   stopouts  maxLots  worstFloat")
    for sname, share, hl in scenarios:
        ruin = 0
        agg = []
        for pair in pairs:
            for sd in seeds:
                t = tp.synthetic(pair, weeks=weeks, transient_share=share,
                                 half_life_min=hl, seed=sd)
                b, _ = run_one(pair, "dv", t)
                m = metrics(b, f"{pair} dv")
                m.update(scenario=sname, seed=sd, strategy="dv", pair=pair,
                         reasons=reason_mix(b))
                agg.append(m)
                results.append(m)
                ruin += 1 if m["ruined"] else 0
        mm = _mean_metrics(agg, f"ALL dv {sname[:6]}")
        extra = (f"{mm['stopouts']:>8.2f} {mm['max_lots']:>8.2f} "
                 f"{mm['worst_floating']:>11.0f}   ruin {ruin}/{len(agg)}")
        print(fmt_row(mm, extra))

    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nRaw results written to {out_path}")
    return results


def _mean_metrics(rows, label):
    keys = ["trades", "net", "ret_pct", "win_pct", "max_dd_pct", "stopouts",
            "max_lots", "worst_floating", "expectancy"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    pfs = [r["pf"] for r in rows if np.isfinite(r["pf"])]
    out["pf"] = float(np.mean(pfs)) if pfs else float("inf")
    out["label"] = label
    out["trades"] = int(round(out["trades"]))
    return out


def breakeven_sweep(weeks, seeds, pairs=("EURUSD", "GBPUSD", "USDJPY")):
    """How much mean reversion does v2 need before it clears costs?"""
    print("\n" + "=" * 100)
    print("PART 3 - How much transient (mean-reverting) variance v2 needs to break even")
    print("=" * 100)
    print(f"{'pair':<8} " + " ".join(f"{s:>8.0%}" for s in
                                     [0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
          + "     <- transient share, cell = mean return %")
    table = {}
    for pair in pairs:
        row = []
        for share in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
            rets = []
            for sd in seeds:
                t = tp.synthetic(pair, weeks=weeks, transient_share=share, seed=sd)
                b, _ = run_one(pair, "v2", t)
                rets.append(metrics(b)["ret_pct"])
            row.append(float(np.mean(rets)))
        table[pair] = row
        print(f"{pair:<8} " + " ".join(f"{v:>8.1f}" for v in row))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--strategy", default="v2", choices=["v1", "v2", "dv"])
    ap.add_argument("--mt5", action="append", default=[],
                    help="SYMBOL=/path/to/M1.csv  (repeatable)")
    ap.add_argument("--balance", type=float, default=10_000.0)
    ap.add_argument("--commission", type=float, default=7.0)
    ap.add_argument("--slippage", type=int, default=2)
    ap.add_argument("--tf", type=int, default=30,
                    help="signal timeframe in minutes for --mt5 runs (default 30)")
    ap.add_argument("--out", default="backtest/results.json")
    a = ap.parse_args()

    if a.mt5:
        print(HEADER)
        allm = []
        for spec in a.mt5:
            sym, _, path = spec.partition("=")
            t = tp.load_mt5_csv(path, sym)
            print(f"  loaded {sym}: {len(t)} M1 bars "
                  f"{np.datetime64(int(t.t[0]),'m')} .. {np.datetime64(int(t.t[-1]),'m')}")
            b, s = run_one(sym, a.strategy, t, balance=a.balance,
                           commission=a.commission, slippage=a.slippage,
                           sig_tf=a.tf)
            m = metrics(b, f"{sym} {a.strategy} M{a.tf}")
            m["reasons"] = reason_mix(b)
            allm.append(m)
            print(fmt_row(m))
            print(f"     exits: {reason_mix(b)}")
        with open(a.out, "w") as fh:
            json.dump(allm, fh, indent=1, default=float)
        print(f"\nWritten to {a.out}")
        return

    seeds = list(range(1, a.seeds + 1))
    weeks = 26 if a.quick else a.weeks
    if a.suite or a.quick:
        suite(weeks, seeds, a.out, quick=a.quick)
    if a.sweep:
        breakeven_sweep(weeks, seeds)


if __name__ == "__main__":
    main()
