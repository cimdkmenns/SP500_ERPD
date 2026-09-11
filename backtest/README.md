# Backtest harness

A broker-accurate replay engine for the two MQL5 scalp EAs in `../mql5`.

**Start with [`ASSESSMENT.md`](ASSESSMENT.md)** — the findings, the numbers, and
what to do next in your own MetaTrader 5 terminal.

Requires numpy only.

```
python3 backtest/final.py                        # the headline evidence
python3 backtest/run.py --tf 30 --strategy v2 \
    --mt5 GBPUSD=/path/GBPUSD_M1.csv             # run against YOUR exported bars
```

This engine is not a substitute for the MT5 Strategy Tester. It is a way to
check the cost arithmetic and the risk mechanics quickly, and to sanity-check a
tester result before trusting it.
