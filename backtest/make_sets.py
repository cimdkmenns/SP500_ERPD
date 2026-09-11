"""Generate MetaTrader 5 .set files for SnapScalp_MR_v2.

  python3 backtest/make_sets.py --tf M30 --out mql5/presets

Load the resulting file in the Strategy Tester (Inputs tab > Load) or on a
live chart (Inputs tab > Load). One file per pair.
"""

import argparse
import os

TF = {"M5": 5, "M15": 15, "M30": 30, "H1": 16385, "H4": 16388}

# Order matches the input declaration order in SnapScalp_MR_v2.mq5.
BASE = [
    ("InpMagic", 770201),
    ("InpComment", "SnapScalp2"),
    ("InpPreset", 0),                 # PRESET_AUTO
    ("InpSignalTF", 30),
    ("InpRegimeTF", 16385),           # PERIOD_H1
    ("InpEntryModel", 2),             # MODEL_BOTH - needed for trade count at M30
    ("InpEmaPeriod", 20),
    ("InpAtrPeriod", 14),
    ("InpRsiPeriod", 2),
    ("InpRegimeMode", 0),             # REGIME_RANGE_ONLY
    ("InpAdxPeriod", 14),
    ("InpAdxMax", 28.0),
    ("InpBiasMode", 0),               # BIAS_NONE
    ("InpBiasEmaPeriod", 200),
    ("InpStretchATR", 1.30),
    ("InpRsiBuyBelow", 8.0),
    ("InpRsiSellAbove", 92.0),
    ("InpRequireRejection", "true"),
    ("InpBandPeriod", 20),
    ("InpBandDeviation", 2.0),
    ("InpBandRsiBuyBelow", 25.0),
    ("InpBandRsiSellAbove", 75.0),
    ("InpBandMinStretchATR", 0.60),
    ("InpAtrSlowPeriod", 100),
    ("InpMinAtrRatio", 0.50),
    ("InpMaxAtrRatio", 2.50),
    ("InpSLatr", 1.60),
    ("InpTPMode", 1),                 # TP_MEAN_REVERT
    ("InpTPatr", 1.60),
    ("InpPartialPct", 50.0),          # scale-out pays for itself at M30
    ("InpPartialATR", 0.60),
    ("InpBreakEvenAtR", 0.40),
    ("InpBreakEvenOffsetR", 0.10),
    ("InpTrailStartR", 0.80),
    ("InpTrailATR", 1.00),
    ("InpMaxBarsInTrade", 18),
    ("InpTimeStopKeepR", 0.30),
    ("InpCommissionRT", 7.00),
    ("InpMaxSpreadPts", 20),
    ("InpMinTPtoCostRatio", 2.50),
    ("InpSlippagePts", 10),
    ("InpRiskPercent", 0.50),
    ("InpMaxLots", 5.00),
    ("InpLossesToReduce", 2),
    ("InpRiskMultAfterLoss", 0.50),
    ("InpWinsToBoost", 3),
    ("InpRiskMultAfterWin", 1.25),
    ("InpMaxTradesPerDay", 4),
    ("InpMinBarsBetweenTrades", 1),
    ("InpDailyLossPct", 2.00),
    ("InpWeeklyLossPct", 4.00),
    ("InpMaxEquityDDPct", 10.00),
    ("InpFlattenOnGuard", "true"),
    ("InpMaxConsecLosses", 3),
    ("InpCooldownBars", 24),
    ("InpUseSessions", "true"),
    ("InpSess1StartHour", 7),
    ("InpSess1EndHour", 11),
    ("InpSess2StartHour", 13),
    ("InpSess2EndHour", 17),
    ("InpUseSession3", "false"),
    ("InpSess3StartHour", 2),
    ("InpSess3EndHour", 6),
    ("InpAvoidRollover", "true"),
    ("InpRolloverStart", 23),
    ("InpRolloverEnd", 1),
    ("InpTradeMonday", "true"),
    ("InpTradeFriday", "true"),
    ("InpFridayStopHour", 19),
    ("InpFridayFlatten", "true"),
    ("InpFridayCloseHour", 20),
    ("InpVerboseLog", "false"),
    ("InpMinTradesForTester", 40),
]

PER_PAIR = {
    "EURUSD": {"InpMaxSpreadPts": 12, "InpAdxMax": 28.0, "InpStretchATR": 1.30},
    "GBPUSD": {"InpMaxSpreadPts": 20, "InpAdxMax": 25.0, "InpStretchATR": 1.40},
    "USDJPY": {"InpMaxSpreadPts": 15, "InpAdxMax": 28.0, "InpStretchATR": 1.30,
               "InpUseSession3": "true", "InpMaxBarsInTrade": 24},
}

MAGICS = {"EURUSD": 770201, "GBPUSD": 770202, "USDJPY": 770203}


def write_set(pair, tf_name, out_dir):
    vals = dict(BASE)
    vals["InpSignalTF"] = TF[tf_name]
    vals["InpMagic"] = MAGICS[pair]
    vals.update(PER_PAIR[pair])

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"SnapScalp_v2_{pair}_{tf_name}.set")
    with open(path, "w", newline="\r\n") as fh:
        fh.write(f"; SnapScalp_MR_v2 - {pair} {tf_name}\n")
        fh.write("; Starting point for optimisation, NOT a tuned result.\n")
        fh.write("; Set InpCommissionRT to your broker's actual commission first.\n")
        for k, _ in BASE:
            fh.write(f"{k}={vals[k]}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="M30", choices=list(TF))
    ap.add_argument("--out", default="mql5/presets")
    a = ap.parse_args()
    for pair in MAGICS:
        print("wrote", write_set(pair, a.tf, a.out))


if __name__ == "__main__":
    main()
