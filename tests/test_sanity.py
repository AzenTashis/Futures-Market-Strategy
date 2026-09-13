"""
tests/test_sanity.py
====================
Small self-checks that protect the claims the project makes about itself.
Run with:  python tests/test_sanity.py

These are not unit tests of every function - they are the handful of things that,
if broken, would make every result in the report wrong.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import backtester, data_loader, indicators, strategies   # noqa: E402
from src import risk_metrics, robustness                          # noqa: E402

CONFIG = {
    "sma_fast": 20, "sma_slow": 50, "rsi_period": 14, "vol_window": 20,
    "momentum_window": 10, "rolling_max_window": 252, "volume_ma_window": 20,
    "rsi_oversold": 30, "rsi_exit_long": 50, "rsi_overbought": 70,
    "rsi_exit_short": 50, "transaction_cost": 0.0005, "risk_free_rate": 0.0,
    "trading_days": 252, "allow_short_trend": False, "allow_short_rsi": False,
}

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


WARMUP = 60          # rows dropped so every indicator is fully formed


def build_clean():
    """Deterministic synthetic OHLCV to test the machinery against."""
    raw = data_loader.generate_synthetic("TEST=F", "2015-01-01", "2024-12-31")
    return data_loader.clean_ohlcv(raw, "TEST=F", verbose=False)


def build_frame(clean=None):
    clean = build_clean() if clean is None else clean
    return indicators.add_indicators(clean, CONFIG).iloc[WARMUP:]


class _FakeResponse:
    """Minimal stand-in for the object urlopen returns."""

    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _with_fake_stooq(payload):
    """Run download_stooq against a canned response body."""
    import urllib.request
    original = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: _FakeResponse(payload)
    try:
        return data_loader.download_stooq("ES=F", "2024-01-01", "2024-01-05",
                                          max_retries=1, pause_seconds=0)
    finally:
        urllib.request.urlopen = original


def _stooq_parses_ok():
    payload = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,4780.0,4800.0,4770.0,4795.0,1200000\n"
        "2024-01-03,4795.0,4810.0,4780.0,4785.0,1100000\n"
        "2024-01-04,4785.0,4795.0,4750.0,4760.0,1300000\n"
    )
    df = _with_fake_stooq(payload)
    return (not df.empty
            and len(df) == 3
            and list(df.columns[:5]) == ["Open", "High", "Low", "Close", "Volume"]
            and isinstance(df.index, pd.DatetimeIndex)
            and abs(float(df["Close"].iloc[-1]) - 4760.0) < 1e-9)


def _stooq_rejects_limit_page():
    # Stooq returns this as plain text with a 200 status, not an HTTP error.
    return _with_fake_stooq("Exceeded the daily hits limit").empty


def main():
    clean = build_clean()
    df = build_frame(clean)
    print("\nSanity checks")
    print("-" * 60)

    # 1. RSI must stay inside its definitional bounds.
    rsi_values = df["RSI"].dropna()
    check("RSI stays within [0, 100]",
          bool((rsi_values >= 0).all() and (rsi_values <= 100).all()))

    # 2. THE look-ahead guard: today's position must equal yesterday's signal.
    signal = strategies.sma_crossover_signal(df, CONFIG)
    bt = backtester.run_backtest(df, signal, CONFIG)
    shifted = bt["Signal"].shift(1).fillna(0.0)
    check("position[t] == signal[t-1] (no look-ahead)",
          bool((bt["Position"] - shifted).abs().max() < 1e-12))

    # 3. Changing a FUTURE price must not change a PAST position. This is the
    #    strongest version of the look-ahead test: perturb the last 100 closes
    #    and confirm every earlier position is untouched.
    tampered_clean = clean.copy()
    tampered_clean.iloc[-100:, tampered_clean.columns.get_loc("Close")] *= 1.25
    tampered = build_frame(tampered_clean)       # same warm-up, same index
    sig2 = strategies.sma_crossover_signal(tampered, CONFIG)
    bt2 = backtester.run_backtest(tampered, sig2, CONFIG)
    # Only the last 100 closes were touched; the slowest moving average looks
    # back 50 days, so nothing before (len - 150) may move.
    cut = len(df) - 150
    check("future prices cannot change past positions",
          bool(np.abs(bt["Position"].iloc[:cut].to_numpy()
                      - bt2["Position"].iloc[:cut].to_numpy()).max() < 1e-12))

    # 4. Costs must reduce returns - never improve them.
    free = dict(CONFIG, transaction_cost=0.0)
    bt_free = backtester.run_backtest(df, signal, free)
    check("transaction costs lower net performance",
          bt_free["StrategyEquity"].iloc[-1] >= bt["StrategyEquity"].iloc[-1])

    # 5. A permanently-long signal, net of zero cost, must reproduce buy and
    #    hold exactly - except for day 1, which the strategy sits out because
    #    its position was decided on day 0. That one-day gap IS the shift, so
    #    the benchmark is adjusted for it rather than the shift being removed.
    always_long = pd.Series(1.0, index=df.index)
    bt_long = backtester.run_backtest(df, always_long, free)
    bh_ex_first = bt_long["BuyHoldEquity"].iloc[-1] / (1 + bt_long["Return"].iloc[0])
    diff = bt_long["StrategyEquity"].iloc[-1] / bh_ex_first - 1
    check("always-long, zero-cost == buy and hold (after day 1)",
          abs(diff) < 1e-9, f"(relative difference {diff:.2e})")

    # 6. The April-2020 crude oil case: a negative close must be removed, the
    #    following day flagged, and its return blanked - never silently
    #    turned into a nonsense percentage.
    negative = build_clean()
    pos = len(negative) // 2
    negative.iloc[pos, negative.columns.get_loc("Close")] = -37.0
    cleaned = data_loader.clean_ohlcv(negative, "TEST=F", verbose=False)
    check("negative close is removed from the price series",
          len(cleaned) == len(negative) - 1)
    check("the day after a removed price is flagged PriceBreak",
          bool(cleaned["PriceBreak"].sum() == 1))
    enriched = indicators.add_indicators(cleaned, CONFIG)
    broken_day = enriched.index[enriched["PriceBreak"].astype(bool)][0]
    # The return on the break day must be the REAL move between the two
    # surrounding positive closes - not blanked, and not computed against
    # the removed negative price. An earlier version blanked it, which
    # silently deleted a genuine 45% loss from the crude oil series.
    prev_close = float(negative["Close"].iloc[pos - 1])
    break_close = float(enriched.loc[broken_day, "Close"])
    expected = break_close / prev_close - 1.0
    check("return across a removed price spans the two surrounding closes",
          abs(float(enriched.loc[broken_day, "Return"]) - expected) < 1e-9,
          f"(got {float(enriched.loc[broken_day, 'Return']):.6f}, "
          f"expected {expected:.6f})")

    # 7. A frame with no bad prices must be left completely alone.
    untouched = data_loader.clean_ohlcv(build_clean(), "TEST=F", verbose=False)
    check("clean data is not altered by the price-break logic",
          bool(untouched["PriceBreak"].sum() == 0))

    # 8. Cost sensitivity must slope the right way: more cost, less Sharpe.
    sweep = robustness.cost_sweep({"TEST=F": df}, dict(CONFIG),
                                  cost_grid=[0.0, 0.001, 0.005])
    monotonic = True
    for _, group in sweep.groupby(["Instrument", "Strategy"]):
        values = group.sort_values("Cost")["Sharpe Ratio"].to_numpy()
        if not np.all(np.diff(values) <= 1e-12):
            monotonic = False
    check("Sharpe ratio falls monotonically as transaction cost rises", monotonic)

    # 9. Split-sample slices must together cover the whole sample exactly once.
    split = robustness.split_sample(bt, CONFIG, "TEST=F", "test", n_splits=2)
    check("split-sample periods cover every row exactly once",
          int(split["Days"].sum()) == len(bt))

    # 10. The Stooq parser, tested against a canned response. The live
    #     endpoint cannot be reached from a test, so the parsing branch is
    #     exercised with a fabricated payload in exactly Stooq's CSV format.
    check("Stooq parser handles a normal CSV response",
          _stooq_parses_ok())
    check("Stooq parser rejects a rate-limit page instead of parsing it",
          _stooq_rejects_limit_page())

    # 11. Known-answer check on the risk metrics.
    flat = pd.Series([0.01] * 252)
    check("annualised return of a constant 1%/day series is correct",
          abs(risk_metrics.annualized_return(flat) - (1.01 ** 252 - 1)) < 1e-9)
    check("max drawdown of a monotonically rising series is 0",
          abs(risk_metrics.max_drawdown(flat)) < 1e-12)

    print("-" * 60)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("All checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
