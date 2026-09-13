"""
strategies.py
=============
The two trading rules tested in this project.

Signal convention (used consistently everywhere)
------------------------------------------------
Each strategy returns a `signal` Series where

    signal[t] = the position I WANT to hold, decided using data up to and
                including the close of day t.

The backtester then shifts it forward by one bar:

    position[t] = signal[t-1]

so the return earned on day t is always produced by a position that was
decided before day t started. That single shift is what keeps the backtest
free of look-ahead bias, and it is the first thing I would point to if asked
how I know the results are not cheating.

Position values: +1 long, 0 flat, -1 short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Strategy 1 - trend following / momentum
# ----------------------------------------------------------------------
def sma_crossover_signal(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Moving-average crossover.

    Default (long/flat):
        SMA_fast >  SMA_slow  -> long  (+1)
        SMA_fast <= SMA_slow  -> flat  ( 0)

    Optional (long/short), enabled with config['allow_short_trend']:
        SMA_fast <= SMA_slow  -> short (-1)

    Idea: when the short average is above the long average, recent prices are
    above older prices - the market is in an uptrend. Trend-following bets
    that trends persist a little longer than a coin flip would suggest. It
    makes money in sustained moves and bleeds in choppy, sideways markets,
    which is exactly what the regime analysis is designed to show.

    Why 20 and 50: roughly one trading month against one trading quarter.
    They are conventional, round, and - importantly - were NOT optimised on
    this data, so the results are not the product of curve fitting.
    """
    fast = df[f"SMA{config['sma_fast']}"]
    slow = df[f"SMA{config['sma_slow']}"]

    flat_value = -1 if config.get("allow_short_trend", False) else 0
    signal = pd.Series(np.where(fast > slow, 1, flat_value),
                       index=df.index, dtype=float)

    # Before both averages exist there is no opinion - stay flat.
    signal[fast.isna() | slow.isna()] = 0.0
    return signal


# ----------------------------------------------------------------------
# Strategy 2 - RSI mean reversion
# ----------------------------------------------------------------------
def rsi_mean_reversion_signal(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    RSI-based mean reversion.

    Long rules (default):
        enter long when RSI < oversold  (default 30)
        exit  long when RSI > exit_long (default 50)

    Optional short side, enabled with config['allow_short_rsi']:
        enter short when RSI > overbought (default 70)
        exit  short when RSI < exit_short (default 50)

    Idea: after an unusually one-sided stretch of days, price often snaps back
    toward its recent average. The entry is deliberately a *stretched* reading
    and the exit is the neutral middle (50) rather than the opposite extreme -
    waiting for RSI 70 to close a long would give back most of the bounce.

    The rule is genuinely stateful (you must know whether you are already in a
    position to decide what to do next), so it is written as an explicit loop.
    Slower than a vectorised version, but unambiguous to read and to explain -
    and at ~2,500 rows per instrument the speed does not matter.
    """
    rsi_values = df["RSI"]
    oversold = config["rsi_oversold"]
    exit_long = config["rsi_exit_long"]
    overbought = config["rsi_overbought"]
    exit_short = config["rsi_exit_short"]
    allow_short = config.get("allow_short_rsi", False)

    signal = np.zeros(len(df), dtype=float)
    position = 0.0

    for i, value in enumerate(rsi_values.to_numpy()):
        if np.isnan(value):
            position = 0.0                      # no reading yet -> stay flat
        elif position == 0.0:
            if value < oversold:
                position = 1.0
            elif allow_short and value > overbought:
                position = -1.0
        elif position == 1.0:
            if value > exit_long:
                position = 0.0
        elif position == -1.0:
            if value < exit_short:
                position = 0.0
        signal[i] = position

    return pd.Series(signal, index=df.index, dtype=float)


# ----------------------------------------------------------------------
# Registry - keeps main.py free of if/else chains
# ----------------------------------------------------------------------
STRATEGIES = {
    "Trend Following (SMA Crossover)": sma_crossover_signal,
    "Mean Reversion (RSI)": rsi_mean_reversion_signal,
}


def build_signal(strategy_name: str, df: pd.DataFrame, config: dict) -> pd.Series:
    """Look a strategy up by name and produce its signal."""
    if strategy_name not in STRATEGIES:
        raise KeyError(f"Unknown strategy '{strategy_name}'. "
                       f"Available: {list(STRATEGIES)}")
    return STRATEGIES[strategy_name](df, config)
