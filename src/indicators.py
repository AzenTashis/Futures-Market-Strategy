"""
indicators.py
=============
Technical indicators, written out from their definitions with pandas/numpy
rather than pulled from a TA library. Two reasons:

  1. In an interview I have to be able to explain exactly what each number is.
  2. TA packages differ in their smoothing conventions (especially for RSI),
     so writing the formula makes the methodology explicit.

Every function takes a Series/DataFrame and returns a Series. Nothing here
looks into the future: every value at time t uses only data up to and
including t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Returns
# ----------------------------------------------------------------------
def daily_return(close: pd.Series) -> pd.Series:
    """Simple percentage return: r_t = P_t / P_{t-1} - 1."""
    return close.pct_change()


def log_return(close: pd.Series) -> pd.Series:
    """
    Log return: ln(P_t / P_{t-1}).
    Log returns add up across time, which makes multi-day aggregation and
    volatility scaling cleaner. Simple returns add up across assets.
    """
    return np.log(close / close.shift(1))


# ----------------------------------------------------------------------
# Trend
# ----------------------------------------------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average - the unweighted mean of the last `window` values."""
    return series.rolling(window=window, min_periods=window).mean()


def momentum(close: pd.Series, window: int) -> pd.Series:
    """
    Price momentum over `window` days: P_t / P_{t-window} - 1.
    Positive means the market is higher than it was `window` days ago.
    """
    return close / close.shift(window) - 1.0


# ----------------------------------------------------------------------
# Volatility and drawdown
# ----------------------------------------------------------------------
def rolling_volatility(returns: pd.Series, window: int,
                       trading_days: int = 252) -> pd.Series:
    """
    Annualised rolling standard deviation of daily returns.

    Daily sigma is scaled by sqrt(252) under the usual assumption that daily
    returns are roughly independent, so variance grows linearly with time and
    standard deviation with the square root of time.
    """
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(trading_days)


def rolling_max(close: pd.Series, window: int) -> pd.Series:
    """Highest close over a trailing `window` (the running peak)."""
    return close.rolling(window=window, min_periods=1).max()


def rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    """
    Drawdown against the trailing peak: P_t / max(P over window) - 1.
    Always <= 0. A value of -0.15 means the price is 15% below its recent high.
    """
    return close / rolling_max(close, window) - 1.0


# ----------------------------------------------------------------------
# Oscillator
# ----------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (the original 1978
    definition, and what most charting platforms show).

    Steps:
      1. delta_t = P_t - P_{t-1}
      2. gain_t  = max(delta_t, 0)   loss_t = max(-delta_t, 0)
      3. Smooth gains and losses with Wilder's moving average, which is an
         exponential average with alpha = 1/period.
      4. RS  = avg_gain / avg_loss
         RSI = 100 - 100 / (1 + RS)

    The result sits between 0 and 100. High = recent moves have been mostly
    up; low = mostly down. Conventionally >70 is called "overbought" and
    <30 "oversold", but see INTERVIEW_NOTES.md - those labels are a
    description of recent price action, not a prediction.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing == EWM with alpha = 1/period, adjust=False.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)   # avoid divide-by-zero
    out = 100.0 - (100.0 / (1.0 + rs))
    # A stretch with zero losses is a genuine RSI of 100.
    out = out.where(avg_loss.notna(), np.nan)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return out


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------
def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Attach every indicator used by the project to a cleaned OHLCV frame.

    Returns a new DataFrame - the input is not modified.
    """
    out = df.copy()
    close = out["Close"]

    out["Return"] = daily_return(close)
    out["LogReturn"] = log_return(close)

    # A PriceBreak day is one whose previous observation was removed by the
    # cleaner (the April 2020 negative crude close is the real example).
    #
    # Its return is computed NORMALLY, from the last surviving positive close
    # to this one. That is deliberate, and it is worth being precise about why:
    #
    #   The thing that is undefined is the return TO and FROM a negative price.
    #   The move between the two surrounding POSITIVE closes is perfectly well
    #   defined and perfectly real - on 17 April 2020 WTI closed at $18.27 and
    #   on 21 April at $10.01, and anyone holding across that lost 45%.
    #
    # An earlier version of this code blanked that return instead. That looked
    # cautious and was actually much worse: it deleted a genuine -45% day from
    # the record and inflated crude's buy-and-hold return from roughly 10% a
    # year to 17%. Removing a bad observation is legitimate; removing a real
    # loss is not.
    #
    # The one distortion that remains, and it is small: that return spans more
    # than one trading day, so it slightly overstates single-day volatility on
    # that date. The PriceBreak column is kept in the output so the day is
    # visible and auditable rather than hidden.

    out[f"SMA{config['sma_fast']}"] = sma(close, config["sma_fast"])
    out[f"SMA{config['sma_slow']}"] = sma(close, config["sma_slow"])

    out["Volatility"] = rolling_volatility(out["Return"], config["vol_window"],
                                           config["trading_days"])
    out["RSI"] = rsi(close, config["rsi_period"])
    out["Momentum"] = momentum(close, config["momentum_window"])

    out["RollingMax"] = rolling_max(close, config["rolling_max_window"])
    out["Drawdown"] = rolling_drawdown(close, config["rolling_max_window"])

    if "Volume" in out.columns:
        out["VolumeMA"] = out["Volume"].rolling(
            window=config["volume_ma_window"], min_periods=1).mean()

    return out
