"""
regime_analysis.py
==================
Splits history into simple market regimes and measures how each strategy
behaved inside each one.

Why this section exists
-----------------------
A single headline Sharpe ratio hides the most important fact about any rule:
it does not work equally well all the time. Trend following earns its money in
sustained moves and gives it back in choppy ranges; mean reversion is the
mirror image. Showing that explicitly is more honest - and more interesting in
an interview - than quoting one average number.

Two independent classifications are used, both from indicators already in the
data:

  Trend regime   Bullish  if SMA_fast >  SMA_slow
                 Bearish  if SMA_fast <= SMA_slow

  Vol regime     High-Volatility if 20-day annualised vol is above its own
                 running median, Low-Volatility otherwise.

Look-ahead note: the volatility threshold is an EXPANDING median (only past
data), not the full-sample median. A full-sample median would label days using
information that did not exist at the time. It is only a descriptive label
here - it never feeds a trading signal - but keeping it causal means the table
can be read as "what an observer would have known at the time".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import risk_metrics


def classify_regimes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach TrendRegime and VolRegime columns to an indicator frame."""
    out = pd.DataFrame(index=df.index)

    fast = df[f"SMA{config['sma_fast']}"]
    slow = df[f"SMA{config['sma_slow']}"]

    trend = pd.Series("Undefined", index=df.index, dtype=object)
    valid_trend = fast.notna() & slow.notna()
    trend[valid_trend & (fast > slow)] = "Bullish"
    trend[valid_trend & (fast <= slow)] = "Bearish"
    out["TrendRegime"] = trend

    vol = df["Volatility"]
    min_obs = int(config.get("regime_vol_min_periods", 252))
    # Expanding median = the median of everything known so far (causal).
    threshold = vol.expanding(min_periods=min_obs).median()

    vol_regime = pd.Series("Undefined", index=df.index, dtype=object)
    valid_vol = vol.notna() & threshold.notna()
    vol_regime[valid_vol & (vol > threshold)] = "High-Volatility"
    vol_regime[valid_vol & (vol <= threshold)] = "Low-Volatility"
    out["VolRegime"] = vol_regime
    out["VolThreshold"] = threshold
    out["Volatility"] = vol

    return out


def _regime_row(returns: pd.Series, config: dict) -> dict:
    """
    Statistics for one regime slice.

    Caveat worth saying out loud: a regime slice is a set of days scattered
    through history, not one continuous window. Compounding and annualising
    them is therefore an approximation - it answers "what would this rule earn
    if conditions like these persisted for a year", not "what happened between
    two dates".
    """
    rf = config.get("risk_free_rate", 0.0)
    td = config.get("trading_days", 252)
    r = returns.dropna()
    if len(r) < 2:
        return {"Days": int(len(r)), "Total Return": np.nan,
                "Annualized Return": np.nan, "Annualized Volatility": np.nan,
                "Sharpe Ratio": np.nan, "Win Rate": np.nan,
                "Max Drawdown": np.nan}
    return {
        "Days": int(len(r)),
        "Total Return": risk_metrics.total_return(r),
        "Annualized Return": risk_metrics.annualized_return(r, td),
        "Annualized Volatility": risk_metrics.annualized_volatility(r, td),
        "Sharpe Ratio": risk_metrics.sharpe_ratio(r, rf, td),
        "Win Rate": float((r > 0).mean()),          # share of positive DAYS
        "Max Drawdown": risk_metrics.max_drawdown(r),
    }


def regime_performance(bt: pd.DataFrame, regimes: pd.DataFrame, config: dict,
                       instrument: str, strategy_name: str) -> pd.DataFrame:
    """
    Break one instrument x strategy backtest down by regime.

    Buy-and-hold is included for every regime as the benchmark, so the table
    answers "did the rule beat simply holding the contract in this regime?"
    """
    joined = bt.join(regimes[["TrendRegime", "VolRegime"]], how="left")
    rows = []

    for column in ["TrendRegime", "VolRegime"]:
        for label, group in joined.groupby(column, observed=True):
            if label == "Undefined":
                continue
            stats = _regime_row(group["StrategyReturn"], config)
            bh = _regime_row(group["Return"], config)
            rows.append({
                "Instrument": instrument,
                "Strategy": strategy_name,
                "Regime Type": "Trend" if column == "TrendRegime" else "Volatility",
                "Regime": label,
                **stats,
                "BH Total Return": bh["Total Return"],
                "BH Sharpe Ratio": bh["Sharpe Ratio"],
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        order = ["Bullish", "Bearish", "High-Volatility", "Low-Volatility"]
        result["_order"] = result["Regime"].apply(
            lambda x: order.index(x) if x in order else 99)
        result = result.sort_values("_order").drop(columns="_order")
    return result


def regime_day_counts(regimes: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """How many days each regime covered - useful context for the tables."""
    rows = []
    for column in ["TrendRegime", "VolRegime"]:
        counts = regimes[column].value_counts()
        for label, count in counts.items():
            if label == "Undefined":
                continue
            rows.append({"Instrument": instrument, "Regime": label,
                         "Days": int(count),
                         "Share of Sample": float(count / len(regimes))})
    return pd.DataFrame(rows)
