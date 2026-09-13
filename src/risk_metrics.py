"""
risk_metrics.py
===============
Performance and risk statistics computed from a series of daily returns.

Conventions used throughout (stated once, applied everywhere):
  * 252 trading days per year.
  * Risk-free rate is an ANNUAL simple rate, converted to a daily rate with
    (1 + rf)**(1/252) - 1. Default is 0, which makes the Sharpe ratio a plain
    return-per-unit-of-risk number and keeps the comparison across
    instruments clean.
  * Returns are simple (not log) returns, so equity curves are built by
    compounding: equity_t = prod(1 + r).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def to_daily_rf(annual_rf: float, trading_days: int = 252) -> float:
    """Convert an annual risk-free rate into its daily equivalent."""
    if annual_rf == 0:
        return 0.0
    return (1.0 + annual_rf) ** (1.0 / trading_days) - 1.0


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative growth of 1 unit of capital."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def total_return(returns: pd.Series) -> float:
    """Total compounded return over the whole sample."""
    if returns.dropna().empty:
        return np.nan
    return float(equity_curve(returns).iloc[-1] - 1.0)


def annualized_return(returns: pd.Series, trading_days: int = 252) -> float:
    """
    Geometric (CAGR-style) annualised return.
    Geometric rather than arithmetic because compounding is what an investor
    actually experiences: +50% then -50% is not 0%.
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0                       # capital wiped out
    return growth ** (trading_days / len(r)) - 1.0


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """Standard deviation of daily returns scaled by sqrt(252)."""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(returns: pd.Series, annual_rf: float = 0.0,
                 trading_days: int = 252) -> float:
    """
    Sharpe ratio = excess return per unit of total volatility.

        Sharpe = mean(r - rf_daily) / std(r) * sqrt(252)

    Interpretation: how much return the strategy earned for each unit of risk
    it took. It punishes a strategy for being volatile, whether the volatility
    is upside or downside - which is exactly why Sortino is reported too.
    """
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    excess = r - to_daily_rf(annual_rf, trading_days)
    return float(excess.mean() / sd * np.sqrt(trading_days))


def downside_deviation(returns: pd.Series, annual_rf: float = 0.0,
                       trading_days: int = 252) -> float:
    """
    Annualised standard deviation of returns BELOW the target rate.
    Upside moves are set to zero rather than dropped, so the statistic keeps
    the full sample length in its denominator.
    """
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    shortfall = np.minimum(r - to_daily_rf(annual_rf, trading_days), 0.0)
    return float(np.sqrt((shortfall ** 2).mean()) * np.sqrt(trading_days))


def sortino_ratio(returns: pd.Series, annual_rf: float = 0.0,
                  trading_days: int = 252) -> float:
    """
    Sortino = excess return / downside deviation.
    Same idea as Sharpe but it only counts volatility that actually hurts.
    """
    r = returns.dropna()
    dd = downside_deviation(r, annual_rf, trading_days)
    if not dd or np.isnan(dd) or dd == 0:
        return np.nan
    excess = (r - to_daily_rf(annual_rf, trading_days)).mean() * trading_days
    return float(excess / dd)


def max_drawdown(returns: pd.Series) -> float:
    """
    Largest peak-to-trough fall of the equity curve, as a negative fraction.
    This is the "how bad did it get" number - the one that decides whether a
    strategy is survivable in practice, regardless of its average return.
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    eq = equity_curve(r)
    return float((eq / eq.cummax() - 1.0).min())


def calmar_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualised return divided by the size of the worst drawdown."""
    mdd = max_drawdown(returns)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(annualized_return(returns, trading_days) / abs(mdd))


# ----------------------------------------------------------------------
# Trade-level statistics
# ----------------------------------------------------------------------
def trade_statistics(trades: pd.DataFrame) -> dict:
    """
    Summarise a table of completed trades.

    Win rate is defined as the share of *trades* that finished positive
    (not the share of profitable days).
    Profit factor = gross profit / gross loss; above 1 means the winners
    outweigh the losers in money terms.
    """
    empty = {
        "Num Trades": 0, "Win Rate": np.nan, "Avg Trade Return": np.nan,
        "Best Trade": np.nan, "Worst Trade": np.nan, "Profit Factor": np.nan,
        "Avg Holding Days": np.nan,
    }
    if trades is None or trades.empty:
        return empty

    ret = trades["TradeReturn"].dropna()
    if ret.empty:
        return empty

    wins = ret[ret > 0]
    losses = ret[ret < 0]
    gross_loss = float(-losses.sum())

    return {
        "Num Trades": int(len(ret)),
        "Win Rate": float(len(wins) / len(ret)),
        "Avg Trade Return": float(ret.mean()),
        "Best Trade": float(ret.max()),
        "Worst Trade": float(ret.min()),
        "Profit Factor": float(wins.sum() / gross_loss) if gross_loss > 0 else np.nan,
        "Avg Holding Days": float(trades["HoldingDays"].mean()),
    }


# ----------------------------------------------------------------------
# Full summary
# ----------------------------------------------------------------------
def summarize(returns: pd.Series, trades: pd.DataFrame | None = None,
              annual_rf: float = 0.0, trading_days: int = 252) -> dict:
    """Every headline statistic for one return stream, in one dictionary."""
    r = returns.dropna()
    stats = {
        "Total Return": total_return(r),
        "Annualized Return": annualized_return(r, trading_days),
        "Annualized Volatility": annualized_volatility(r, trading_days),
        "Sharpe Ratio": sharpe_ratio(r, annual_rf, trading_days),
        "Sortino Ratio": sortino_ratio(r, annual_rf, trading_days),
        "Downside Deviation": downside_deviation(r, annual_rf, trading_days),
        "Max Drawdown": max_drawdown(r),
        "Calmar Ratio": calmar_ratio(r, trading_days),
        "Daily Win Rate": float((r > 0).mean()) if len(r) else np.nan,
        "Trading Days": int(len(r)),
    }
    stats.update(trade_statistics(trades))
    return stats
