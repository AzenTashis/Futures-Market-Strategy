"""
robustness.py
=============
Two checks that ask the uncomfortable question: *is this result real, or did I
get lucky?*

Neither is complicated. Both reuse the backtester that is already there. They
are in the project because a backtest without them is just one number, and one
number is not evidence.

1. COST SENSITIVITY
   Re-run every strategy across a grid of transaction costs and find the
   BREAK-EVEN COST - the level at which the strategy stops making money.
   This is the single most informative robustness check available here. A rule
   that only works at zero cost never worked at all; a rule that survives four
   times its assumed cost has something behind it.

2. SPLIT SAMPLE
   Cut the history into halves and report each separately. If a strategy earns
   everything in the first half and nothing in the second, the average hides
   that completely. This is the cheapest honest version of out-of-sample
   testing - not a substitute for proper walk-forward analysis, but it catches
   the most common embarrassment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtester, risk_metrics, strategies

# Costs to test, as a fraction of notional per unit of turnover.
# 0 up to 0.50%: from "free" to "expensive retail" - the project's default of
# 0.05% sits near the low end, which is the honest place for it to sit.
DEFAULT_COST_GRID = [0.0, 0.0002, 0.0005, 0.0010, 0.0015,
                     0.0020, 0.0030, 0.0040, 0.0050]


# ----------------------------------------------------------------------
# 1. Cost sensitivity
# ----------------------------------------------------------------------
def cost_sweep(frames: dict, config: dict,
               cost_grid: list[float] | None = None) -> pd.DataFrame:
    """
    Re-backtest every instrument x strategy at each cost level.

    The SIGNAL is computed once per instrument x strategy and reused, because
    the trading rule does not depend on the cost - only the P&L does. That
    keeps the sweep fast and, more importantly, makes it a clean experiment:
    exactly one thing changes between rows.
    """
    grid = cost_grid or DEFAULT_COST_GRID
    rows = []

    for ticker, df in frames.items():
        for strategy_name in strategies.STRATEGIES:
            signal = strategies.build_signal(strategy_name, df, config)
            for cost in grid:
                cfg = dict(config, transaction_cost=cost)
                bt = backtester.run_backtest(df, signal, cfg)
                returns = bt["StrategyReturn"]
                rows.append({
                    "Instrument": ticker,
                    "Strategy": strategy_name,
                    "Cost": cost,
                    "Cost (bps)": cost * 10_000,
                    "Annualized Return": risk_metrics.annualized_return(
                        returns, config["trading_days"]),
                    "Sharpe Ratio": risk_metrics.sharpe_ratio(
                        returns, config["risk_free_rate"], config["trading_days"]),
                    "Max Drawdown": risk_metrics.max_drawdown(returns),
                    "Total Cost Paid": float(bt["Cost"].sum()),
                    "Num Position Changes": int((bt["Turnover"] > 0).sum()),
                })
    return pd.DataFrame(rows)


def break_even_costs(sweep: pd.DataFrame) -> pd.DataFrame:
    """
    For each instrument x strategy, find the cost at which annualised return
    crosses zero, by linear interpolation between the two bracketing grid
    points.

    How to read the result:
      * "0 bps"  - the rule loses money even when trading is free. The idea
                   itself does not work on this market; costs are irrelevant.
      * a small number - the apparent edge is thinner than realistic costs.
      * a number well above your assumed cost - the result has some room in
        it. That is the most you can say; it is not proof of anything.
    """
    rows = []
    for (instrument, strategy), group in sweep.groupby(["Instrument", "Strategy"]):
        group = group.sort_values("Cost")
        costs = group["Cost"].to_numpy()
        rets = group["Annualized Return"].to_numpy()

        if rets[0] <= 0:
            break_even, note = 0.0, "unprofitable even at zero cost"
        elif rets[-1] > 0:
            break_even = float(costs[-1])
            note = f"still profitable at the top of the grid ({costs[-1]*10_000:.0f} bps)"
        else:
            idx = int(np.argmax(rets <= 0))          # first non-positive point
            x0, x1 = costs[idx - 1], costs[idx]
            y0, y1 = rets[idx - 1], rets[idx]
            break_even = float(x0 + (x1 - x0) * (y0 / (y0 - y1)))
            note = "interpolated between grid points"

        # A rule still profitable at the top of the grid has a break-even
        # somewhere ABOVE it - the grid cannot say where. Marking it ">50 bps"
        # rather than "50 bps" keeps that distinction visible.
        if note.startswith("still profitable"):
            display = f">{break_even * 10_000:.0f} bps"
        elif break_even == 0.0:
            display = "0 bps"
        else:
            display = f"{break_even * 10_000:.1f} bps"

        rows.append({
            "Instrument": instrument,
            "Strategy": strategy,
            "Break-even Cost": break_even,
            "Break-even Cost (bps)": break_even * 10_000,
            "Break-even": display,
            "Note": note,
        })
    return pd.DataFrame(rows).sort_values(["Instrument", "Strategy"])


# ----------------------------------------------------------------------
# 2. Split sample
# ----------------------------------------------------------------------
def split_sample(bt: pd.DataFrame, config: dict, instrument: str,
                 strategy_name: str, n_splits: int = 2) -> pd.DataFrame:
    """
    Measure performance separately over contiguous equal-length slices of the
    sample, with buy-and-hold alongside for each slice.

    Note on method: the strategy is NOT re-run per slice. It runs once across
    the whole history and the resulting daily returns are then measured in
    pieces. That is deliberate - re-running would reset the position at each
    boundary and invent trades that never happened.

    Note on what this is not: both halves are still in-sample, because the
    parameters (20/50, RSI 30/50) were chosen before any of it. It is a
    consistency check, not a true out-of-sample test. Walk-forward analysis is
    the proper version and is listed under future improvements.
    """
    rf = config.get("risk_free_rate", 0.0)
    td = config.get("trading_days", 252)

    returns = bt["StrategyReturn"]
    market = bt["Return"]
    chunks = np.array_split(np.arange(len(bt)), n_splits)

    rows = []
    for i, positions in enumerate(chunks, start=1):
        if len(positions) < 2:
            continue
        window = bt.index[positions]
        r = returns.iloc[positions]
        m = market.iloc[positions]
        label = f"Half {i}" if n_splits == 2 else f"Period {i}"
        rows.append({
            "Instrument": instrument,
            "Strategy": strategy_name,
            "Period": label,
            "Start": window.min().date(),
            "End": window.max().date(),
            "Days": int(len(positions)),
            "Annualized Return": risk_metrics.annualized_return(r, td),
            "Annualized Volatility": risk_metrics.annualized_volatility(r, td),
            "Sharpe Ratio": risk_metrics.sharpe_ratio(r, rf, td),
            "Max Drawdown": risk_metrics.max_drawdown(r),
            "Daily Win Rate": float((r.dropna() > 0).mean()),
            "BH Annualized Return": risk_metrics.annualized_return(m, td),
            "BH Sharpe Ratio": risk_metrics.sharpe_ratio(m, rf, td),
        })
    return pd.DataFrame(rows)


def consistency_flag(split_table: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the split-sample table to one line per instrument x strategy saying
    whether the result held up. Blunt on purpose - the point is to make a bad
    result impossible to skim past.
    """
    rows = []
    for (instrument, strategy), group in split_table.groupby(["Instrument", "Strategy"]):
        group = group.sort_values("Period")
        sharpes = group["Sharpe Ratio"].to_numpy()
        returns = group["Annualized Return"].to_numpy()

        if np.isnan(sharpes).any():
            verdict = "insufficient data"
        elif (sharpes > 0).all() and (returns > 0).all():
            verdict = "consistent (positive in every period)"
        elif (sharpes > 0).any() or (returns > 0).any():
            verdict = "INCONSISTENT (positive in some periods, negative in others)"
        else:
            verdict = "negative in every period"

        # Sharpe and compounded return can disagree in sign when volatility is
        # extreme: the average daily return is positive while the COMPOUNDED
        # return is negative, because a 50% loss needs a 100% gain to undo.
        # That is volatility drag, and it is a real warning sign, not a
        # rounding artefact - so it gets called out rather than averaged away.
        drag = [(s_, r_) for s_, r_ in zip(sharpes, returns)
                if not np.isnan(s_) and not np.isnan(r_) and s_ > 0 > r_]
        if drag:
            verdict += " - WARNING: positive Sharpe but negative compounded " \
                       "return (volatility drag)"

        rows.append({
            "Instrument": instrument,
            "Strategy": strategy,
            "Sharpe by period": ", ".join(
                "n/a" if np.isnan(s_) else f"{s_:.2f}" for s_ in sharpes),
            "Return by period": ", ".join(
                "n/a" if np.isnan(r_) else f"{r_*100:.1f}%" for r_ in returns),
            "Verdict": verdict,
        })
    return pd.DataFrame(rows).sort_values(["Instrument", "Strategy"])
