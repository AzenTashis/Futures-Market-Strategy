"""
backtester.py
=============
A deliberately small, transparent vectorised backtester.

What it models
--------------
* A position of +1 / 0 / -1 in the futures contract, sized as "100% of the
  notional I am willing to risk". No leverage multiplier, no position sizing,
  no margin mechanics - those would add assumptions I cannot verify.
* A proportional transaction cost charged on every change in position, meant
  to stand in for commission plus the bid-ask spread. Reversing a long into a
  short changes position by 2, so it is charged twice - which is correct.
* Returns are measured close-to-close.

What it does NOT model (stated openly, and repeated in the README)
------------------------------------------------------------------
  slippage beyond the flat cost, margin calls, contract roll costs,
  overnight financing, exchange fees, or the fact that a real fill is not
  guaranteed at the close.

The one thing it is careful about is timing: `position = signal.shift(1)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import risk_metrics


# ----------------------------------------------------------------------
# Core engine
# ----------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, signal: pd.Series, config: dict) -> pd.DataFrame:
    """
    Turn a signal into a day-by-day record of the strategy.

    Returns a DataFrame indexed by date with:
        Close, Return         market data
        Signal                what the rule decided, using data up to day t
        Position              what was actually held on day t  (= Signal[t-1])
        Turnover              |change in position| on day t
        Cost                  transaction cost charged on day t
        GrossReturn           Position * Return   (before costs)
        StrategyReturn        GrossReturn - Cost  (net)
        StrategyEquity        compounded net equity curve, starts at 1
        BuyHoldEquity         compounded market equity curve, starts at 1
    """
    tc = float(config.get("transaction_cost", 0.0))

    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["Return"] = df["Return"].fillna(0.0)

    out["Signal"] = signal.reindex(df.index).fillna(0.0)

    # ---- THE look-ahead guard -------------------------------------------
    # The signal formed at the close of day t-1 is the position held through
    # day t. Nothing decided on day t can influence day t's return.
    out["Position"] = out["Signal"].shift(1).fillna(0.0)

    # Turnover on day t is the size of the trade done at the start of day t.
    out["Turnover"] = out["Position"].diff().abs().fillna(out["Position"].abs())
    out["Cost"] = out["Turnover"] * tc

    out["GrossReturn"] = out["Position"] * out["Return"]
    out["StrategyReturn"] = out["GrossReturn"] - out["Cost"]

    out["StrategyEquity"] = risk_metrics.equity_curve(out["StrategyReturn"])
    out["BuyHoldEquity"] = risk_metrics.equity_curve(out["Return"])
    return out


# ----------------------------------------------------------------------
# Trade extraction
# ----------------------------------------------------------------------
def extract_trades(bt: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Convert the day-by-day position record into a list of round-trip trades.

    A trade is one unbroken stretch of days holding the same non-zero
    position. Its return compounds the NET daily returns over the holding
    period and then subtracts the cost of getting out (which lands on the
    first day after the position closes).

    A position still open on the final day of the sample is reported with
    Open = True so it is obvious that its result is not final.
    """
    tc = float(config.get("transaction_cost", 0.0))
    pos = bt["Position"].to_numpy()
    net = bt["StrategyReturn"].to_numpy()
    close = bt["Close"].to_numpy()
    dates = bt.index

    trades = []
    i = 0
    n = len(pos)
    while i < n:
        if pos[i] == 0:
            i += 1
            continue

        direction = pos[i]
        start = i
        while i + 1 < n and pos[i + 1] == direction:
            i += 1
        end = i                                   # last day holding the position

        # Compound the net daily returns actually earned while in the trade.
        growth = float(np.prod(1.0 + net[start:end + 1]))

        # Exit cost: charged on the day the position is closed (end + 1). If
        # the trade is still open at the end of the sample, charge the cost of
        # closing it anyway so the number is not flattering.
        if end + 1 < n:
            exit_turnover = abs(pos[end + 1] - direction)
            still_open = False
            exit_date = dates[end + 1]
        else:
            exit_turnover = abs(direction)
            still_open = True
            exit_date = dates[end]
        growth *= (1.0 - exit_turnover * tc)

        trades.append({
            "EntryDate": dates[start],
            "ExitDate": exit_date,
            "Direction": "Long" if direction > 0 else "Short",
            "EntryPrice": float(close[start]),
            "ExitPrice": float(close[end]),
            "HoldingDays": int(end - start + 1),
            "TradeReturn": growth - 1.0,
            "Open": still_open,
        })
        i += 1

    return pd.DataFrame(trades)


# ----------------------------------------------------------------------
# One instrument x one strategy
# ----------------------------------------------------------------------
def evaluate(df: pd.DataFrame, signal: pd.Series, config: dict,
             instrument: str, instrument_name: str,
             strategy_name: str) -> dict:
    """
    Run the backtest and package everything a caller might need:
    the daily record, the trade list, and a one-row summary that also
    carries the matching buy-and-hold numbers for comparison.
    """
    bt = run_backtest(df, signal, config)
    trades = extract_trades(bt, config)

    rf = config.get("risk_free_rate", 0.0)
    td = config.get("trading_days", 252)

    strat = risk_metrics.summarize(bt["StrategyReturn"], trades, rf, td)
    bh = risk_metrics.summarize(bt["Return"], None, rf, td)

    summary = {
        "Instrument": instrument,
        "Instrument Name": instrument_name,
        "Strategy": strategy_name,
        "Start": bt.index.min().date(),
        "End": bt.index.max().date(),
        "Trading Days": strat["Trading Days"],
        "Total Return": strat["Total Return"],
        "Annualized Return": strat["Annualized Return"],
        "Annualized Volatility": strat["Annualized Volatility"],
        "Sharpe Ratio": strat["Sharpe Ratio"],
        "Sortino Ratio": strat["Sortino Ratio"],
        "Downside Deviation": strat["Downside Deviation"],
        "Max Drawdown": strat["Max Drawdown"],
        "Calmar Ratio": strat["Calmar Ratio"],
        "Num Trades": strat["Num Trades"],
        "Win Rate": strat["Win Rate"],
        "Avg Trade Return": strat["Avg Trade Return"],
        "Best Trade": strat["Best Trade"],
        "Worst Trade": strat["Worst Trade"],
        "Profit Factor": strat["Profit Factor"],
        "Avg Holding Days": strat["Avg Holding Days"],
        "Time In Market": float((bt["Position"] != 0).mean()),
        "Total Cost Paid": float(bt["Cost"].sum()),
        "BH Total Return": bh["Total Return"],
        "BH Annualized Return": bh["Annualized Return"],
        "BH Annualized Volatility": bh["Annualized Volatility"],
        "BH Sharpe Ratio": bh["Sharpe Ratio"],
        "BH Max Drawdown": bh["Max Drawdown"],
    }

    return {"backtest": bt, "trades": trades, "summary": summary}
