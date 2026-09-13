"""
visualization.py
================
All charts for the project. Plain matplotlib, saved as PNG into
`outputs/plots/`.

Styling rules followed here (deliberately restrained - these are research
charts, not marketing graphics):
  * One idea per chart, titled with what the chart shows.
  * Recessive grid, no chart junk, no 3-D, no dual y-axes. Two measures on
    different scales get two charts, never two axes on one.
  * A fixed, colour-blind-checked series palette, assigned by identity:
    instrument/series 1 blue, 2 orange, 3 teal - never recycled by rank, so a
    given instrument keeps its colour across every chart in the deck.
  * A legend whenever there is more than one series; none when there is one
    (the title already names it).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")            # no display needed - write files directly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- palette -----------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # blue, orange, teal
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d9d6"
NEUTRAL = "#8c8b86"
POSITIVE = "#1baf7a"
NEGATIVE = "#e34948"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _slug(text: str) -> str:
    """File-system-safe version of a ticker or strategy name."""
    return (text.replace("=", "_").replace(" ", "_").replace("/", "-")
                .replace("(", "").replace(")", "").lower())


def _finish(fig, ax_or_axes, path: str, note: str | None = None) -> str:
    """Apply shared cosmetics, add an optional footnote, save and close."""
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.ravel(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
    if note:
        fig.text(0.01, -0.02, note, fontsize=8, color=MUTED, ha="left", va="top")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# Auto-chooses the number of decimals, so a 2.5% tick is not rounded to 3%.
PCT = matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=None)


def _pct(x, _pos=None) -> str:
    return PCT(x)


def _bar_labels(ax, bars, fmt="{:.2f}") -> None:
    """Direct-label bars so identity never depends on colour alone."""
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        offset = 3 if height >= 0 else -12
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED)


# ----------------------------------------------------------------------
# 1. Price with moving averages
# ----------------------------------------------------------------------
def plot_price_with_smas(df: pd.DataFrame, ticker: str, name: str,
                         config: dict, outdir: str) -> str:
    fast, slow = config["sma_fast"], config["sma_slow"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, df["Close"], color=NEUTRAL, linewidth=1.0, label="Close")
    ax.plot(df.index, df[f"SMA{fast}"], color=SERIES[0], linewidth=1.6,
            label=f"SMA {fast}")
    ax.plot(df.index, df[f"SMA{slow}"], color=SERIES[1], linewidth=1.6,
            label=f"SMA {slow}")
    ax.set_title(f"{name} ({ticker}) - price with {fast}- and {slow}-day moving averages")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    return _finish(fig, ax, os.path.join(outdir, f"01_price_sma_{_slug(ticker)}.png"),
                   "The crossover strategy is long whenever the fast average is above the slow one.")


# ----------------------------------------------------------------------
# 2. RSI
# ----------------------------------------------------------------------
def plot_rsi(df: pd.DataFrame, ticker: str, name: str,
             config: dict, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(df.index, df["RSI"], color=SERIES[0], linewidth=1.1)
    ax.axhline(config["rsi_overbought"], color=NEGATIVE, linewidth=1.0, linestyle="--")
    ax.axhline(config["rsi_oversold"], color=POSITIVE, linewidth=1.0, linestyle="--")
    ax.axhline(50, color=NEUTRAL, linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI")
    ax.set_title(f"{name} ({ticker}) - {config['rsi_period']}-day RSI")
    ax.annotate(f"overbought {config['rsi_overbought']}", xy=(0.995, config["rsi_overbought"]),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color=MUTED)
    ax.annotate(f"oversold {config['rsi_oversold']}", xy=(0.995, config["rsi_oversold"]),
                xycoords=("axes fraction", "data"), ha="right", va="top",
                fontsize=8, color=MUTED)
    return _finish(fig, ax, os.path.join(outdir, f"02_rsi_{_slug(ticker)}.png"),
                   "Mean reversion buys below the lower line and exits back through the middle line at 50.")


# ----------------------------------------------------------------------
# 3. Strategy vs buy-and-hold equity
# ----------------------------------------------------------------------
def plot_equity_curves(bt: pd.DataFrame, ticker: str, name: str,
                       strategy_name: str, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bt.index, bt["BuyHoldEquity"], color=NEUTRAL, linewidth=1.4,
            label="Buy and hold")
    ax.plot(bt.index, bt["StrategyEquity"], color=SERIES[0], linewidth=1.8,
            label=strategy_name)
    ax.axhline(1.0, color=GRID, linewidth=1.0)
    ax.set_ylabel("Growth of 1 unit of capital")
    ax.set_title(f"{name} ({ticker}) - {strategy_name} vs buy and hold")
    ax.legend(loc="upper left")
    final_s = bt["StrategyEquity"].iloc[-1]
    final_b = bt["BuyHoldEquity"].iloc[-1]
    ax.annotate(f"{final_s:.2f}x", xy=(bt.index[-1], final_s), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=MUTED, va="center")
    ax.annotate(f"{final_b:.2f}x", xy=(bt.index[-1], final_b), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=MUTED, va="center")
    fname = f"03_equity_{_slug(ticker)}_{_slug(strategy_name)}.png"
    return _finish(fig, ax, os.path.join(outdir, fname),
                   "Net of transaction costs. Past performance does not imply future results.")


# ----------------------------------------------------------------------
# 4. Drawdown
# ----------------------------------------------------------------------
def plot_drawdown(bt: pd.DataFrame, ticker: str, name: str,
                  strategy_name: str, outdir: str) -> str:
    strat_dd = bt["StrategyEquity"] / bt["StrategyEquity"].cummax() - 1.0
    bh_dd = bt["BuyHoldEquity"] / bt["BuyHoldEquity"].cummax() - 1.0

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(bt.index, bh_dd, 0, color=NEUTRAL, alpha=0.25,
                    linewidth=0, label="Buy and hold")
    ax.plot(bt.index, strat_dd, color=SERIES[0], linewidth=1.4, label=strategy_name)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Drawdown from peak")
    ax.set_title(f"{name} ({ticker}) - drawdown, {strategy_name} vs buy and hold")
    ax.legend(loc="lower left")
    fname = f"04_drawdown_{_slug(ticker)}_{_slug(strategy_name)}.png"
    return _finish(fig, ax, os.path.join(outdir, fname),
                   "Drawdown is the fall from the highest equity reached so far - the pain an investor actually feels.")


# ----------------------------------------------------------------------
# 5. Rolling volatility
# ----------------------------------------------------------------------
def plot_rolling_volatility(data: dict, config: dict, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (ticker, df) in enumerate(data.items()):
        ax.plot(df.index, df["Volatility"], color=SERIES[i % len(SERIES)],
                linewidth=1.2, label=ticker)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Annualised volatility")
    ax.set_title(f"{config['vol_window']}-day rolling annualised volatility")
    ax.legend(loc="upper left", ncols=3)
    return _finish(fig, ax, os.path.join(outdir, "05_rolling_volatility.png"),
                   "Daily standard deviation scaled by sqrt(252). Crude oil is structurally the most volatile of the three.")


# ----------------------------------------------------------------------
# 6. Strategy return comparison across instruments
# ----------------------------------------------------------------------
def plot_return_comparison(summary: pd.DataFrame, outdir: str) -> str:
    return _grouped_metric_bar(
        summary, "Annualized Return",
        "Annualised return by instrument and strategy",
        "Annualised return", os.path.join(outdir, "06_return_comparison.png"),
        percent=True,
        note="Grey bars are buy and hold on the same contract over the same window - the benchmark each rule has to beat.")


# ----------------------------------------------------------------------
# 7. Sharpe comparison
# ----------------------------------------------------------------------
def plot_sharpe_comparison(summary: pd.DataFrame, outdir: str) -> str:
    return _grouped_metric_bar(
        summary, "Sharpe Ratio",
        "Sharpe ratio by instrument and strategy",
        "Sharpe ratio", os.path.join(outdir, "07_sharpe_comparison.png"),
        percent=False, bh_column="BH Sharpe Ratio",
        note="Return per unit of volatility, risk-free rate as configured. Higher is better; negative means the rule lost money.")


def _grouped_metric_bar(summary: pd.DataFrame, column: str, title: str,
                        ylabel: str, path: str, percent: bool,
                        bh_column: str | None = None,
                        note: str | None = None) -> str:
    """Grouped bars: one group per instrument, one bar per strategy + benchmark."""
    bh_column = bh_column or f"BH {column}"
    instruments = list(dict.fromkeys(summary["Instrument"]))
    strategies = list(dict.fromkeys(summary["Strategy"]))

    n_series = len(strategies) + 1
    x = np.arange(len(instruments), dtype=float)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, strategy in enumerate(strategies):
        values = [
            summary.loc[(summary["Instrument"] == inst) &
                        (summary["Strategy"] == strategy), column].mean()
            for inst in instruments
        ]
        bars = ax.bar(x + (i - (n_series - 1) / 2) * width, values, width * 0.9,
                      color=SERIES[i % len(SERIES)], label=strategy)
        _bar_labels(ax, bars, "{:.1%}" if percent else "{:.2f}")

    bh_values = [
        summary.loc[summary["Instrument"] == inst, bh_column].mean()
        for inst in instruments
    ]
    bars = ax.bar(x + (len(strategies) - (n_series - 1) / 2) * width,
                  bh_values, width * 0.9, color=NEUTRAL, label="Buy and hold")
    _bar_labels(ax, bars, "{:.1%}" if percent else "{:.2f}")

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x, instruments)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="best")
    return _finish(fig, ax, path, note)


# ----------------------------------------------------------------------
# 8. Regime-wise performance
# ----------------------------------------------------------------------
def plot_regime_performance(regime_table: pd.DataFrame, outdir: str) -> str:
    """Average Sharpe per regime, one group of bars per strategy."""
    if regime_table.empty:
        return ""
    order = ["Bullish", "Bearish", "High-Volatility", "Low-Volatility"]
    regimes = [r for r in order if r in set(regime_table["Regime"])]
    strategies = list(dict.fromkeys(regime_table["Strategy"]))

    x = np.arange(len(regimes), dtype=float)
    width = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, strategy in enumerate(strategies):
        values = [
            regime_table.loc[(regime_table["Strategy"] == strategy) &
                             (regime_table["Regime"] == regime),
                             "Sharpe Ratio"].mean()
            for regime in regimes
        ]
        bars = ax.bar(x + (i - (len(strategies) - 1) / 2) * width, values,
                      width * 0.9, color=SERIES[i % len(SERIES)], label=strategy)
        _bar_labels(ax, bars, "{:.2f}")

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x, regimes)
    ax.set_ylabel("Sharpe ratio (average across instruments)")
    ax.set_title("Strategy performance by market regime")
    ax.legend(loc="best")
    return _finish(fig, ax, os.path.join(outdir, "08_regime_performance.png"),
                   "Regime slices are scattered days, not continuous periods - read these as conditional averages.")


# ----------------------------------------------------------------------
# 9. Event-study returns
# ----------------------------------------------------------------------
def plot_event_returns(event_summary: pd.DataFrame, event_label: str,
                       outdir: str) -> str:
    rows = event_summary[event_summary["Event Type"] == event_label]
    if rows.empty:
        return ""
    instruments = list(rows["Instrument"])
    labels = ["Day -1", "Event day", "Day +1"]
    columns = ["Mean Return t-1", "Mean Return t0", "Mean Return t+1"]

    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(len(instruments), 1)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, instrument in enumerate(instruments):
        row = rows[rows["Instrument"] == instrument].iloc[0]
        values = [row[c] for c in columns]
        bars = ax.bar(x + (i - (len(instruments) - 1) / 2) * width, values,
                      width * 0.9, color=SERIES[i % len(SERIES)], label=instrument)
        _bar_labels(ax, bars, "{:.2%}")

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v*100:.2f}%"))
    ax.set_ylabel("Mean daily return")
    n = int(rows["Num Events"].max())
    ax.set_title(f"Average return around {event_label} events (n = {n})")
    ax.legend(loc="best")
    fname = f"09_event_returns_{_slug(event_label)}.png"
    return _finish(fig, ax, os.path.join(outdir, fname),
                   "An average near zero is a real finding: markets price the expected part of a decision beforehand.")


# ----------------------------------------------------------------------
# 10. Event-window volatility vs baseline
# ----------------------------------------------------------------------
def plot_event_volatility(event_summary: pd.DataFrame, event_label: str,
                          outdir: str) -> str:
    rows = event_summary[event_summary["Event Type"] == event_label]
    if rows.empty:
        return ""
    vol_column = next((c for c in rows.columns if c.startswith("Event Window Vol")), None)
    if vol_column is None:
        return ""

    instruments = list(rows["Instrument"])
    x = np.arange(len(instruments), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, rows[vol_column].to_numpy(), width * 0.92,
                   color=SERIES[0], label="Event window")
    bars2 = ax.bar(x + width / 2, rows["Baseline Vol (other days)"].to_numpy(),
                   width * 0.92, color=NEUTRAL, label="All other days")
    _bar_labels(ax, bars1, "{:.1%}")
    _bar_labels(ax, bars2, "{:.1%}")

    ax.set_xticks(x, instruments)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Annualised volatility")
    ax.set_title(f"Volatility around {event_label} events vs normal days")
    ax.legend(loc="best")
    fname = f"10_event_volatility_{_slug(event_label)}.png"
    return _finish(fig, ax, os.path.join(outdir, fname),
                   "A ratio above 1 means the market moves more than usual in the window around the event.")


# ----------------------------------------------------------------------
# 11. Cost sensitivity
# ----------------------------------------------------------------------
def plot_cost_sensitivity(sweep: pd.DataFrame, config: dict, outdir: str) -> str:
    """
    Sharpe ratio against transaction cost, one panel per instrument.

    Small multiples rather than one crowded axis: six lines on a single chart
    would need six colours, and the point of this chart is the SHAPE of each
    line - how fast it falls - not a comparison of all six at once.
    The dashed vertical line marks the cost the main study assumes.
    """
    if sweep.empty:
        return ""
    instruments = list(dict.fromkeys(sweep["Instrument"]))
    strategies_list = list(dict.fromkeys(sweep["Strategy"]))

    fig, axes = plt.subplots(1, len(instruments),
                             figsize=(4.2 * len(instruments), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, instrument in zip(axes, instruments):
        subset = sweep[sweep["Instrument"] == instrument]
        for i, strategy in enumerate(strategies_list):
            line = subset[subset["Strategy"] == strategy].sort_values("Cost")
            ax.plot(line["Cost (bps)"], line["Sharpe Ratio"],
                    color=SERIES[i % len(SERIES)], linewidth=1.8,
                    marker="o", markersize=4, label=strategy)
        ax.axhline(0, color=INK, linewidth=0.9)
        ax.axvline(config["transaction_cost"] * 10_000, color=NEUTRAL,
                   linewidth=1.0, linestyle="--")
        ax.set_title(instrument, fontsize=11)
        ax.set_xlabel("Transaction cost (bps per unit of turnover)")

    axes[0].set_ylabel("Sharpe ratio")
    axes[0].annotate("assumed cost", xy=(config["transaction_cost"] * 10_000, 0.02),
                     xycoords=("data", "axes fraction"), rotation=90,
                     fontsize=8, color=MUTED, ha="right", va="bottom")
    axes[-1].legend(loc="best")
    fig.suptitle("How quickly does each strategy die as trading gets expensive?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _finish(fig, axes, os.path.join(outdir, "11_cost_sensitivity.png"),
                   "A line crossing zero just right of the dashed line had almost no edge to begin with. The break-even in the table uses annualised return, which turns negative marginally before Sharpe does because of volatility drag.")


# ----------------------------------------------------------------------
# 12. Split-sample consistency
# ----------------------------------------------------------------------
def plot_split_sample(split_table: pd.DataFrame, outdir: str) -> str:
    """Sharpe ratio in each half of the sample, side by side."""
    if split_table.empty:
        return ""
    short = {"Trend Following (SMA Crossover)": "Trend", "Mean Reversion (RSI)": "RSI"}
    combos = (split_table[["Instrument", "Strategy"]]
              .drop_duplicates().itertuples(index=False))
    combos = list(combos)
    periods = list(dict.fromkeys(split_table["Period"]))

    x = np.arange(len(combos), dtype=float)
    width = 0.8 / max(len(periods), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, period in enumerate(periods):
        values = []
        for instrument, strategy in combos:
            match = split_table[(split_table["Instrument"] == instrument)
                                & (split_table["Strategy"] == strategy)
                                & (split_table["Period"] == period)]
            values.append(match["Sharpe Ratio"].mean() if not match.empty else np.nan)
        bars = ax.bar(x + (i - (len(periods) - 1) / 2) * width, values,
                      width * 0.9, color=SERIES[i % len(SERIES)], label=period)
        _bar_labels(ax, bars, "{:.2f}")

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x, [f"{inst}\n{short.get(strat, strat)}" for inst, strat in combos])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Does the result hold up in both halves of the sample?")
    ax.legend(loc="best")
    return _finish(fig, ax, os.path.join(outdir, "12_split_sample.png"),
                   "A bar that is positive in one half and negative in the other means the average was hiding something.")
