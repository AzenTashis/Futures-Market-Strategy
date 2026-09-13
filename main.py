"""
main.py
=======
Futures Market Strategy Backtesting & Risk Analysis - entry point.

Run the whole study with:

    python main.py

Everything is driven by the CONFIG dictionary below. Change a number there and
the entire pipeline (data, indicators, signals, backtests, regimes, event
study, charts, tables) follows.

Pipeline
--------
  1. download and clean daily OHLCV data for each futures contract
  2. compute indicators
  3. build signals for every strategy
  4. backtest each strategy on each instrument, net of transaction costs
  5. break performance down by market regime
  6. run the FOMC and volatility-proxy event studies
  7. run the robustness checks (cost sensitivity, split sample)
  8. draw the charts
  9. write CSV + Excel outputs
 10. print a summary to the terminal

This is a research and educational project. Nothing in it is a trading
recommendation, and no result here implies future profitability.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd

from src import (
    backtester,
    data_loader,
    event_study,
    indicators,
    regime_analysis,
    robustness,
    strategies,
    visualization,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# CONFIGURATION - every tunable lives here
# ======================================================================
CONFIG = {
    # ---- instruments -------------------------------------------------
    # Yahoo Finance continuous front-month futures tickers.
    "tickers": {
        "ES=F": "S&P 500 E-mini Futures",
        "GC=F": "Gold Futures",
        "CL=F": "Crude Oil (WTI) Futures",
    },

    # ---- data source -------------------------------------------------
    # "yahoo" = Yahoo Finance via yfinance (needs the package).
    # "stooq" = Stooq CSV endpoint (no package, no API key).
    # Both are free. If the chosen one fails, the other is tried automatically.
    "data_source": "yahoo",
    "fallback_source": True,

    # ---- sample period ----------------------------------------------
    "start_date": "2016-01-01",
    "end_date": None,                 # None = up to today
    "min_rows_required": 300,         # refuse to analyse a stub series
    "download_retries": 3,
    "offline_demo": False,            # True = synthetic data, for testing only
    "use_cache": False,               # True = re-use data/raw/*.csv, no network
    # Deliberately False: if a download fails the run should STOP, not quietly
    # continue on made-up data. Synthetic data is opt-in via --offline.
    "allow_synthetic_fallback": False,

    # ---- indicator parameters ---------------------------------------
    "sma_fast": 20,                   # ~1 trading month
    "sma_slow": 50,                   # ~1 trading quarter
    "rsi_period": 14,                 # Wilder's original setting
    "vol_window": 20,                 # rolling volatility lookback
    "momentum_window": 10,
    "rolling_max_window": 252,        # ~1 year peak for the drawdown indicator
    "volume_ma_window": 20,

    # ---- strategy parameters ----------------------------------------
    "rsi_oversold": 30,               # enter long below this
    "rsi_exit_long": 50,              # exit long above this
    "rsi_overbought": 70,             # enter short above this (if enabled)
    "rsi_exit_short": 50,             # exit short below this
    "allow_short_trend": False,       # True -> SMA rule flips short, not flat
    "allow_short_rsi": False,         # True -> RSI rule also trades the short side

    # ---- backtest assumptions ---------------------------------------
    "transaction_cost": 0.0005,       # 0.05% of notional per unit of turnover
    "risk_free_rate": 0.0,            # annual, simple
    "trading_days": 252,

    # ---- regime analysis --------------------------------------------
    "regime_vol_min_periods": 252,    # min history before labelling vol regimes

    # ---- robustness checks -------------------------------------------
    "cost_grid": None,                # None -> robustness.DEFAULT_COST_GRID
    "n_splits": 2,                    # halves of the sample, for consistency

    # ---- event study -------------------------------------------------
    "fomc_instruments": ["ES=F", "GC=F"],   # rate-sensitive contracts
    "event_window": 5,                      # +/- days for the volatility window
    "proxy_event_top_pct": 0.05,            # top 5% of days by volatility

    # ---- paths -------------------------------------------------------
    "raw_data_dir": os.path.join(BASE_DIR, "data", "raw"),
    "processed_data_dir": os.path.join(BASE_DIR, "data", "processed"),
    "reference_data_dir": os.path.join(BASE_DIR, "data", "reference"),
    "plots_dir": os.path.join(BASE_DIR, "outputs", "plots"),
    "results_dir": os.path.join(BASE_DIR, "outputs", "results"),
}


# ======================================================================
# Run log
# ======================================================================
class _Tee:
    """
    Mirror everything printed to the terminal into a log file as well.

    Why bother: the terminal output contains things the CSV files do not -
    how many rows each download produced, what the cleaner dropped, which
    data provider actually served each ticker, and any warnings. Keeping a
    copy means a run can be audited after the fact instead of relying on
    whatever is still in the scrollback.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "w", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


# ======================================================================
# Small presentation helpers
# ======================================================================
def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fmt_pct(x) -> str:
    return "n/a" if pd.isna(x) else f"{x * 100:7.2f}%"


def fmt_num(x) -> str:
    return "n/a" if pd.isna(x) else f"{x:7.2f}"


# ======================================================================
# Pipeline steps
# ======================================================================
def build_indicator_frames(data: dict, config: dict) -> dict:
    """Step 2 - attach indicators and drop the warm-up rows."""
    banner("STEP 2/10  INDICATORS")
    frames = {}
    warmup = max(config["sma_slow"], config["rsi_period"] + 1,
                 config["vol_window"], config["momentum_window"])
    for ticker, df in data.items():
        enriched = indicators.add_indicators(df, config)
        # The first `warmup` rows have incomplete indicators. Dropping them
        # keeps every strategy comparable on the same usable sample.
        enriched = enriched.iloc[warmup:].copy()
        frames[ticker] = enriched
        out = os.path.join(config["processed_data_dir"],
                           f"{ticker.replace('=', '_')}_indicators.csv")
        enriched.to_csv(out)
        print(f"  {ticker}: {len(enriched)} usable rows, "
              f"{enriched.shape[1]} columns -> {os.path.basename(out)}")
    return frames


def run_all_backtests(frames: dict, config: dict) -> dict:
    """Steps 3-4 - signals and backtests for every instrument x strategy."""
    banner("STEP 3/10  SIGNALS  &  STEP 4/10  BACKTESTS")
    results = {}
    for ticker, df in frames.items():
        name = config["tickers"][ticker]
        for strategy_name in strategies.STRATEGIES:
            signal = strategies.build_signal(strategy_name, df, config)
            result = backtester.evaluate(df, signal, config, ticker, name,
                                         strategy_name)
            results[(ticker, strategy_name)] = result
            s = result["summary"]
            print(f"  {ticker:>5} | {strategy_name:<32} | "
                  f"ann.ret {fmt_pct(s['Annualized Return'])} | "
                  f"Sharpe {fmt_num(s['Sharpe Ratio'])} | "
                  f"maxDD {fmt_pct(s['Max Drawdown'])} | "
                  f"{s['Num Trades']:>3} trades")
    return results


def run_regime_analysis(frames: dict, results: dict, config: dict):
    """Step 5 - performance conditioned on market regime."""
    banner("STEP 5/10  MARKET REGIME ANALYSIS")
    regime_tables, count_tables, regime_frames = [], [], {}

    for ticker, df in frames.items():
        regimes = regime_analysis.classify_regimes(df, config)
        regime_frames[ticker] = regimes
        count_tables.append(regime_analysis.regime_day_counts(regimes, ticker))
        for (tick, strategy_name), result in results.items():
            if tick != ticker:
                continue
            table = regime_analysis.regime_performance(
                result["backtest"], regimes, config, ticker, strategy_name)
            regime_tables.append(table)

    regime_table = (pd.concat(regime_tables, ignore_index=True)
                    if regime_tables else pd.DataFrame())
    counts = (pd.concat(count_tables, ignore_index=True)
              if count_tables else pd.DataFrame())

    if not counts.empty:
        print("  Regime coverage (share of sample):")
        for _, row in counts.iterrows():
            print(f"    {row['Instrument']:>5} {row['Regime']:<16} "
                  f"{row['Days']:>5} days  ({row['Share of Sample']*100:5.1f}%)")
    return regime_table, counts, regime_frames


def run_event_studies(frames: dict, config: dict):
    """Step 6 - FOMC study plus the volatility-proxy cross-check."""
    banner("STEP 6/10  EVENT STUDY")
    summaries, details = [], []

    fomc_path = os.path.join(config["reference_data_dir"], "fomc_dates.csv")
    fomc_dates = event_study.load_fomc_dates(fomc_path)
    print(f"  Loaded {len(fomc_dates)} FOMC announcement dates from "
          f"{os.path.basename(fomc_path)}")

    for ticker in config["fomc_instruments"]:
        if ticker not in frames or len(fomc_dates) == 0:
            continue
        summary, detail = event_study.run_event_study(
            frames[ticker], fomc_dates, ticker, "FOMC",
            window=config["event_window"], trading_days=config["trading_days"])
        if summary.get("Num Events", 0):
            summaries.append(summary)
            details.append(detail)
            print(f"    {ticker}: {summary['Num Events']} events | "
                  f"mean event-day return {fmt_pct(summary['Mean Return t0'])} | "
                  f"vol ratio {fmt_num(summary['Vol Ratio (event/baseline)'])}")

    for ticker, df in frames.items():
        proxy_dates = event_study.volatility_proxy_dates(
            df, config["proxy_event_top_pct"])
        summary, detail = event_study.run_event_study(
            df, proxy_dates, ticker, "High-Volatility Proxy",
            window=config["event_window"], trading_days=config["trading_days"])
        if summary.get("Num Events", 0):
            summaries.append(summary)
            details.append(detail)
            print(f"    {ticker}: {summary['Num Events']} proxy days | "
                  f"mean |event-day return| {fmt_pct(summary['Mean Abs Return t0'])}")

    summary_table = pd.DataFrame(summaries) if summaries else pd.DataFrame()
    detail_table = (pd.concat(details, ignore_index=True)
                    if details else pd.DataFrame())
    return summary_table, detail_table


def run_robustness_checks(frames: dict, results: dict, config: dict):
    """Step 7 - cost sensitivity and split-sample consistency."""
    banner("STEP 7/10  ROBUSTNESS CHECKS")

    sweep = robustness.cost_sweep(frames, config, config.get("cost_grid"))
    break_even = robustness.break_even_costs(sweep)
    print("  Break-even transaction cost (the level at which the rule stops "
          "making money):")
    for _, row in break_even.iterrows():
        print(f"    {row['Instrument']:>5} | {row['Strategy']:<32} | "
              f"{row['Break-even']:>10}   ({row['Note']})")
    print(f"  Assumed cost in the main study: "
          f"{config['transaction_cost']*10_000:.1f} bps")

    split_frames = []
    for (ticker, strategy_name), result in results.items():
        split_frames.append(robustness.split_sample(
            result["backtest"], config, ticker, strategy_name,
            n_splits=config.get("n_splits", 2)))
    split_table = (pd.concat(split_frames, ignore_index=True)
                   if split_frames else pd.DataFrame())

    consistency = (robustness.consistency_flag(split_table)
                   if not split_table.empty else pd.DataFrame())
    if not consistency.empty:
        print("\n  Does it hold up across the sample?")
        for _, row in consistency.iterrows():
            print(f"    {row['Instrument']:>5} | {row['Strategy']:<32} | "
                  f"Sharpe {row['Sharpe by period']:<14} | "
                  f"Return {row['Return by period']:<16} | "
                  f"{row['Verdict']}")

    return sweep, break_even, split_table, consistency


def make_charts(frames: dict, results: dict, regime_table: pd.DataFrame,
                event_summary: pd.DataFrame, sweep: pd.DataFrame,
                split_table: pd.DataFrame, config: dict) -> list[str]:
    """Step 8 - every chart in the deck."""
    banner("STEP 8/10  CHARTS")
    outdir = config["plots_dir"]
    data_loader.ensure_dirs(outdir)
    paths = []

    for ticker, df in frames.items():
        name = config["tickers"][ticker]
        paths.append(visualization.plot_price_with_smas(df, ticker, name, config, outdir))
        paths.append(visualization.plot_rsi(df, ticker, name, config, outdir))

    for (ticker, strategy_name), result in results.items():
        name = config["tickers"][ticker]
        paths.append(visualization.plot_equity_curves(
            result["backtest"], ticker, name, strategy_name, outdir))
        paths.append(visualization.plot_drawdown(
            result["backtest"], ticker, name, strategy_name, outdir))

    paths.append(visualization.plot_rolling_volatility(frames, config, outdir))

    summary_table = pd.DataFrame([r["summary"] for r in results.values()])
    paths.append(visualization.plot_return_comparison(summary_table, outdir))
    paths.append(visualization.plot_sharpe_comparison(summary_table, outdir))

    if not regime_table.empty:
        paths.append(visualization.plot_regime_performance(regime_table, outdir))

    if not event_summary.empty:
        for label in event_summary["Event Type"].unique():
            paths.append(visualization.plot_event_returns(event_summary, label, outdir))
            paths.append(visualization.plot_event_volatility(event_summary, label, outdir))

    if not sweep.empty:
        paths.append(visualization.plot_cost_sensitivity(sweep, config, outdir))
    if not split_table.empty:
        paths.append(visualization.plot_split_sample(split_table, outdir))

    paths = [p for p in paths if p]
    print(f"  {len(paths)} charts written to outputs/plots/")
    return paths


def write_outputs(results: dict, regime_table: pd.DataFrame,
                  regime_counts: pd.DataFrame, event_summary: pd.DataFrame,
                  event_detail: pd.DataFrame, sweep: pd.DataFrame,
                  break_even: pd.DataFrame, split_table: pd.DataFrame,
                  consistency: pd.DataFrame, config: dict,
                  synthetic: bool) -> pd.DataFrame:
    """Step 9 - CSV files and the multi-sheet Excel report."""
    banner("STEP 9/10  RESULT TABLES")
    results_dir = config["results_dir"]
    data_loader.ensure_dirs(results_dir)

    summary_table = pd.DataFrame([r["summary"] for r in results.values()])
    summary_table = summary_table.sort_values(["Instrument", "Strategy"])

    trade_frames = []
    for (ticker, strategy_name), result in results.items():
        trades = result["trades"].copy()
        if trades.empty:
            continue
        trades.insert(0, "Strategy", strategy_name)
        trades.insert(0, "Instrument", ticker)
        trade_frames.append(trades)
    trade_table = (pd.concat(trade_frames, ignore_index=True)
                   if trade_frames else pd.DataFrame())

    if synthetic:
        # Stamp every table so synthetic output can never be mistaken for real.
        for table in (summary_table, trade_table, regime_table, event_summary,
                      event_detail, sweep, break_even, split_table, consistency):
            if isinstance(table, pd.DataFrame) and not table.empty:
                table.insert(0, "DATA_SOURCE", "SYNTHETIC - NOT REAL MARKET DATA")

    files = {
        "strategy_summary.csv": summary_table,
        "trade_summary.csv": trade_table,
        "regime_analysis.csv": regime_table,
        "event_study.csv": event_summary,
        "event_study_detail.csv": event_detail,
        "regime_day_counts.csv": regime_counts,
        "cost_sensitivity.csv": sweep,
        "break_even_cost.csv": break_even,
        "split_sample.csv": split_table,
        "split_sample_verdict.csv": consistency,
    }
    for filename, table in files.items():
        path = os.path.join(results_dir, filename)
        table.to_csv(path, index=False)
        print(f"  {filename:<28} {len(table):>5} rows")

    # --- Excel ---------------------------------------------------------
    excel_path = os.path.join(results_dir, "final_report.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            _assumptions_sheet(config, synthetic).to_excel(
                writer, sheet_name="Assumptions", index=False)
            summary_table.to_excel(writer, sheet_name="Strategy Summary", index=False)
            trade_table.to_excel(writer, sheet_name="Trade Summary", index=False)
            regime_table.to_excel(writer, sheet_name="Regime Analysis", index=False)
            event_summary.to_excel(writer, sheet_name="Event Study", index=False)
            break_even.to_excel(writer, sheet_name="Break-even Cost", index=False)
            sweep.to_excel(writer, sheet_name="Cost Sensitivity", index=False)
            split_table.to_excel(writer, sheet_name="Split Sample", index=False)
        print("  final_report.xlsx written with 8 sheets")
    except Exception as exc:
        print(f"  [WARN] Excel export failed ({exc!r}). "
              f"CSV outputs are unaffected. Install openpyxl to enable it.")

    return summary_table


def _assumptions_sheet(config: dict, synthetic: bool) -> pd.DataFrame:
    """A sheet recording exactly how the numbers were produced."""
    rows = [
        ("Data source", "SYNTHETIC (offline demo)" if synthetic
         else f"{config['data_source']} daily OHLCV"),
        ("Instruments", ", ".join(config["tickers"])),
        ("Sample start", config["start_date"]),
        ("Sample end", config["end_date"] or "today"),
        ("Transaction cost", f"{config['transaction_cost']*100:.3f}% per unit of turnover"),
        ("Risk-free rate", f"{config['risk_free_rate']*100:.2f}% annual"),
        ("Trading days per year", config["trading_days"]),
        ("Fast / slow moving average", f"{config['sma_fast']} / {config['sma_slow']}"),
        ("RSI period", config["rsi_period"]),
        ("RSI entry / exit (long)", f"{config['rsi_oversold']} / {config['rsi_exit_long']}"),
        ("Short selling enabled", f"trend={config['allow_short_trend']}, "
                                  f"rsi={config['allow_short_rsi']}"),
        ("Signal timing", "position[t] = signal[t-1] (no look-ahead)"),
        ("Event window", f"+/- {config['event_window']} trading days"),
        ("Robustness checks", f"cost sweep + {config.get('n_splits', 2)}-way split sample"),
        ("Disclaimer", "Research and educational project. Not investment advice. "
                       "Past performance does not indicate future results."),
    ]
    return pd.DataFrame(rows, columns=["Assumption", "Value"])


def print_final_summary(summary_table: pd.DataFrame, regime_table: pd.DataFrame,
                        event_summary: pd.DataFrame, break_even: pd.DataFrame,
                        consistency: pd.DataFrame, synthetic: bool) -> None:
    """Step 10 - the terminal report."""
    banner("STEP 10/10  SUMMARY")
    if synthetic:
        print("\n  *** SYNTHETIC DATA MODE - these numbers are meaningless. ***")
        print("  *** Re-run with a working internet connection for real results. ***\n")

    header = (f"  {'Instrument':<6} {'Strategy':<32} {'Ann.Ret':>9} {'Vol':>8} "
              f"{'Sharpe':>7} {'MaxDD':>9} {'Trades':>7} {'Win%':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in summary_table.iterrows():
        print(f"  {row['Instrument']:<6} {row['Strategy']:<32} "
              f"{fmt_pct(row['Annualized Return'])} "
              f"{fmt_pct(row['Annualized Volatility'])} "
              f"{fmt_num(row['Sharpe Ratio'])} "
              f"{fmt_pct(row['Max Drawdown'])} "
              f"{int(row['Num Trades']):>7} "
              f"{fmt_pct(row['Win Rate'])}")

    print("\n  Benchmark - buy and hold on the same contracts and window:")
    for instrument, group in summary_table.groupby("Instrument"):
        row = group.iloc[0]
        print(f"  {instrument:<6} {'Buy and hold':<32} "
              f"{fmt_pct(row['BH Annualized Return'])} "
              f"{fmt_pct(row['BH Annualized Volatility'])} "
              f"{fmt_num(row['BH Sharpe Ratio'])} "
              f"{fmt_pct(row['BH Max Drawdown'])}")

    if not regime_table.empty:
        print("\n  Sharpe ratio by regime (averaged across instruments):")
        pivot = regime_table.pivot_table(index="Strategy", columns="Regime",
                                         values="Sharpe Ratio", aggfunc="mean")
        order = [c for c in ["Bullish", "Bearish", "High-Volatility",
                             "Low-Volatility"] if c in pivot.columns]
        text = pivot[order].round(2).to_string()
        print("  " + text.replace("\n", "\n  "))

    if not event_summary.empty:
        fomc = event_summary[event_summary["Event Type"] == "FOMC"]
        if not fomc.empty:
            print("\n  FOMC announcement days:")
            for _, row in fomc.iterrows():
                print(f"  {row['Instrument']:<6} n={int(row['Num Events']):<4} "
                      f"mean event-day return {fmt_pct(row['Mean Return t0'])}  "
                      f"p={row['p-value (t0 vs 0)']:.3f}  "
                      f"event/normal volatility ratio "
                      f"{fmt_num(row['Vol Ratio (event/baseline)'])}")

    if not break_even.empty:
        print("\n  Break-even transaction cost vs the "
              f"assumed level:")
        for _, row in break_even.iterrows():
            print(f"  {row['Instrument']:<6} {row['Strategy']:<32} "
                  f"{row['Break-even']:>10}")

    if not consistency.empty:
        print("\n  Consistency across the sample:")
        for _, row in consistency.iterrows():
            print(f"  {row['Instrument']:<6} {row['Strategy']:<32} "
                  f"{row['Verdict']}")

    print("\n  Outputs: outputs/results/*.csv, outputs/results/final_report.xlsx, "
          "outputs/plots/*.png")
    print("  A full copy of this terminal output is saved to outputs/run_log.txt")
    print("\n  Research and educational project only. Not investment advice. "
          "Past performance does not indicate future results.\n")


# ======================================================================
# CLI
# ======================================================================
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Futures market strategy backtesting and risk analysis.")
    parser.add_argument("--source", choices=["yahoo", "stooq"],
                        help="data provider (default: yahoo; stooq needs no package or API key)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="do not try the other provider if the chosen one fails")
    parser.add_argument("--use-cache", action="store_true",
                        help="re-run on data already in data/raw/ without contacting any provider")
    parser.add_argument("--offline", action="store_true",
                        help="use synthetic data (pipeline testing only - results are meaningless)")
    parser.add_argument("--start", type=str, help="sample start date, YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="sample end date, YYYY-MM-DD")
    parser.add_argument("--cost", type=float,
                        help="transaction cost per unit of turnover, e.g. 0.0005 for 0.05%%")
    parser.add_argument("--tickers", type=str,
                        help="comma-separated ticker override, e.g. 'ES=F,GC=F'")
    parser.add_argument("--allow-short", action="store_true",
                        help="let both strategies take short positions")
    return parser.parse_args(argv)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = dict(config)
    if args.source:
        config["data_source"] = args.source
    if args.no_fallback:
        config["fallback_source"] = False
    if args.use_cache:
        config["use_cache"] = True
    if args.offline:
        config["offline_demo"] = True
    if args.start:
        config["start_date"] = args.start
    if args.end:
        config["end_date"] = args.end
    if args.cost is not None:
        config["transaction_cost"] = args.cost
    if args.tickers:
        wanted = [t.strip() for t in args.tickers.split(",") if t.strip()]
        config["tickers"] = {t: config["tickers"].get(t, t) for t in wanted}
        config["fomc_instruments"] = [t for t in config["fomc_instruments"]
                                      if t in config["tickers"]]
    if args.allow_short:
        config["allow_short_trend"] = True
        config["allow_short_rsi"] = True
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    config = apply_overrides(CONFIG, args)

    # Start the run log before anything else is printed.
    data_loader.ensure_dirs(os.path.join(BASE_DIR, "outputs"))
    log_path = os.path.join(BASE_DIR, "outputs", "run_log.txt")
    tee = None
    try:
        tee = _Tee(sys.stdout, log_path)
        sys.stdout = tee
    except Exception as exc:
        print(f"[WARN] Could not open a run log ({exc!r}). Continuing without one.")

    try:
        return _run(args, config, log_path)
    finally:
        if tee is not None:
            sys.stdout = tee.stream
            tee.close()


def _run(args, config: dict, log_path: str) -> int:
    banner("FUTURES MARKET STRATEGY BACKTESTING & RISK ANALYSIS")
    print(f"  Instruments      : {', '.join(config['tickers'])}")
    print(f"  Data source      : {config['data_source']}"
          f"{' (with automatic fallback)' if config.get('fallback_source', True) else ''}")
    print(f"  Sample           : {config['start_date']} -> "
          f"{config['end_date'] or 'today'}")
    print(f"  Transaction cost : {config['transaction_cost']*100:.3f}% per unit of turnover")
    print(f"  Strategies       : {', '.join(strategies.STRATEGIES)}")

    data_loader.ensure_dirs(config["raw_data_dir"], config["processed_data_dir"],
                            config["plots_dir"], config["results_dir"])

    try:
        banner("STEP 1/10  DATA")
        data, synthetic = data_loader.load_market_data(config)

        frames = build_indicator_frames(data, config)
        results = run_all_backtests(frames, config)
        regime_table, regime_counts, _ = run_regime_analysis(frames, results, config)
        event_summary, event_detail = run_event_studies(frames, config)
        sweep, break_even, split_table, consistency = run_robustness_checks(
            frames, results, config)
        make_charts(frames, results, regime_table, event_summary,
                    sweep, split_table, config)
        summary_table = write_outputs(results, regime_table, regime_counts,
                                      event_summary, event_detail, sweep,
                                      break_even, split_table, consistency,
                                      config, synthetic)
        print_final_summary(summary_table, regime_table, event_summary,
                            break_even, consistency, synthetic)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except RuntimeError as exc:
        # Expected, explainable failures (no data, etc.) - show the message,
        # not a wall of traceback.
        print(f"\n[STOPPED] {exc}")
        return 2
    except Exception as exc:
        # Anything unexpected: show the full traceback, it is a real bug.
        print(f"\n[FATAL] Unexpected error: {exc!r}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
