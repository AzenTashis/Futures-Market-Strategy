"""
event_study.py
==============
Two event studies, both runnable offline with no paid news/sentiment API.

1. FOMC study (the main one)
   Federal Reserve policy announcements are the cleanest scheduled macro event
   available: the date is known in advance, the whole market watches it, and
   S&P 500 futures and Gold futures are two of the most rate-sensitive
   instruments there are. Equities respond to the growth/liquidity signal;
   gold responds mainly through real rates and the dollar.

   Dates come from `data/reference/fomc_dates.csv`, compiled from the Federal
   Reserve's published FOMC calendar. Decision days are the SECOND day of each
   two-day meeting - that is when the statement is released. This is a static
   file by design: it makes the study reproducible and removes a network
   dependency. It must be extended by hand for future years.

2. Volatility-proxy study (the backup / cross-check)
   The top X% of days by 20-day volatility, or the top X% by volume, treated
   as "something happened" days. This is explicitly a MARKET-EVENT PROXY:
   it is defined from price and volume, not from news, and it says nothing
   about sentiment. It is included because it works on any instrument -
   including Crude Oil, where FOMC is not the dominant driver.

Method (same for both)
----------------------
For every event date mapped to a trading day:
  * return on day  t-1  (the run-in)
  * return on day  t     (the event day)
  * return on day  t+1  (the follow-through)
  * realised volatility across the window [t-w, t+w]
and the event-window volatility is compared against the volatility of all
days that are NOT in any event window - the baseline.

Caveats that belong on the slide, not in a footnote
---------------------------------------------------
* Event windows overlap for closely-spaced events; the observations are not
  fully independent, so the t-statistic is indicative only.
* Yahoo's daily futures bar is close-to-close, so an announcement at 2pm ET
  lands inside the event-day bar. That is the right bar to look at, but it
  mixes the announcement with the rest of the day's news.
* A statistically flat average is a real result, not a failure. Markets price
  in the expected part of a decision in advance; only the surprise moves price.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:                                   # graceful degradation
    HAVE_SCIPY = False


# ----------------------------------------------------------------------
# Event date sources
# ----------------------------------------------------------------------
def load_fomc_dates(path: str) -> pd.DatetimeIndex:
    """Read the static FOMC announcement-date file."""
    if not os.path.exists(path):
        print(f"  [WARN] FOMC date file not found at {path}.")
        return pd.DatetimeIndex([])
    try:
        table = pd.read_csv(path)
        dates = pd.to_datetime(table["date"], errors="coerce").dropna()
        return pd.DatetimeIndex(sorted(dates.unique()))
    except Exception as exc:
        print(f"  [WARN] Could not read FOMC dates ({exc!r}).")
        return pd.DatetimeIndex([])


def volatility_proxy_dates(df: pd.DataFrame, top_pct: float = 0.05) -> pd.DatetimeIndex:
    """
    'Market-event proxy': the top `top_pct` of days ranked by 20-day
    annualised volatility. NOT a news event - see the module docstring.
    """
    vol = df["Volatility"].dropna()
    if vol.empty:
        return pd.DatetimeIndex([])
    cutoff = vol.quantile(1.0 - top_pct)
    return pd.DatetimeIndex(vol[vol >= cutoff].index)


def volume_proxy_dates(df: pd.DataFrame, top_pct: float = 0.05) -> pd.DatetimeIndex:
    """'Market-event proxy': the top `top_pct` of days by traded volume."""
    if "Volume" not in df.columns:
        return pd.DatetimeIndex([])
    vol = df["Volume"].dropna()
    if vol.empty:
        return pd.DatetimeIndex([])
    cutoff = vol.quantile(1.0 - top_pct)
    return pd.DatetimeIndex(vol[vol >= cutoff].index)


# ----------------------------------------------------------------------
# Mapping event dates onto trading days
# ----------------------------------------------------------------------
def map_to_trading_days(event_dates: pd.DatetimeIndex,
                        index: pd.DatetimeIndex,
                        max_shift_days: int = 3) -> list[int]:
    """
    Turn calendar event dates into integer positions in the price index.

    If an announcement falls on a non-trading day (the 15 March 2020 Sunday
    emergency cut, for example) the FIRST TRADING DAY ON OR AFTER it is used -
    that is the session in which the market could actually react.
    """
    positions = []
    index = pd.DatetimeIndex(index)
    for date in event_dates:
        if date < index[0] or date > index[-1]:
            continue
        pos = index.searchsorted(date, side="left")
        if pos >= len(index):
            continue
        if (index[pos] - date).days > max_shift_days:
            continue
        positions.append(int(pos))
    return sorted(set(positions))


# ----------------------------------------------------------------------
# The study
# ----------------------------------------------------------------------
def run_event_study(df: pd.DataFrame, event_dates: pd.DatetimeIndex,
                    instrument: str, event_label: str,
                    window: int = 5, trading_days: int = 252) -> tuple[dict, pd.DataFrame]:
    """
    Returns
    -------
    summary : one-row dict of aggregate statistics
    detail  : one row per event with its t-1 / t0 / t+1 returns
    """
    returns = df["Return"]
    index = df.index
    positions = map_to_trading_days(event_dates, index)

    empty = ({"Instrument": instrument, "Event Type": event_label,
              "Num Events": 0}, pd.DataFrame())
    if not positions:
        return empty

    r = returns.to_numpy()
    rows = []
    window_positions: set[int] = set()

    for pos in positions:
        rows.append({
            "Instrument": instrument,
            "Event Type": event_label,
            "EventDate": index[pos],
            "Return t-1": r[pos - 1] if pos - 1 >= 0 else np.nan,
            "Return t0": r[pos],
            "Return t+1": r[pos + 1] if pos + 1 < len(r) else np.nan,
            "Abs Return t0": abs(r[pos]) if not np.isnan(r[pos]) else np.nan,
        })
        lo, hi = max(0, pos - window), min(len(r) - 1, pos + window)
        window_positions.update(range(lo, hi + 1))

    detail = pd.DataFrame(rows)

    # --- event-window vs baseline volatility -----------------------------
    mask = np.zeros(len(r), dtype=bool)
    mask[sorted(window_positions)] = True
    event_returns = pd.Series(r[mask], index=index[mask]).dropna()
    base_returns = pd.Series(r[~mask], index=index[~mask]).dropna()

    event_vol = float(event_returns.std(ddof=1) * np.sqrt(trading_days)) \
        if len(event_returns) > 1 else np.nan
    base_vol = float(base_returns.std(ddof=1) * np.sqrt(trading_days)) \
        if len(base_returns) > 1 else np.nan

    # --- is the event-day move different from zero on average? -----------
    event_day = detail["Return t0"].dropna()
    if HAVE_SCIPY and len(event_day) > 2:
        t_stat, p_value = scipy_stats.ttest_1samp(event_day, 0.0)
        t_stat, p_value = float(t_stat), float(p_value)
    else:
        t_stat = p_value = np.nan

    summary = {
        "Instrument": instrument,
        "Event Type": event_label,
        "Num Events": int(len(detail)),
        "Mean Return t-1": float(detail["Return t-1"].mean()),
        "Mean Return t0": float(detail["Return t0"].mean()),
        "Mean Return t+1": float(detail["Return t+1"].mean()),
        "Median Return t0": float(detail["Return t0"].median()),
        "Mean Abs Return t0": float(detail["Abs Return t0"].mean()),
        "Std Return t0": float(detail["Return t0"].std(ddof=1)),
        "Positive t0 Share": float((detail["Return t0"] > 0).mean()),
        "t-stat (t0 vs 0)": t_stat,
        "p-value (t0 vs 0)": p_value,
        f"Event Window Vol (+/-{window}d)": event_vol,
        "Baseline Vol (other days)": base_vol,
        "Vol Ratio (event/baseline)": (event_vol / base_vol)
            if base_vol and not np.isnan(base_vol) and base_vol != 0 else np.nan,
        "Event Window Days": int(mask.sum()),
        "Baseline Days": int((~mask).sum()),
    }
    return summary, detail


def average_event_path(detail_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Average t-1 / t0 / t+1 return per instrument, reshaped for plotting.
    """
    if not detail_frames:
        return pd.DataFrame()
    all_detail = pd.concat([d for d in detail_frames if not d.empty],
                           ignore_index=True)
    if all_detail.empty:
        return pd.DataFrame()
    grouped = all_detail.groupby(["Instrument", "Event Type"])[
        ["Return t-1", "Return t0", "Return t+1"]].mean()
    return grouped.reset_index()
