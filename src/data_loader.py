"""
data_loader.py
==============
Downloads, validates and stores daily OHLCV data for the futures contracts
studied in this project.

Design notes
------------
* TWO independent data sources are supported, selected with CONFIG["data_source"]
  or the --source flag. Both are free and need no API key:

    "yahoo"  Yahoo Finance via the `yfinance` package. Tickers ES=F / GC=F / CL=F.
    "stooq"  Stooq (stooq.com) plain-CSV endpoint, read with the standard
             library - no extra package at all. Tickers es.f / gc.f / cl.f.

  Having two matters for a practical reason: Yahoo rate-limits, and an
  unattended pipeline that depends on one free endpoint is a pipeline that
  breaks. It also matters for a research reason - running the same study on
  two independent providers is the cheapest possible check that a result is
  not an artefact of one vendor's data.

* Both provide *continuous front-month* futures series: the series rolls to
  the next contract as the front month expires. That is convenient but it
  means the price series contains small roll gaps. We work in *percentage
  returns*, which keeps the effect small, and the limitation is documented
  in the README.
* Every downloaded file is written to `data/raw/` untouched, and the cleaned
  version to `data/processed/`. Keeping both means the cleaning step is
  auditable - a reviewer can diff the two.
* If the network is unavailable, an explicitly-labelled synthetic dataset can
  be generated so the pipeline can still be smoke-tested offline. Synthetic
  output is stamped as such everywhere so it can never be mistaken for real
  market results.
"""

from __future__ import annotations

import io
import os
import time
import urllib.error
import urllib.request
import zlib

import numpy as np
import pandas as pd

# Columns the rest of the project relies on.
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# ----------------------------------------------------------------------
# Folder helpers
# ----------------------------------------------------------------------
def ensure_dirs(*paths: str) -> None:
    """Create every folder in `paths` if it does not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Recent versions of yfinance return a MultiIndex column layout
    (level 0 = field, level 1 = ticker) even for a single ticker.
    Flatten it back to plain column names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Keep the level that contains the field names ('Open', 'Close', ...).
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = [c[0] for c in df.columns]
    return df


def download_one(ticker: str, start: str, end: str | None,
                 max_retries: int = 3, pause_seconds: float = 2.0) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single ticker.

    Returns an empty DataFrame (never raises) if the download fails, so that
    one bad ticker cannot abort the whole run. The caller decides what to do.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [ERROR] yfinance is not installed. Run: pip install -r requirements.txt")
        return pd.DataFrame()

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # auto_adjust=False keeps the raw OHLC and adds 'Adj Close' when
            # Yahoo provides one. Futures have no dividends/splits, so for
            # ES=F / GC=F / CL=F 'Adj Close' equals 'Close'; we still keep the
            # column when it exists so the code also works on equity tickers.
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                return _flatten_columns(df, ticker)

            # Fallback path: the Ticker().history() endpoint sometimes works
            # when the bulk download endpoint returns nothing.
            hist = yf.Ticker(ticker).history(start=start, end=end,
                                             interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty:
                return _flatten_columns(hist, ticker)

            last_error = "empty response"
        except Exception as exc:                      # network / API errors
            last_error = repr(exc)

        if attempt < max_retries:
            print(f"  [WARN] {ticker}: attempt {attempt} failed ({last_error}). Retrying...")
            time.sleep(pause_seconds)

    print(f"  [ERROR] {ticker}: download failed after {max_retries} attempts ({last_error}).")
    return pd.DataFrame()


# ----------------------------------------------------------------------
# Source 2: Stooq (no API key, no third-party package)
# ----------------------------------------------------------------------
# Stooq serves daily history as a plain CSV from a URL. That makes it the
# simplest possible fallback: no package to install, no key to register, no
# authentication. Symbols for continuous futures use a '.f' suffix.
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"

# Mapping from this project's canonical (Yahoo-style) tickers to Stooq symbols.
STOOQ_SYMBOLS = {
    "ES=F": "es.f",     # S&P 500 E-mini futures
    "GC=F": "gc.f",     # Gold futures
    "CL=F": "cl.f",     # Crude oil (WTI) futures
    # ETF proxies, if the futures symbols ever stop working. They track the
    # same exposures but are NOT futures - say so if you use them.
    "SPY": "spy.us",
    "GLD": "gld.us",
    "USO": "uso.us",
}


def download_stooq(ticker: str, start: str, end: str | None,
                   max_retries: int = 3, pause_seconds: float = 2.0) -> pd.DataFrame:
    """
    Download daily OHLCV from Stooq's CSV endpoint.

    Returns an empty DataFrame on any failure - the caller decides what to do.

    Stooq returns the whole available history and does not take date
    parameters, so the range is trimmed here after download.
    """
    symbol = STOOQ_SYMBOLS.get(ticker, ticker.lower())
    url = STOOQ_URL.format(symbol=symbol)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # Stooq rejects the default Python user-agent, so set a normal one.
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8", errors="replace")

            # Failures come back as plain text with a 200 status, not as an
            # HTTP error - so check the body before trying to parse it.
            head = payload[:200].lower()
            if "exceeded" in head or "limit" in head:
                last_error = "Stooq daily request limit reached"
            elif not payload.strip() or "date" not in head:
                last_error = f"unexpected response: {payload[:80]!r}"
            else:
                df = pd.read_csv(io.StringIO(payload))
                if df.empty:
                    last_error = "empty CSV"
                else:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                    df = df.loc[pd.Timestamp(start):
                                pd.Timestamp(end) if end else None]
                    if "Volume" not in df.columns:
                        # Some Stooq futures series carry no volume column at
                        # all. Fill it rather than failing - volume is only
                        # used by the volume-proxy event study.
                        df["Volume"] = np.nan
                    if not df.empty:
                        return df
                    last_error = "no rows in the requested date range"
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last_error = repr(exc)
        except Exception as exc:
            last_error = repr(exc)

        if attempt < max_retries:
            print(f"  [WARN] {ticker} (stooq): attempt {attempt} failed "
                  f"({last_error}). Retrying...")
            time.sleep(pause_seconds)

    print(f"  [ERROR] {ticker} (stooq): download failed after {max_retries} "
          f"attempts ({last_error}).")
    return pd.DataFrame()


# ----------------------------------------------------------------------
# Source dispatcher
# ----------------------------------------------------------------------
DOWNLOADERS = {
    "yahoo": download_one,
    "stooq": download_stooq,
}


def download(ticker: str, config: dict) -> pd.DataFrame:
    """
    Fetch one ticker from the configured source, falling back to the other
    source if the first one fails and `fallback_source` is enabled.

    The fallback is on by default because the two providers fail for
    unrelated reasons - Yahoo rate-limits, Stooq caps daily requests - so
    trying both turns two flaky endpoints into one fairly reliable one.
    """
    source = config.get("data_source", "yahoo")
    if source not in DOWNLOADERS:
        raise ValueError(f"Unknown data_source {source!r}. "
                         f"Options: {list(DOWNLOADERS)}")

    order = [source]
    if config.get("fallback_source", True):
        order += [s for s in DOWNLOADERS if s != source]

    for i, name in enumerate(order):
        if i > 0:
            print(f"  [FALLBACK] Trying '{name}' instead of '{order[i-1]}'.")
        df = DOWNLOADERS[name](ticker, config["start_date"], config["end_date"],
                               max_retries=config.get("download_retries", 3))
        if not df.empty:
            if i > 0:
                print(f"  [SOURCE] {ticker}: served by '{name}'.")
            return df
    return pd.DataFrame()


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------
def clean_ohlcv(df: pd.DataFrame, ticker: str, verbose: bool = True) -> pd.DataFrame:
    """
    Clean a raw OHLCV frame.

    Steps, in order:
      1. Normalise the index to a plain DatetimeIndex named 'Date'.
      2. Keep only the columns we need; fail loudly if a core one is absent.
      3. Sort by date and drop duplicate dates (keep the last observation).
      4. Drop rows with a missing Close.
      5. Handle NON-POSITIVE closes - see the long note in _handle_non_positive.
         WTI crude settled at about -$37 on 20 April 2020, so this is a real
         case in this dataset, not a hypothetical one.
      6. Drop only STRUCTURALLY IMPOSSIBLE bars (High < Low). Bars where the
         close sits outside the day's high/low range are COUNTED AND KEPT,
         because for futures a settlement price is set by an exchange
         procedure and can legitimately fall outside the traded range.
         Deleting those would quietly throw away good data.
      7. Forward-fill *only* Volume gaps (a missing volume print is a data
         artefact; a missing price is not something we are willing to invent).

    The frame gains one extra column, `PriceBreak`: True on any day whose
    previous observation was removed. That day's return is then measured
    between the two surrounding POSITIVE closes - a real, tradable move -
    rather than against the removed price. The column exists so those days
    stay visible and auditable in every output file.
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Index -------------------------------------------------------------
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)     # drop timezone for clean CSVs
    df.index.name = "Date"

    # 2. Columns -----------------------------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [ERROR] {ticker}: missing required columns {missing}. Skipping.")
        return pd.DataFrame()

    keep = REQUIRED_COLUMNS + (["Adj Close"] if "Adj Close" in df.columns else [])
    df = df[keep].copy()

    n_start = len(df)

    # 3. Order and duplicates ---------------------------------------------
    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    n_dup = int(dup.sum())
    df = df[~dup]

    # 4. Missing closes ----------------------------------------------------
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
    missing_close = df["Close"].isna()
    n_missing = int(missing_close.sum())
    df = df[~missing_close]

    # 5. Non-positive closes (the April 2020 crude oil case) ---------------
    df, n_non_positive = _handle_non_positive(df, ticker, verbose)

    # 6. Bar geometry ------------------------------------------------------
    # Structurally impossible: the high is below the low. Nothing can be done
    # with such a bar, so it goes.
    impossible = (df["High"] < df["Low"]).fillna(False)
    n_impossible = int(impossible.sum())
    df = df[~impossible]

    # Settlement outside the traded range: legal for futures. Counted, kept.
    outside = ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).fillna(False)
    n_outside = int(outside.sum())

    # 7. Volume ------------------------------------------------------------
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df["Volume"] = df["Volume"].replace(0, np.nan).ffill()

    if "PriceBreak" not in df.columns:
        df["PriceBreak"] = False

    if verbose:
        print(f"  [CLEAN] {ticker}: {n_start} rows in -> {len(df)} rows out "
              f"(dropped {n_dup} duplicate dates, {n_missing} missing closes, "
              f"{n_non_positive} non-positive closes, {n_impossible} impossible bars)")
        if n_outside:
            pct = 100.0 * n_outside / max(len(df), 1)
            print(f"  [NOTE]  {ticker}: {n_outside} bars ({pct:.1f}%) have a close "
                  f"outside the day's high/low range. Kept - for futures the "
                  f"settlement price is set by exchange procedure and can sit "
                  f"outside the traded range. Investigate if this share is large.")
    return df


def _handle_non_positive(df: pd.DataFrame, ticker: str,
                         verbose: bool) -> tuple[pd.DataFrame, int]:
    """
    Deal with closes at or below zero.

    Why this needs its own function
    -------------------------------
    On 20 April 2020 the front-month WTI contract settled at roughly -$37.
    That is a genuine, historically important price, not a data error - but a
    PERCENTAGE return is meaningless across it. Going from +$20 to -$37 gives
    pct_change() a number (-2.85, i.e. "-285%"), and that number is nonsense:
    percentage returns assume a positive price base, and a log return is
    undefined outright.

    So the treatment is:
      * remove the non-positive observations (they cannot be part of a
        percentage-return series),
      * mark the FIRST SURVIVING DAY AFTER each removal with PriceBreak=True,
        so the affected day is visible and auditable in every output file,
      * measure that day's return between the two surrounding POSITIVE closes
        - $18.27 on 17 April to $10.01 on 21 April, a real and tradable -45%,
      * say loudly what happened, with dates, rather than doing it silently.

    What this project does NOT do is blank that return. An earlier version
    did, on the reasoning that the gap was unmeasurable. That was wrong in a
    way worth recording: it removed a genuine 45% loss from the series and
    inflated crude's buy-and-hold return from about 10% a year to 17%. The
    undefined quantity is the return to and from the negative price - not the
    move between the two positive prices on either side of it.

    Residual distortion, stated plainly: that return spans more than one
    trading day, so it slightly overstates single-day volatility on that date.
    And a study that took April 2020 seriously would model the contract roll
    and the negative settlement explicitly, which is beyond this project's
    scope. Both belong in the limitations section, and both are there.
    """
    non_positive = (df["Close"] <= 0).fillna(False)
    n = int(non_positive.sum())
    if n == 0:
        df = df.copy()
        df["PriceBreak"] = False
        return df, 0

    share = n / len(df)
    if share > 0.05:
        raise ValueError(
            f"{ticker}: {n} of {len(df)} closes ({share:.1%}) are non-positive. "
            f"That is too many to be the April-2020 crude episode - the series "
            f"is probably corrupt. Refusing to analyse it."
        )

    dates = [d.date().isoformat() for d in df.index[non_positive]]
    if verbose:
        print(f"  [PRICE] {ticker}: {n} non-positive close(s) found on "
              f"{', '.join(dates)}.")
        print(f"  [PRICE] Dropped: a percentage return across a negative price "
              f"is undefined. The next day is flagged PriceBreak and its return "
              f"is measured between the two surrounding positive closes - a "
              f"real move, not a blank. See README assumption 8.")

    # Position (not label) of the first surviving row after each removed row.
    positions = np.flatnonzero(non_positive.to_numpy())
    keep_mask = ~non_positive.to_numpy()
    kept_index = df.index[keep_mask]

    break_dates = []
    for pos in positions:
        later = np.flatnonzero(keep_mask[pos + 1:])
        if later.size:
            break_dates.append(df.index[pos + 1 + later[0]])

    out = df[keep_mask].copy()
    out["PriceBreak"] = False
    if break_dates:
        out.loc[kept_index.isin(pd.DatetimeIndex(break_dates)), "PriceBreak"] = True
    return out, n


# ----------------------------------------------------------------------
# Offline synthetic data (clearly labelled - for pipeline testing only)
# ----------------------------------------------------------------------
def generate_synthetic(ticker: str, start: str, end: str | None,
                       seed: int = 0) -> pd.DataFrame:
    """
    Build a random-walk OHLCV series so the pipeline can be run without a
    network connection.

    THIS IS NOT MARKET DATA. It exists purely so the code can be smoke-tested
    offline. Any numbers produced from it are meaningless and every output
    file produced in this mode is stamped SYNTHETIC.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    dates = pd.bdate_range(start=start, end=end, name="Date")
    # crc32 (not hash()) so the same ticker always gives the same series -
    # Python randomises str hashing per process, which would make the
    # offline demo irreproducible run to run.
    rng = np.random.default_rng((zlib.crc32(ticker.encode()) + seed) % (2**32))

    n = len(dates)
    # Mild trend + volatility clustering so the series at least *behaves* like
    # a price series (the strategies then exercise realistic code paths).
    vol = 0.008 * (1 + 0.5 * np.abs(np.sin(np.linspace(0, 12 * np.pi, n))))
    shocks = rng.normal(0.0002, 1.0, n) * vol
    close = 100.0 * np.exp(np.cumsum(shocks))

    intraday = np.abs(rng.normal(0, 1, n)) * vol * close
    open_ = close * (1 + rng.normal(0, 0.3, n) * vol)
    high = np.maximum.reduce([open_, close]) + intraday
    low = np.minimum.reduce([open_, close]) - intraday
    volume = rng.integers(50_000, 400_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Adj Close": close, "Volume": volume},
        index=dates,
    )


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def load_market_data(config: dict) -> tuple[dict[str, pd.DataFrame], bool]:
    """
    Download (or generate), clean and persist data for every configured ticker.

    Returns
    -------
    data : dict mapping ticker -> cleaned OHLCV DataFrame
    synthetic_used : True if ANY series came from the offline generator
    """
    raw_dir = config["raw_data_dir"]
    proc_dir = config["processed_data_dir"]
    ensure_dirs(raw_dir, proc_dir)

    tickers = config["tickers"]
    start = config["start_date"]
    end = config["end_date"]
    offline = config.get("offline_demo", False)
    source = config.get("data_source", "yahoo")
    allow_fallback = config.get("allow_synthetic_fallback", False)
    min_rows = config.get("min_rows_required", 300)

    data: dict[str, pd.DataFrame] = {}
    synthetic_used = False

    for ticker, name in tickers.items():
        print(f"\n[DATA] {ticker} ({name}) via '{source}'")

        raw_path = os.path.join(raw_dir, f"{ticker.replace('=', '_')}_raw.csv")

        if offline:
            raw = generate_synthetic(ticker, start, end)
            synthetic_used = True
            print("  [MODE] OFFLINE DEMO - synthetic random-walk data, NOT market data.")
        elif config.get("use_cache", False) and os.path.exists(raw_path):
            # Re-run the whole study on the data already downloaded. Useful
            # for re-analysis without hitting the provider again, and for
            # reproducing an earlier run exactly.
            raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            print(f"  [CACHE] Using the previously downloaded file "
                  f"{os.path.basename(raw_path)} ({len(raw)} rows). "
                  f"No network request made.")
        else:
            raw = download(ticker, config)
            if raw.empty and allow_fallback:
                print("  [FALLBACK] Download failed; generating SYNTHETIC data so the "
                      "pipeline can still run. Results from this series are NOT real.")
                raw = generate_synthetic(ticker, start, end)
                synthetic_used = True

        if raw.empty:
            print(f"  [SKIP] {ticker}: no usable data.")
            continue

        # Persist the raw pull before touching it (skipped when the raw file
        # IS the source - no point rewriting it unchanged).
        if not config.get("use_cache", False) or not os.path.exists(raw_path):
            raw.to_csv(raw_path)

        clean = clean_ohlcv(raw, ticker)
        if len(clean) < min_rows:
            print(f"  [SKIP] {ticker}: only {len(clean)} clean rows, "
                  f"need at least {min_rows}.")
            continue

        proc_path = os.path.join(proc_dir, f"{ticker.replace('=', '_')}_clean.csv")
        clean.to_csv(proc_path)
        print(f"  [SAVE] {raw_path} / {proc_path}")
        print(f"  [RANGE] {clean.index.min().date()} -> {clean.index.max().date()} "
              f"({len(clean)} trading days)")

        data[ticker] = clean

    if not data:
        raise RuntimeError(
            "No instrument produced usable data.\n"
            "  * Check your internet connection and that yfinance is installed\n"
            "    (pip install -r requirements.txt).\n"
            "  * Yahoo occasionally rate-limits; waiting a minute and re-running\n"
            "    usually fixes it.\n"
            "  * Try the other provider: 'python main.py --source stooq'\n"
            "    (Stooq needs no package and no API key).\n"
            "  * To test the pipeline itself without a network connection, run\n"
            "    'python main.py --offline' - but note that those results are\n"
            "    generated from a random walk and are meaningless."
        )

    _save_combined(data, proc_dir)
    return data, synthetic_used


def _save_combined(data: dict[str, pd.DataFrame], proc_dir: str) -> None:
    """
    Write one tidy ('long') file holding every instrument, alongside the
    per-instrument files. Long format is used because the instruments do not
    share an identical trading calendar - an outer join in wide format would
    create NaN holes that are easy to mishandle.
    """
    frames = []
    for ticker, df in data.items():
        tmp = df.copy()
        tmp.insert(0, "Ticker", ticker)
        frames.append(tmp.reset_index())
    combined = pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"])
    out = os.path.join(proc_dir, "combined_prices.csv")
    combined.to_csv(out, index=False)
    print(f"\n[SAVE] Combined dataset -> {out} ({len(combined)} rows)")
