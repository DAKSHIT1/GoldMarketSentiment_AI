"""
refresh_data.py
===============
Incremental live data refresh used before every Telegram alert.

  1. Append new XAU/USD 15m bars (TwelveData, America/New_York) to
     data/gold_15m_full.csv — only bars since the last stored one.
  2. Refresh UUP / IEF daily macro proxies (yfinance).
  3. Rebuild session rows for every session from the last stored one onward
     using the SAME feature code as the batch pipeline
     (feature_engine.generate_features + session_builder), over a trailing
     window with enough warm-up for EMA200 / rolling stats.
  4. Append only COMPLETED sessions to data/session_regime_v2.csv.
     An incomplete stored session (e.g. a partial last row) is replaced.

Run:  python refresh_data.py            (all steps)
      python refresh_data.py --no-fetch (rebuild sessions from local data only)
      python refresh_data.py --no-fetch --rebuild-from 2025-01-01
                                        (repair stored sessions from a date)

Non-tradable bars (weekend OTC quotes, 17:00 break) are dropped on write and
before building sessions — see session_builder.tradable_bars. Sessions are split
by trading day and holiday stubs (< MIN_SESSION_BARS bars) are dropped. When the
session schema changes (SESSION_SCHEMA vs data/session_regime_v2.meta.json) the
whole table is rebuilt automatically on the next run.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)  # feature_engine reads data/uup_1d.csv with relative paths

from config import API_KEY, DATA_15M_FILE                      # noqa: E402
from feature_engine import generate_features                    # noqa: E402
from session_builder import (tag_sessions, build_session_db_v2,  # noqa: E402
                             add_rolling_features, classify_v2,
                             session_is_complete, tradable_bars,
                             fill_forward_cols, session_start_flags,
                             trading_day, MIN_SESSION_BARS)

SESSION_FILE   = "data/session_regime_v2.csv"
SESSION_META   = "data/session_regime_v2.meta.json"
# Bump when session construction changes; a stored table built by an older schema is
# rebuilt from scratch on the next run (self-healing, no manual data migration).
#   1: original (18:00-bar session split)
#   2: tradable bars only, trading-day session split, sessions < MIN_SESSION_BARS dropped
SESSION_SCHEMA = 2
NY_TZ          = "America/New_York"
WARMUP_SESSIONS = 45          # sessions of history fed to generate_features before the first rebuilt one
CHUNK_DAYS     = 40           # 40 days of 15m bars < TwelveData's 5000-row limit
TD_SLEEP_SEC   = 8            # free tier: 8 requests / minute


def log(msg):
    print(f"[refresh] {msg}", flush=True)


def now_ny():
    return pd.Timestamp.now(tz=NY_TZ).tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
#  1. GOLD 15m BARS
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_chunk(start, end):
    params = {
        "symbol": "XAU/USD", "interval": "15min", "outputsize": 5000,
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": NY_TZ, "apikey": API_KEY,
    }
    for attempt in range(4):
        try:
            res = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30).json()
        except Exception as e:
            log(f"network error ({e}), retry {attempt + 1}/4")
            time.sleep(15)
            continue
        if "values" in res:
            df = pd.DataFrame(res["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            for c in ["open", "high", "low", "close"]:
                df[c] = df[c].astype(float)
            df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
            return df[["datetime", "open", "high", "low", "close", "volume"]]
        msg = str(res.get("message", res))
        if "No data is available" in msg:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        if res.get("code") == 429 or "credits" in msg.lower():
            log(f"rate limited, waiting 65s ({msg[:80]})")
            time.sleep(65)
            continue
        raise RuntimeError(f"TwelveData error: {msg}")
    raise RuntimeError("TwelveData: retries exhausted")


def update_gold_bars():
    if not API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY is not set")
    bars = pd.read_csv(DATA_15M_FILE, parse_dates=["datetime"])
    last = bars["datetime"].max()
    end_all = now_ny()
    log(f"stored bars end at {last}; fetching through {end_all:%Y-%m-%d %H:%M} NY")

    new = []
    start = last  # re-fetch the last bar too, in case it was still forming
    while start < end_all:
        end = min(start + timedelta(days=CHUNK_DAYS), end_all)
        chunk = _fetch_chunk(start, end)
        log(f"  {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}: {len(chunk)} bars")
        if len(chunk):
            new.append(chunk)
        start = end
        if start < end_all:
            time.sleep(TD_SLEEP_SEC)

    if not new:
        log("no new bars")
        return 0
    fresh = pd.concat(new)
    # drop the bar that is still forming (its 15 minutes have not finished)
    fresh = fresh[fresh["datetime"] + timedelta(minutes=15) <= end_all]
    merged = (pd.concat([bars, fresh])
                .drop_duplicates(subset="datetime", keep="last")
                .sort_values("datetime").reset_index(drop=True))
    n_raw = len(merged)
    merged = tradable_bars(merged)   # drop weekend OTC quotes + 17:00 break bars
    added = len(merged) - len(bars)
    tmp = DATA_15M_FILE + ".tmp"
    merged.to_csv(tmp, index=False)
    os.replace(tmp, DATA_15M_FILE)
    log(f"gold bars: {added:+d} net, {n_raw - len(merged)} non-tradable dropped "
        f"(now end at {merged['datetime'].max()})")
    return added


# ─────────────────────────────────────────────────────────────────────────────
#  2. MACRO (UUP / IEF daily)
# ─────────────────────────────────────────────────────────────────────────────
def update_macro():
    from fetch_macro import download_macro_yahoo
    ok = download_macro_yahoo("UUP", "data/uup_1d.csv")
    download_macro_yahoo("IEF", "data/ief_1d.csv")
    if not ok:   # UUP drives dxy_slope (half of the active rule) — fail the refresh step
        raise RuntimeError("UUP macro refresh failed (stored file kept)")


# ─────────────────────────────────────────────────────────────────────────────
#  3/4. SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_sessions_from(bars, from_date, warmup_sessions=WARMUP_SESSIONS):
    """
    Build session rows (no rolling features yet) for every session whose start
    is >= from_date, using generate_features on a trailing window.
    Returns DataFrame with a 'complete' flag.
    """
    bars = tradable_bars(bars.sort_values("datetime").reset_index(drop=True))
    starts = bars.index[session_start_flags(bars["datetime"])]
    start_times = bars.loc[starts, "datetime"]
    target = start_times[start_times >= from_date]
    if target.empty:
        return pd.DataFrame()
    first_target_pos = list(start_times.index).index(target.index[0])
    win_pos = max(0, first_target_pos - warmup_sessions)
    window = bars.iloc[start_times.index[win_pos]:].copy().reset_index(drop=True)

    feats = generate_features(window, is_live=False)
    feats = tag_sessions(feats)
    sess = build_session_db_v2(feats)
    sess["date"] = pd.to_datetime(sess["date"])
    sess = sess[sess["date"] >= from_date].reset_index(drop=True)

    last_bar = feats.groupby("session_id")["datetime"].max()
    nbars = feats.groupby("session_id").size()
    max_sid = feats["session_id"].max()
    sess["complete"] = [
        session_is_complete(d, last_bar[s], s < max_sid)
        for s, d in zip(sess["sid"], sess["date"])
    ]
    sess["nbars"] = [int(nbars[s]) for s in sess["sid"]]
    return sess


def _read_meta():
    try:
        with open(SESSION_META) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_meta():
    tmp = SESSION_META + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"schema": SESSION_SCHEMA}, f, indent=2)
        f.write("\n")
    os.replace(tmp, SESSION_META)


def rebuild_sessions(rebuild_from=None):
    """
    Rebuild stored sessions from the last stored one (default) or from the first
    stored session on/after `rebuild_from` (e.g. to repair history).
    """
    stored = pd.read_csv(SESSION_FILE)
    stored["date"] = pd.to_datetime(stored["date"])
    stored = stored.sort_values("date").reset_index(drop=True)
    bars = pd.read_csv(DATA_15M_FILE, parse_dates=["datetime"])

    # Always rebuild the last stored session: it may have been written while incomplete.
    from_date = stored["date"].iloc[-1]
    # Self-heal: a table from an older schema (or one containing weekend pseudo-sessions)
    # is rebuilt from the first stored session with the current code.
    schema = int(_read_meta().get("schema", 1))
    n_weekend = int(trading_day(stored["date"]).dt.day_of_week.isin([4, 5]).sum())
    if schema < SESSION_SCHEMA or n_weekend:
        log(f"session table schema {schema} (current {SESSION_SCHEMA}), {n_weekend} weekend rows "
            f"→ full rebuild")
        rebuild_from = stored["date"].iloc[0]
    if rebuild_from is not None:
        later = stored.loc[stored["date"] >= pd.Timestamp(rebuild_from), "date"]
        if len(later):
            from_date = later.iloc[0]
    fresh = build_sessions_from(bars, from_date)
    if fresh.empty:
        log("no sessions to rebuild")
        return stored

    incomplete = fresh[~fresh["complete"]]
    if len(incomplete):
        log(f"skipping in-progress session(s): {[str(d) for d in incomplete['date']]}")
    thin = fresh[fresh["complete"] & (fresh["nbars"] < MIN_SESSION_BARS)]
    if len(thin):
        log(f"dropping thin session(s) (<{MIN_SESSION_BARS} bars): {[str(d) for d in thin['date']]}")
    fresh = (fresh[fresh["complete"] & (fresh["nbars"] >= MIN_SESSION_BARS)]
             .drop(columns=["complete", "nbars"]).reset_index(drop=True))

    keep = stored[stored["date"] < from_date].reset_index(drop=True)
    base_sid = int(stored.loc[stored["date"] == from_date, "sid"].iloc[0])
    fresh["sid"] = range(base_sid, base_sid + len(fresh))

    combined = pd.concat([keep, fresh], ignore_index=True, sort=False)
    start_idx = len(keep)
    combined = add_rolling_features(combined, start_idx=start_idx)
    for i in range(start_idx, len(combined)):
        combined.loc[i, "regime_v2"] = classify_v2(combined.loc[i])
        combined.loc[i, "quarter"] = str(combined.loc[i, "date"].to_period("Q"))

    # research-only forward columns: row indices after a repair may shift, so recompute all
    combined = fill_forward_cols(combined)
    combined = combined[stored.columns]
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    tmp = SESSION_FILE + ".tmp"
    combined.to_csv(tmp, index=False)
    os.replace(tmp, SESSION_FILE)
    _write_meta()
    replaced = int((stored["date"] >= from_date).sum())
    log(f"sessions: {len(fresh)} written ({replaced} stored rows replaced); "
        f"latest complete = {combined['date'].iloc[-1]}")
    return combined


def refresh(fetch=True, rebuild_from=None):
    if fetch:
        update_gold_bars()
        update_macro()
    return rebuild_sessions(rebuild_from)


if __name__ == "__main__":
    rf = None
    if "--rebuild-from" in sys.argv:
        rf = sys.argv[sys.argv.index("--rebuild-from") + 1]
    refresh(fetch="--no-fetch" not in sys.argv, rebuild_from=rf)
