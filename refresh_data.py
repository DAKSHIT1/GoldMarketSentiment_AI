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
"""

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
                             session_is_complete)

SESSION_FILE   = "data/session_regime_v2.csv"
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
    added = len(merged) - len(bars)
    merged.to_csv(DATA_15M_FILE, index=False)
    log(f"gold bars: +{added} (now end at {merged['datetime'].max()})")
    return added


# ─────────────────────────────────────────────────────────────────────────────
#  2. MACRO (UUP / IEF daily)
# ─────────────────────────────────────────────────────────────────────────────
def update_macro():
    from fetch_macro import download_macro_yahoo
    download_macro_yahoo("UUP", "data/uup_1d.csv")
    download_macro_yahoo("IEF", "data/ief_1d.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  3/4. SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_sessions_from(bars, from_date, warmup_sessions=WARMUP_SESSIONS):
    """
    Build session rows (no rolling features yet) for every session whose start
    is >= from_date, using generate_features on a trailing window.
    Returns DataFrame with a 'complete' flag.
    """
    bars = bars.sort_values("datetime").reset_index(drop=True)
    starts = bars.index[(bars["datetime"].dt.hour == 18) & (bars["datetime"].dt.minute == 0)]
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
    max_sid = feats["session_id"].max()
    sess["complete"] = [
        session_is_complete(d, last_bar[s], s < max_sid)
        for s, d in zip(sess["sid"], sess["date"])
    ]
    return sess


def rebuild_sessions():
    stored = pd.read_csv(SESSION_FILE)
    stored["date"] = pd.to_datetime(stored["date"])
    stored = stored.sort_values("date").reset_index(drop=True)
    bars = pd.read_csv(DATA_15M_FILE, parse_dates=["datetime"])

    # Always rebuild the last stored session: it may have been written while incomplete.
    from_date = stored["date"].iloc[-1]
    fresh = build_sessions_from(bars, from_date)
    if fresh.empty:
        log("no sessions to rebuild")
        return stored

    incomplete = fresh[~fresh["complete"]]
    if len(incomplete):
        log(f"skipping in-progress session(s): {[str(d) for d in incomplete['date']]}")
    fresh = fresh[fresh["complete"]].drop(columns=["complete"]).reset_index(drop=True)

    keep = stored[stored["date"] < from_date].reset_index(drop=True)
    base_sid = int(stored.loc[stored["date"] == from_date, "sid"].iloc[0])
    fresh["sid"] = range(base_sid, base_sid + len(fresh))

    combined = pd.concat([keep, fresh], ignore_index=True, sort=False)
    start_idx = len(keep)
    combined = add_rolling_features(combined, start_idx=start_idx)
    for i in range(start_idx, len(combined)):
        combined.loc[i, "regime_v2"] = classify_v2(combined.loc[i])
        combined.loc[i, "quarter"] = str(combined.loc[i, "date"].to_period("Q"))
        for c in ["fwd_ret", "fwd_pts", "fwd_up", "fwd_down"]:
            combined.loc[i, c] = np.nan

    combined = combined[stored.columns]
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined.to_csv(SESSION_FILE, index=False)
    replaced = 1 if len(fresh) and fresh["date"].iloc[0] == from_date else 0
    log(f"sessions: {len(fresh)} written ({replaced} replaced, {len(fresh) - replaced} new); "
        f"latest complete = {combined['date'].iloc[-1]}")
    return combined


def refresh(fetch=True):
    if fetch:
        update_gold_bars()
        update_macro()
    return rebuild_sessions()


if __name__ == "__main__":
    refresh(fetch="--no-fetch" not in sys.argv)
