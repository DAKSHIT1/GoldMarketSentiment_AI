"""
level_rejection_scanner.py
==========================
Standalone 30-minute Level Rejection Scanner for XAU/USD.

Looks ONLY at the most recently CLOSED 30m candle and alerts (Telegram) when it
cleanly rejects one of today's Daily math levels (MathLevelEngine, mode=Daily,
mult=4, session open 18:00 America/New_York): U1-U8 / L1-L8 (Open excluded).

  SHORT: opens <= level+$1.50, high >= level-$1.50, closes BELOW the level,
         and upper wick >= 35% of range OR bearish body >= 40% of range.
  LONG : opens >= level-$1.50, low <= level+$1.50, closes ABOVE the level,
         and lower wick >= 35% of range OR bullish body >= 40% of range.
  Bars with range < $3.00 are ignored.

Silent when there is no rejection. Dedup: state/level_rejection_state.json stores the
last alerted candle so the same bar never alerts twice.

Usage:
  python level_rejection_scanner.py              check once and alert
  python level_rejection_scanner.py --dry-run    print instead of sending; state NOT saved
  python level_rejection_scanner.py --watch      local loop, checks shortly after :00 / :30
  python level_rejection_scanner.py --asof "2026-11-24 09:30"   (dry-run replay, NY time:
                                                 treat that bar-close time as "now")

Secrets: TWELVEDATA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID from the environment
(or the gitignored config_local.py: API_KEY / TELEGRAM_*).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
import requests

from level_engine import MathLevelEngine

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(ROOT, "state", "level_rejection_state.json")

NY_TZ = "America/New_York"
IST_TZ = "Asia/Kolkata"
BAR_MIN = 30
TOUCH_DIST = 1.50       # $ — high/low must come within this of the level (or pierce it)
WICK_FRAC = 0.35        # rejection wick >= 35% of the bar range
BODY_FRAC = 0.40        # OR directional body >= 40% of the bar range
MIN_RANGE = 3.00        # $ — ignore dead/tiny bars
MAX_LEVEL = 8           # U1-U8 / L1-L8
FETCH_BARS = 120        # ~2.5 days of 30m bars: always covers the current 18:00 session open
SETTLE_SEC = 20         # a bar counts as closed this many seconds after its end


def _secret(*names):
    for n in names:
        v = os.environ.get(n, "")
        if v:
            return v
    try:
        import config_local
        for n in names:
            v = getattr(config_local, n, "") or ""
            if v:
                return v
    except ImportError:
        pass
    return ""


API_KEY = _secret("TWELVEDATA_API_KEY", "API_KEY")
BOT_TOKEN = _secret("TELEGRAM_BOT_TOKEN")
CHAT_ID = _secret("TELEGRAM_CHAT_ID")


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ── data ─────────────────────────────────────────────────────────────────────
def fetch_30m():
    """Latest 30m XAU/USD bars, bar-START times in naive New York time, ascending."""
    if not API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY is not set")
    params = {"symbol": "XAU/USD", "interval": "30min", "outputsize": FETCH_BARS,
              "timezone": NY_TZ, "apikey": API_KEY}
    res = None
    for attempt in range(3):
        try:
            res = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30).json()
        except Exception as e:
            log(f"network error ({e}), retry {attempt + 1}/3")
            time.sleep(5)
            continue
        if "values" in res:
            break
        msg = str(res.get("message", res))
        if res.get("code") == 429 or "credits" in msg.lower():
            log("rate limited, waiting 20s")
            time.sleep(20)
            continue
        raise RuntimeError(f"TwelveData error: {msg}")
    if not res or "values" not in res:
        raise RuntimeError("TwelveData: no data after retries")
    df = pd.DataFrame(res["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)


def closed_bars(df, now_ny):
    """Drop any bar that is still forming at now_ny."""
    end = df["datetime"] + pd.Timedelta(minutes=BAR_MIN)
    return df[end + pd.Timedelta(seconds=SETTLE_SEC) <= now_ny].reset_index(drop=True)


# ── levels ───────────────────────────────────────────────────────────────────
def session_start(bar_start):
    """18:00 NY open of the trading session containing bar_start."""
    d = bar_start.normalize() + pd.Timedelta(hours=18)
    return d if bar_start >= d else d - pd.Timedelta(days=1)


def build_levels(df, bar_start):
    """Daily levels for the session containing bar_start (open = first bar of that session)."""
    s0 = session_start(bar_start)
    sess = df[(df["datetime"] >= s0) & (df["datetime"] <= bar_start)]
    if sess.empty:
        return None, None
    eng = MathLevelEngine("Daily")
    eng.reset_levels(float(sess.iloc[0]["open"]))
    lv = {}   # Open is deliberately excluded: only U1-U8 / L1-L8
    for i in range(MAX_LEVEL):
        lv[f"U{i + 1}"] = float(eng.upper_levels[i])
        lv[f"L{i + 1}"] = float(eng.lower_levels[i])
    return lv, s0


# ── rejection logic ──────────────────────────────────────────────────────────
def detect(bar, levels):
    o, h, l, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
    rng = h - l
    if rng < MIN_RANGE:
        return None
    body_dn = max(o - c, 0) / rng
    body_up = max(c - o, 0) / rng
    up_wick = (h - max(o, c)) / rng
    lo_wick = (min(o, c) - l) / rng
    hits = []

    if up_wick >= WICK_FRAC or body_dn >= BODY_FRAC:
        for name, p in levels.items():
            if name.startswith("U") and o <= p + TOUCH_DIST and h >= p - TOUCH_DIST and c < p:
                hits.append(("SHORT", name, p, body_dn, up_wick, abs(h - p)))
    if lo_wick >= WICK_FRAC or body_up >= BODY_FRAC:
        for name, p in levels.items():
            if name.startswith("L") and o >= p - TOUCH_DIST and l <= p + TOUCH_DIST and c > p:
                hits.append(("LONG", name, p, body_up, lo_wick, abs(l - p)))
    if not hits:
        return None
    # Single alert per bar: the level the extreme came closest to
    d, name, p, body, wick, _ = min(hits, key=lambda x: x[5])
    return {"direction": d, "level": name, "level_price": p, "body": body, "wick": wick,
            "open": o, "high": h, "low": l, "close": c}


def build_msg(sig, bar_start):
    end_ist = (bar_start + pd.Timedelta(minutes=BAR_MIN)).tz_localize(NY_TZ).tz_convert(IST_TZ)
    icon = "🔴" if sig["direction"] == "SHORT" else "🟢"
    wick_name = "Upper wick" if sig["direction"] == "SHORT" else "Lower wick"
    return (f"{icon} <b>{sig['direction']}</b> — XAU/USD 30m rejection\n"
            f"Level: <b>{sig['level']} @ {sig['level_price']:,.2f}</b>\n"
            f"30m close: {sig['close']:,.2f}\n"
            f"Body {sig['body'] * 100:.0f}% | {wick_name} {sig['wick'] * 100:.0f}%\n"
            f"Bar closed: {end_ist:%d %b %H:%M} IST")


# ── state / telegram ─────────────────────────────────────────────────────────
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"last_alerted_bar": None}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, STATE_FILE)


def send_telegram(text, dry_run):
    if dry_run:
        print("─" * 60 + "\n[DRY RUN] would send:\n" + text + "\n" + "─" * 60)
        return True
    if not BOT_TOKEN or not CHAT_ID:
        log("❌ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=15)
        log(f"{'✅' if r.status_code == 200 else '❌'} Telegram {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        log(f"❌ Telegram error: {e}")
        return False


# ── main check ───────────────────────────────────────────────────────────────
def check_once(dry_run, asof=None):
    df = fetch_30m()
    now_ny = asof if asof is not None else pd.Timestamp.now(tz=NY_TZ).tz_localize(None)
    bars = closed_bars(df, now_ny)
    if bars.empty:
        log("no closed 30m bars")
        return 0
    bar = bars.iloc[-1]
    bar_start = bar["datetime"]
    age_min = (now_ny - bar_start).total_seconds() / 60 - BAR_MIN
    if age_min > 90:
        log(f"latest closed bar {bar_start} is {age_min:.0f} min old (market closed / feed stale) — skip")
        return 0
    levels, _ = build_levels(bars, bar_start)
    if levels is None:
        log("no session open found — skip")
        return 0
    sig = detect(bar, levels)
    key = f"{bar_start:%Y-%m-%d %H:%M}"
    if sig is None:
        log(f"bar {key} NY: no rejection (silent)")
        return 0
    state = load_state()
    if state.get("last_alerted_bar") == key:
        log(f"bar {key} NY: already alerted — skip")
        return 0
    if send_telegram(build_msg(sig, bar_start), dry_run):
        if not dry_run:
            state["last_alerted_bar"] = key
            save_state(state)
        return 0
    return 1


def watch(dry_run):
    log("watch mode: checking shortly after every :00 / :30 (Ctrl+C to stop)")
    while True:
        now = datetime.now()
        wait = (BAR_MIN - (now.minute % BAR_MIN)) * 60 - now.second + SETTLE_SEC
        time.sleep(max(wait, 1))
        try:
            check_once(dry_run)
        except Exception as e:
            log(f"check failed: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print instead of sending; no state saved")
    ap.add_argument("--watch", action="store_true", help="local loop at :00 / :30")
    ap.add_argument("--asof", help='replay: treat "YYYY-MM-DD HH:MM" (NY time) as now (use with --dry-run)')
    a = ap.parse_args()
    if a.asof and not a.dry_run:
        ap.error("--asof requires --dry-run")
    if a.watch:
        watch(a.dry_run)
        return 0
    try:
        return check_once(a.dry_run, pd.Timestamp(a.asof) if a.asof else None)
    except Exception as e:
        log(f"❌ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
