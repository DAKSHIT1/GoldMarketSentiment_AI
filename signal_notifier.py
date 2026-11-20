"""
signal_notifier.py
==================
Gold AI — Telegram notifier for the session-level macro-window strategy.

One message per COMPLETED session, never repeated:
  * the signal for session K (rules in signal_rules.py — same as the forward test),
    which sets the bias for the next WINDOW_SESSIONS sessions,
  * plus the session result recap and rolling stats (former EOD report).

Usage:
  python signal_notifier.py --auto       refresh data, send the latest completed
                                         session once (used by GitHub Actions)
  python signal_notifier.py --dry-run    same, but print instead of sending and
                                         do not record state
  python signal_notifier.py --no-refresh skip the data refresh (combine with the above)
  python signal_notifier.py --force      resend even if already sent
  python signal_notifier.py --test       send the latest message tagged as TEST (not recorded)

Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVEDATA_API_KEY come from the
environment (GitHub secrets) or the gitignored config_local.py.

Dedup: state/notifier_state.json (committed back by the workflow) records every
message key that was sent: "signal:<session start>" and "stale:<session start>".
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import signal_rules

ROOT         = os.path.dirname(os.path.abspath(__file__))
SESSION_FILE = os.path.join(ROOT, "data", "session_regime_v2.csv")
STATE_FILE   = os.path.join(ROOT, "state", "notifier_state.json")
MAX_STATE_KEYS = 400

# Latest completed session must have ended within this many hours, otherwise the
# data feed is considered stale (covers the weekend gap Fri 17:00 → Mon 17:00 NY).
STALE_AFTER_HOURS = 80


def _secret(name):
    v = os.environ.get(name, "")
    if not v:
        try:
            import config_local
            v = getattr(config_local, name, "") or ""
        except ImportError:
            pass
    return v


BOT_TOKEN = _secret("TELEGRAM_BOT_TOKEN")
CHAT_ID   = _secret("TELEGRAM_CHAT_ID")


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  STATE (dedup)
# ─────────────────────────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            s = json.load(f)
        s.setdefault("sent", {})
        return s
    return {"sent": {}}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    sent = state["sent"]
    if len(sent) > MAX_STATE_KEYS:
        keep = sorted(sent.items(), key=lambda kv: kv[1]["sent_at"])[-MAX_STATE_KEYS:]
        state["sent"] = dict(keep)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def already_sent(state, key):
    return key in state["sent"]


def mark_sent(state, key, **info):
    state["sent"][key] = {"sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}


# ─────────────────────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram(text, dry_run=False):
    if dry_run:
        print("─" * 60 + "\n[DRY RUN] would send:\n" + text + "\n" + "─" * 60)
        return True
    if not BOT_TOKEN or not CHAT_ID:
        log("❌ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — cannot send")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML",
                                "disable_web_page_preview": True},
                          timeout=15)
        ok = r.status_code == 200
        log(f"{'✅' if ok else '❌'} Telegram {r.status_code}" + ("" if ok else f": {r.text[:200]}"))
        return ok
    except Exception as e:
        log(f"❌ Telegram error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_sessions():
    df = pd.read_csv(SESSION_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def sf(val, d=0.0):
    try:
        v = float(val)
        return d if np.isnan(v) else v
    except (TypeError, ValueError):
        return d


def session_label(ts):
    """Session starting 18:00 NY on day D is the trading day D+1."""
    trade_day = (ts + pd.Timedelta(hours=6)).date()
    return f"{trade_day:%a %d %b %Y}"


def get_live_gold_price():
    try:
        from config import API_KEY
        if not API_KEY:
            return None
        r = requests.get("https://api.twelvedata.com/price",
                         params={"symbol": "XAU/USD", "apikey": API_KEY}, timeout=8).json()
        return float(r["price"]) if "price" in r else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def build_session_msg(sess, i, test=False):
    recs = sess.to_dict("records")
    row  = recs[i]
    sig  = signal_rules.classify_at(recs, i)
    d    = sig["direction"]
    mu, md = sf(row["max_up"]), sf(row["max_down"])
    o, c = sf(row["sess_open"]), sf(row["sess_close"])
    dpct = (c - o) / o * 100 if o > 0 else 0.0
    W = signal_rules.WINDOW_SESSIONS

    if d == "LONG":
        head = f"🟢🔼 <b>GOLD AI — LOOK FOR LONGS</b> 🟢🔼"
        plan = (f"  → Bias LONG for the next {W} sessions\n"
                f"  → Prefer pullbacks into lower levels / session VWAP\n"
                f"  → Invalidation: deep L2+ breakdown against the bias")
    elif d == "SHORT":
        head = f"🔴🔽 <b>GOLD AI — LOOK FOR SHORTS</b> 🔴🔽"
        plan = (f"  → Bias SHORT for the next {W} sessions\n"
                f"  → Prefer bounces into upper levels / session VWAP\n"
                f"  → Invalidation: strong U2+ breakout against the bias")
    else:
        head = f"⚪ <b>GOLD AI — {sig['label']}</b>"
        plan = "  → No new window. Sit on hands."

    if sig["signals"] and d != "NEUTRAL":
        sigs_txt = "\n".join(
            f"  • {s['name']} ({s['acc']:.1f}% OOS, n={s['n']})"
            for s in sorted(sig["signals"], key=lambda x: -x["acc"]))
        sig_block = (f"💪 <b>Strength:</b> {sig['label'].split('[')[-1].rstrip(']')}\n"
                     f"🎯 <b>Deciding signal:</b> {sig['key_signal']['name']} "
                     f"({sig['confidence']:.1f}% OOS)\n\n"
                     f"<b>Signals firing:</b>\n{sigs_txt}\n")
    else:
        sig_block = f"<b>Reason:</b> {sig['reason']}\n"

    # Active macro windows (directional signals from the previous W sessions)
    active = []
    for j in range(max(0, i - W + 1), i):
        sj = signal_rules.classify_at(recs, j)
        if sj["direction"] != "NEUTRAL":
            active.append(f"{'🟢' if sj['direction'] == 'LONG' else '🔴'}"
                          f"{(recs[j]['date'] + pd.Timedelta(hours=6)):%d %b}")
    active_txt = " ".join(active) if active else "none"

    # Recap of last 5 sessions
    recap = ""
    for j in range(max(0, i - 4), i + 1):
        r = recs[j]
        ro, rc = sf(r["sess_open"]), sf(r["sess_close"])
        rp = (rc - ro) / ro * 100 if ro > 0 else 0
        rd = signal_rules.classify_at(recs, j)["direction"]
        em = "🟢" if rd == "LONG" else "🔴" if rd == "SHORT" else "⚪"
        recap += f"  {em} {(r['date'] + pd.Timedelta(hours=6)):%a %d %b}: {rp:+.2f}%  [{rd}]\n"

    live = get_live_gold_price()
    live_txt = f"\n  Live spot now: ${live:,.2f}" if live else ""

    return (
        (f"🧪 <b>TEST MESSAGE</b>\n\n" if test else "")
        + f"{head}\n\n"
        f"📅 <b>Session:</b> {session_label(row['date'])} (closed)\n"
        f"{sig_block}\n"
        f"📊 <b>Session result:</b>\n"
        f"  Max levels UP: {mu:.0f}  |  DOWN: {md:.0f}\n"
        f"  Open ${o:,.2f} → Close ${c:,.2f}  ({dpct:+.2f}%){live_txt}\n\n"
        f"⚡ <b>Plan:</b>\n{plan}\n\n"
        f"🪟 <b>Windows still active:</b> {active_txt}\n\n"
        f"📋 <b>Last 5 sessions:</b>\n{recap}\n"
        f"<i>Gold AI Antigravity | XAU/USD | one alert per completed session</i>"
    )


def build_stale_msg(last_date, age_h):
    return (f"⚠️ <b>GOLD AI — DATA FEED STALE</b>\n\n"
            f"Latest completed session: {session_label(last_date)}\n"
            f"That is {age_h:.0f}h old — no signal will be sent until data updates.\n"
            f"Check the GitHub Actions log (TwelveData / refresh step).")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(refresh=True, dry_run=False, force=False, test=False):
    if refresh:
        import refresh_data
        try:
            refresh_data.refresh(fetch=True)
        except Exception as e:
            log(f"❌ data refresh failed: {e}")   # fall through: stale guard will catch it

    sess  = load_sessions()
    i     = len(sess) - 1
    last  = sess["date"].iloc[i]
    state = load_state()

    ended = last + pd.Timedelta(hours=23)                       # 17:00 NY next day
    age_h = (pd.Timestamp.now(tz="America/New_York").tz_localize(None) - ended).total_seconds() / 3600
    log(f"latest completed session {last} (ended {age_h:.1f}h ago)")

    if test:
        return 0 if send_telegram(build_session_msg(sess, i, test=True), dry_run) else 1

    if age_h > STALE_AFTER_HOURS:
        key = f"stale:{last:%Y-%m-%d %H:%M}"
        if already_sent(state, key) and not force:
            log(f"stale data; warning already sent ({key})")
            return 1
        if send_telegram(build_stale_msg(last, age_h), dry_run) and not dry_run:
            mark_sent(state, key)
            save_state(state)
        return 1

    key = f"signal:{last:%Y-%m-%d %H:%M}"
    if already_sent(state, key) and not force:
        log(f"already sent {key} — nothing to do")
        return 0

    sig = signal_rules.classify_at(sess.to_dict("records"), i)
    msg = build_session_msg(sess, i)
    if send_telegram(msg, dry_run):
        if not dry_run:
            mark_sent(state, key, direction=sig["direction"], label=sig["label"])
            save_state(state)
        return 0
    return 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)
    sys.exit(run(refresh="--no-refresh" not in args,
                 dry_run="--dry-run" in args,
                 force="--force" in args,
                 test="--test" in args))
