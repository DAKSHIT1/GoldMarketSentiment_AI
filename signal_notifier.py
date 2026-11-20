"""
signal_notifier.py
==================
Gold AI — Telegram notifier + forward PAPER TRADER for the single active rule
(signal_rules.ACTIVE_RULE_KEY): LONG "Macro Pullback — EMA Falling + Dollar
Falling", entered at the close of the signal session and held for exactly
HOLD_SESSIONS completed sessions (no stop), one trade at a time.

Every completed session is processed exactly once, in order:
  * no open trade + rule fires   → PAPER TRADE OPENED alert
  * open trade reaches its exit  → PAPER TRADE CLOSED alert (P&L, running record);
                                   if the exit session also fires, a new trade opens
  * anything else                → no message (no spam while a trade is running)
Missed runs are caught up session by session, so exits/entries are never skipped.

Usage:
  python signal_notifier.py --auto       refresh data, process new sessions (GitHub Actions)
  python signal_notifier.py --dry-run    same, but print messages and do NOT save state
  python signal_notifier.py --no-refresh skip the data refresh (combine with the above)
  python signal_notifier.py --status     send the current paper-trade status (not recorded)
  python signal_notifier.py --test       send a TEST copy of the status (not recorded)

Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TWELVEDATA_API_KEY from the
environment (GitHub secrets) or the gitignored config_local.py.

State: state/notifier_state.json (committed back by the workflow)
  last_processed : start time of the last session processed
  paper_trade    : the open trade (or null)
  closed_trades  : realised paper trades
  sent           : dedup keys of every message sent
"""

import copy
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
MAX_STATE_KEYS   = 400
MAX_CATCHUP      = 20      # sessions processed per run at most
COST_PCT         = 0.02    # round-trip paper cost, % (same as the backtest)
HOLD             = signal_rules.HOLD_SESSIONS
RULE             = signal_rules.RULES_BY_KEY[signal_rules.ACTIVE_RULE_KEY]
SIGNAL_TITLE     = "LONG (Macro Pullback — EMA Falling + Dollar Falling)"

# Latest completed session must have ended within this many hours, otherwise the
# data feed is considered stale (covers the weekend gap and short holidays).
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
#  STATE
# ─────────────────────────────────────────────────────────────────────────────
def load_state():
    s = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            s = json.load(f)
    s.setdefault("sent", {})
    s.setdefault("last_processed", None)
    s.setdefault("paper_trade", None)
    s.setdefault("closed_trades", [])
    return s


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    if len(state["sent"]) > MAX_STATE_KEYS:
        keep = sorted(state["sent"].items(), key=lambda kv: kv[1]["sent_at"])[-MAX_STATE_KEYS:]
        state["sent"] = dict(keep)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


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
#  DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_sessions():
    df = pd.read_csv(SESSION_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def entry_index(sess, trade):
    """Row of the entry session; if that row was removed by a data repair,
    the last session that started on/before the entry."""
    d = sess["date"].to_numpy()
    return int(max(0, (d <= np.datetime64(pd.Timestamp(trade["entry_session"]))).sum() - 1))


def skey(ts):
    return f"{pd.Timestamp(ts):%Y-%m-%d %H:%M}"


def trade_day(ts):
    """Session starting 18:00 NY on day D is trading day D+1."""
    return (pd.Timestamp(ts) + pd.Timedelta(hours=6)).date()


def fmt_day(ts):
    return f"{trade_day(ts):%a %d %b %Y}"


def est_exit_day(sess, i, sessions_left):
    """Estimate the trading day of the exit session from recent session spacing."""
    recent = sess["date"].iloc[max(0, i - 30):i + 1].diff().dropna()
    step = recent.median() if len(recent) else pd.Timedelta(days=1)
    return fmt_day(sess["date"].iloc[i] + step * sessions_left)


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


def rule_fires(recs, i):
    c = signal_rules.classify_at(recs, i)
    return c["direction"] == "LONG" and c["key_signal"] and c["key_signal"]["key"] == RULE["key"]


def record_txt(state):
    ct = state["closed_trades"]
    if not ct:
        return "  No closed paper trades yet."
    r = np.array([t["ret_pct"] for t in ct])
    eq = np.prod(1 + r / 100) - 1
    return (f"  {len(r)} closed | {int((r > 0).sum())}W / {int((r <= 0).sum())}L "
            f"({(r > 0).mean()*100:.0f}% win) | avg {r.mean():+.2f}% | compounded {eq*100:+.2f}%")


def open_trade_txt(trade, sess, i):
    if not trade:
        return "  None — flat, waiting for the next signal."
    e_idx = entry_index(sess, trade)
    done = i - e_idx
    last = float(sess["sess_close"].iloc[i])
    unreal = (last / trade["entry_close"] - 1) * 100
    return (f"  LONG from {fmt_day(trade['entry_session'])} @ ${trade['entry_close']:,.2f}\n"
            f"  Session {done}/{HOLD} done | last close ${last:,.2f} ({unreal:+.2f}% unrealised)\n"
            f"  Exit: close of session {HOLD} (≈ {est_exit_day(sess, i, HOLD - done)})")


BACKTEST_TXT = ("2020–26 backtest (10-session, no stop, 0.02% cost): 87 trades, 59% win, "
                "+1.20%/trade, PF 2.63, max DD −12.8%. Out-of-sample Jul-24→Nov-26: "
                "32 trades, 62.5% win vs 60.5% base.")


# ─────────────────────────────────────────────────────────────────────────────
#  MESSAGES
# ─────────────────────────────────────────────────────────────────────────────
def msg_open(sess, i, trade, state, catchup=False):
    row = sess.iloc[i]
    live = get_live_gold_price()
    live_txt = f"\n  Live spot now: ${live:,.2f}" if live and not catchup else ""
    return (
        f"🟢🔼 <b>GOLD AI — PAPER TRADE OPENED</b>{' (catch-up)' if catchup else ''}\n\n"
        f"📈 <b>Signal:</b> {SIGNAL_TITLE}\n"
        f"📅 <b>Session:</b> {fmt_day(row['date'])} (closed)\n"
        f"💵 <b>Entry reference (session close):</b> ${trade['entry_close']:,.2f}{live_txt}\n"
        f"⏳ <b>Hold window:</b> {HOLD} sessions → exit at the close of session {HOLD} "
        f"(≈ {est_exit_day(sess, i, HOLD)})\n"
        f"🛑 <b>Stop:</b> none (time exit only)\n\n"
        f"🔎 <b>Why:</b>\n"
        f"  EMA200 slope {row['ema_slope']:+.3f} (falling)\n"
        f"  Dollar (UUP) slope {row['dxy_slope']:+.4f} (falling)\n\n"
        f"📂 <b>Open paper trade:</b> this one (session 0/{HOLD}). "
        f"No new entries until it closes.\n\n"
        f"📊 <b>Paper record:</b>\n{record_txt(state)}\n\n"
        f"<i>{BACKTEST_TXT}</i>\n"
        f"<i>Gold AI Antigravity | XAU/USD | paper trading only</i>"
    )


def msg_close(sess, i, t, state, catchup=False):
    em = "✅" if t["ret_pct"] > 0 else "❌"
    return (
        f"{em} <b>GOLD AI — PAPER TRADE CLOSED</b>{' (catch-up)' if catchup else ''}\n\n"
        f"📈 <b>Signal:</b> {SIGNAL_TITLE}\n"
        f"📅 Entry: {fmt_day(t['entry_session'])} @ ${t['entry_close']:,.2f}\n"
        f"📅 Exit:  {fmt_day(t['exit_session'])} @ ${t['exit_close']:,.2f}  ({HOLD} sessions)\n"
        f"💰 <b>Result:</b> {t['ret_pct']:+.2f}% after {COST_PCT:.2f}% cost "
        f"({t['exit_close'] - t['entry_close']:+,.2f} pts)\n\n"
        f"📊 <b>Paper record:</b>\n{record_txt(state)}\n\n"
        f"📂 <b>Open paper trade:</b> none — watching for the next signal.\n\n"
        f"<i>Gold AI Antigravity | XAU/USD | paper trading only</i>"
    )


def msg_status(sess, state, test=False):
    i = len(sess) - 1
    row = sess.iloc[i]
    fires = rule_fires(sess.to_dict("records"), i)
    return (
        (f"🧪 <b>TEST MESSAGE</b>\n\n" if test else "")
        + f"📋 <b>GOLD AI — PAPER TRADING STATUS</b>\n\n"
        f"📈 <b>Strategy:</b> {SIGNAL_TITLE}, {HOLD}-session hold, no stop\n"
        f"📅 <b>Latest completed session:</b> {fmt_day(row['date'])} — close ${row['sess_close']:,.2f}\n"
        f"🔎 Rule on that session: {'FIRES' if fires else 'not firing'} "
        f"(EMA200 slope {row['ema_slope']:+.3f}, dollar slope {row['dxy_slope']:+.4f})\n\n"
        f"📂 <b>Open paper trade:</b>\n{open_trade_txt(state['paper_trade'], sess, i)}\n\n"
        f"📊 <b>Paper record:</b>\n{record_txt(state)}\n\n"
        f"<i>Gold AI Antigravity | XAU/USD | paper trading only</i>"
    )


def msg_armed(sess, state):
    return ("🚀 <b>GOLD AI — PAPER TRADING ARMED</b>\n\n"
            f"Only active rule: {SIGNAL_TITLE}\n"
            f"Hold: {HOLD} sessions, no stop, one trade at a time.\n"
            "You will only get a message when a paper trade OPENS or CLOSES "
            "(or if the data feed goes stale).\n\n"
            + msg_status(sess, state).split("\n\n", 1)[1])


def build_stale_msg(last_date, age_h):
    return (f"⚠️ <b>GOLD AI — DATA FEED STALE</b>\n\n"
            f"Latest completed session: {fmt_day(last_date)}\n"
            f"That is {age_h:.0f}h old — paper trading is paused until data updates.\n"
            f"Check the GitHub Actions log (TwelveData / refresh step).")


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER-TRADE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def process_session(sess, i, state, catchup):
    """Advance the paper trader by one completed session. Returns list of (key, message)."""
    recs = sess.to_dict("records")
    row = recs[i]
    out = []

    trade = state["paper_trade"]
    if trade:
        e_idx = entry_index(sess, trade)
        if i - e_idx >= HOLD:
            exit_close = float(row["sess_close"])
            t = dict(trade, exit_session=skey(row["date"]), exit_close=exit_close,
                     ret_pct=round((exit_close / trade["entry_close"] - 1) * 100 - COST_PCT, 4))
            state["closed_trades"].append(t)
            state["paper_trade"] = trade = None
            out.append((f"close:{t['entry_session']}", msg_close(sess, i, t, state, catchup)))

    if trade is None and rule_fires(recs, i):
        trade = {"entry_session": skey(row["date"]), "entry_close": float(row["sess_close"]),
                 "hold": HOLD, "rule": RULE["key"]}
        state["paper_trade"] = trade
        out.append((f"open:{trade['entry_session']}", msg_open(sess, i, trade, state, catchup)))
    return out


def run(refresh=True, dry_run=False, status=False, test=False):
    if refresh:
        import refresh_data
        try:
            refresh_data.refresh(fetch=True)
        except Exception as e:
            log(f"❌ data refresh failed: {e}")   # stale guard below will catch it

    sess  = load_sessions()
    last  = sess["date"].iloc[-1]
    state = load_state()

    ended = last + pd.Timedelta(hours=23)                       # 17:00 NY next day
    age_h = (pd.Timestamp.now(tz="America/New_York").tz_localize(None) - ended).total_seconds() / 3600
    log(f"latest completed session {last} (ended {age_h:.1f}h ago); "
        f"last processed {state['last_processed']}; open trade: {bool(state['paper_trade'])}")

    if status or test:
        return 0 if send_telegram(msg_status(sess, state, test=test), dry_run) else 1

    if age_h > STALE_AFTER_HOURS:
        key = f"stale:{skey(last)}"
        if key in state["sent"]:
            log(f"stale data; warning already sent ({key})")
            return 1
        if send_telegram(build_stale_msg(last, age_h), dry_run) and not dry_run:
            mark_sent(state, key)
            save_state(state)
        return 1

    # Which sessions are new?
    if state["last_processed"] is None:
        todo = [len(sess) - 1]                       # first run: start flat at the latest session
        first_run = True
    else:
        lp = pd.Timestamp(state["last_processed"])
        todo = list(sess.index[sess["date"] > lp])
        first_run = False
        if len(todo) > MAX_CATCHUP:
            log(f"⚠️ {len(todo)} unprocessed sessions; processing the oldest {MAX_CATCHUP} this run")
            todo = todo[:MAX_CATCHUP]
    if not todo:
        log("no new completed session — nothing to do")
        return 0

    for n, i in enumerate(todo):
        catchup = n < len(todo) - 1
        snapshot = copy.deepcopy(state)
        for key, msg in process_session(sess, i, state, catchup):
            if key in state["sent"]:
                log(f"already sent {key}")
                continue
            if not send_telegram(msg, dry_run):
                # roll the trade state back to before this session, but remember what
                # was already sent so the retry never duplicates a message
                log("send failed — session not advanced; will retry next run")
                snapshot["sent"] = state["sent"]
                if not dry_run:
                    save_state(snapshot)
                return 1
            mark_sent(state, key)
        state["last_processed"] = skey(sess["date"].iloc[i])
        log(f"processed session {state['last_processed']} → open trade: "
            f"{state['paper_trade']['entry_session'] if state['paper_trade'] else 'none'}")

    if first_run and "armed:v1" not in state["sent"]:
        if send_telegram(msg_armed(sess, state), dry_run):
            mark_sent(state, "armed:v1")

    if not dry_run:
        save_state(state)
    else:
        log("dry run — state NOT saved")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)
    sys.exit(run(refresh="--no-refresh" not in args,
                 dry_run="--dry-run" in args,
                 status="--status" in args,
                 test="--test" in args))
