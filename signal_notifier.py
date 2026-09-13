"""
signal_notifier.py
==================
Gold AI — Telegram Notification System

Sends:
  1. Morning alert at 08:00 IST (London open) with today's signal
  2. NY session alert at 14:00 IST
  3. End-of-Day report at 23:30 IST with full session P&L + rolling stats

Setup (2 minutes):
  Step 1: Open Telegram → search @BotFather → /newbot → follow steps → copy BOT_TOKEN
  Step 2: Open Telegram → search @userinfobot → /start → copy your id as CHAT_ID
  Step 3: Fill in BOT_TOKEN and CHAT_ID below, then run:
          python signal_notifier.py
"""

import os, time, json, requests
from datetime import datetime
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  ⚙️  CONFIG — PASTE YOUR VALUES HERE
# ─────────────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or "8701244069:AAHmboXb8HH59TdN6iXZ9NsR866lIwJzTs4"
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID") or "1336601213"

# Alert times (24-hour, IST)
MORNING_HOUR = 8;   MORNING_MIN = 0    # 08:00 IST London open alert
NY_HOUR      = 14;  NY_MIN      = 0    # 14:00 IST NY open alert
EOD_HOUR     = 23;  EOD_MIN     = 30   # 23:30 IST End of Day report

CHECK_EVERY  = 60  # seconds between loop checks
# ─────────────────────────────────────────────────────────────────────────────

ROOT         = os.path.dirname(os.path.abspath(__file__))
SESSION_FILE = os.path.join(ROOT, "data", "session_regime_v2.csv")
STATE_FILE   = os.path.join(ROOT, "logs", "notifier_state.json")
LOG_FILE     = os.path.join(ROOT, "logs", "notifier_log.txt")
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)

# ── Signal accuracy metadata (from OOS forward test) ─────────────────────────
SIGNAL_META = {
    "U4 Overextension":    dict(dir="LONG",  tier="STRONG",  oos=75.0),
    "U3 EMA Bearish":      dict(dir="LONG",  tier="STRONG",  oos=72.4),
    "U2 EMA Bearish":      dict(dir="LONG",  tier="STRONG",  oos=65.1),
    "U1 EMA Bearish":      dict(dir="LONG",  tier="STRONG",  oos=64.8),
    "L1 False Break":      dict(dir="LONG",  tier="STRONG",  oos=73.9),
    "U3 Rebound":          dict(dir="LONG",  tier="STRONG",  oos=71.4),
    "U1+VWAP+DXY Fall":    dict(dir="LONG",  tier="MODERATE",oos=66.7),
    "Pure Up + VWAP Bull": dict(dir="LONG",  tier="MODERATE",oos=64.8),
    "U2 + VWAP Bull":      dict(dir="LONG",  tier="MODERATE",oos=57.1),
    "U2 + EMA Bull":       dict(dir="LONG",  tier="MODERATE",oos=54.8),
    "L4 Overextension":    dict(dir="SHORT", tier="STRONG",  oos=76.5),
    "L3 EMA Bull":         dict(dir="SHORT", tier="STRONG",  oos=73.1),
    "L3 VWAP Bull":        dict(dir="SHORT", tier="STRONG",  oos=71.4),
    "L2 VWAP+EMA Bear":    dict(dir="SHORT", tier="STRONG",  oos=61.8),
    "BEST Short Macro":    dict(dir="SHORT", tier="STRONG",  oos=60.6),
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def send_telegram(text):
    """Send HTML-formatted message to Telegram. Returns True on success."""
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        log("⚠️  Not configured — would send:\n" + text[:200])
        return False
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        ok = r.status_code == 200
        log(f"{'✅' if ok else '❌'} Telegram {r.status_code}: {text[:50]}...")
        return ok
    except Exception as e:
        log(f"❌ Telegram error: {e}")
        return False


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"morning": None, "ny": None, "eod": None}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


def sf(val, d=0.0):
    try:
        v = float(val)
        return d if pd.isna(v) else v
    except:
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL ENGINE — mirrors server.py compute_signal exactly
# ─────────────────────────────────────────────────────────────────────────────
def compute_signal(row):
    mu  = sf(row.get("max_up",       0))
    md  = sf(row.get("max_down",     0))
    es  = sf(row.get("ema_slope",    0))
    vs  = sf(row.get("vwap_slope",   0))
    ds  = sf(row.get("dxy_slope",    0))
    ibd = int(sf(row.get("is_bidir",     0)))
    lfb = int(sf(row.get("l1_false_brk", 0)))
    flp = sf(row.get("roll_flip_rate", 0))

    def neutral(r):
        return {"direction": "NEUTRAL", "reason": r,
                "signals": [], "confidence": 0, "tier": "RANGING"}

    # Stay-out filters
    if flp >= 35:
        return neutral("Flip rate >= 35% — market bidirectional/choppy")
    if mu < 1 and md < 1:
        return neutral("No level broken — pure consolidation")
    if mu >= 1 and md >= 1 and mu < 2 and md < 2:
        return neutral("L1 + U1 same session — ranging/trapped")

    act = {}
    if mu >= 4 and es < -0.0005: act["U4 Overextension"]    = SIGNAL_META["U4 Overextension"]
    if mu >= 3 and es < -0.0005: act["U3 EMA Bearish"]       = SIGNAL_META["U3 EMA Bearish"]
    if mu >= 2 and es < -0.0005: act["U2 EMA Bearish"]       = SIGNAL_META["U2 EMA Bearish"]
    if mu >= 1 and es < -0.0005: act["U1 EMA Bearish"]       = SIGNAL_META["U1 EMA Bearish"]
    if mu >= 3 and ibd == 1:     act["U3 Rebound"]           = SIGNAL_META["U3 Rebound"]
    if lfb == 1 and mu >= 1:     act["L1 False Break"]       = SIGNAL_META["L1 False Break"]
    if mu >= 1 and md < 1 and vs > 0:  act["Pure Up + VWAP Bull"] = SIGNAL_META["Pure Up + VWAP Bull"]
    if mu >= 2 and vs > 0:             act["U2 + VWAP Bull"]      = SIGNAL_META["U2 + VWAP Bull"]
    if mu >= 2 and es > 0.0005:        act["U2 + EMA Bull"]       = SIGNAL_META["U2 + EMA Bull"]
    if mu >= 1 and vs > 0 and ds < 0:  act["U1+VWAP+DXY Fall"]   = SIGNAL_META["U1+VWAP+DXY Fall"]
    if md >= 4 and es > 0.0005:        act["L4 Overextension"]    = SIGNAL_META["L4 Overextension"]
    if md >= 3 and es > 0.0005:        act["L3 EMA Bull"]         = SIGNAL_META["L3 EMA Bull"]
    if md >= 3 and vs > 0:             act["L3 VWAP Bull"]        = SIGNAL_META["L3 VWAP Bull"]
    if md >= 2 and vs < 0 and es < -0.0005: act["L2 VWAP+EMA Bear"] = SIGNAL_META["L2 VWAP+EMA Bear"]
    if md >= 2 and es < -0.0005 and ds > 0: act["BEST Short Macro"]  = SIGNAL_META["BEST Short Macro"]

    longs  = {k: v for k, v in act.items() if v["dir"] == "LONG"}
    shorts = {k: v for k, v in act.items() if v["dir"] == "SHORT"}

    if not longs and not shorts:
        return neutral("No qualifying signal — wait for setup")
    if longs and shorts:
        return neutral("Conflicting LONG + SHORT signals")

    sigs, direction = (longs, "LONG") if longs else (shorts, "SHORT")
    conf = round(float(np.mean([v["oos"] for v in sigs.values()])), 1)
    tier = "STRONG" if any(v["tier"] == "STRONG" for v in sigs.values()) else "MODERATE"
    return {
        "direction":  direction,
        "confidence": conf,
        "tier":       tier,
        "reason":     f"{len(sigs)} signal(s) confirm {'bullish' if direction == 'LONG' else 'bearish'} window",
        "signals":    [{"name": k, "acc": v["oos"], "tier": v["tier"]} for k, v in sigs.items()],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MESSAGE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def get_live_gold_price():
    try:
        from config import API_KEY
        url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={API_KEY}"
        r = requests.get(url, timeout=5).json()
        if "price" in r:
            return float(r["price"])
    except Exception:
        pass
    return None


def build_signal_msg(signal, row, date_str, session="London"):
    d    = signal["direction"]
    conf = signal["confidence"]
    tier = signal["tier"]
    sigs = signal["signals"]
    mu   = sf(row.get("max_up",    0))
    md   = sf(row.get("max_down",  0))
    px   = sf(row.get("sess_open", 0))
    live_px = get_live_gold_price()
    px_str  = f"${live_px:,.2f} (Live Spot)" if live_px else f"~${px:,.1f}"

    if d == "LONG":
        top   = "🟢🔼"
        label = "LOOK FOR LONGS"
        tip   = ("  → Enter on pullbacks into L1–L2 zone\n"
                 "  → Target: U2–U3 levels above\n"
                 "  → Stop: below previous session low")
    else:
        top   = "🔴🔽"
        label = "LOOK FOR SHORTS"
        tip   = ("  → Enter on bounces into U1–U2 zone\n"
                 "  → Target: L2–L3 levels below\n"
                 "  → Stop: above previous session high")

    confluence = tier + ("+ CONFLUENCE" if len(sigs) > 1 else "")
    sigs_txt   = "\n".join(
        f"  • {s['name']}  ({s['acc']:.1f}% OOS accuracy)"
        for s in sorted(sigs, key=lambda x: -x["acc"])
    )

    return (
        f"{top} <b>GOLD AI SIGNAL — {label}</b> {top}\n\n"
        f"📅 <b>Date:</b> {date_str}  |  🕐 {session} Session\n"
        f"💪 <b>Strength:</b> {confluence}\n"
        f"🎯 <b>Confidence:</b> {conf:.1f}%\n\n"
        f"<b>Active Signals Firing:</b>\n{sigs_txt}\n\n"
        f"📊 <b>Session Data:</b>\n"
        f"  Max levels UP:   {mu:.0f}  |  Max levels DOWN: {md:.0f}\n"
        f"  Gold price:      {px_str}\n\n"
        f"⚡ <b>Trade Plan:</b>\n{tip}\n\n"
        f"<i>Gold AI Antigravity | XAU/USD Paper Trading</i>"
    )


def build_neutral_msg(signal, date_str, session="London"):
    return (
        f"⚪ <b>GOLD AI — STAY OUT</b>\n\n"
        f"📅 {date_str}  |  {session} Session\n"
        f"🚫 <b>No trade today</b>\n\n"
        f"<b>Reason:</b> {signal['reason']}\n\n"
        f"💡 <i>Sit on hands. Let the market show its hand first.</i>\n\n"
        f"<i>Gold AI Antigravity | XAU/USD</i>"
    )


def build_eod_report(sess_df, date_str):
    latest = sess_df.iloc[-1].to_dict()
    sig    = compute_signal(latest)
    d      = sig["direction"]
    conf   = sig.get("confidence", 0)
    tier   = sig.get("tier", "—")
    mu     = sf(latest.get("max_up",     0))
    md     = sf(latest.get("max_down",   0))
    o      = sf(latest.get("sess_open",  0))
    c      = sf(latest.get("sess_close", 0))
    dpct   = ((c - o) / o * 100) if o > 0 else 0
    dpts   = c - o

    if d == "LONG":
        de    = "🟢"; dl = "BULLISH — LOOK FOR LONGS"
        out   = f"Session closed {'🟢 in your favor' if dpct > 0 else '🔴 against'} ({dpct:+.2f}%)"
        peak  = f"Peak UP available today: {mu:.0f} ATR level(s) above open"
    elif d == "SHORT":
        de    = "🔴"; dl = "BEARISH — LOOK FOR SHORTS"
        out   = f"Session closed {'🟢 in your favor' if dpct < 0 else '🔴 against'} ({dpct:+.2f}%)"
        peak  = f"Peak DOWN available today: {md:.0f} ATR level(s) below open"
    else:
        de    = "⚪"; dl = "NEUTRAL — NO TRADE"
        out   = f"Correctly stayed out. Session moved {dpct:+.2f}% (noise)"
        peak  = f"Range: {abs(dpct):.2f}% — choppy, filter worked"

    # 5-day recap table
    recap = ""
    for _, r in sess_df.tail(5).iterrows():
        rd   = r["date"].strftime("%b %d")
        ro   = sf(r.get("sess_open",  0))
        rc   = sf(r.get("sess_close", 0))
        rpct = ((rc - ro) / ro * 100) if ro > 0 else 0
        rs   = compute_signal(r.to_dict())
        em   = "🟢" if rs["direction"]=="LONG" else "🔴" if rs["direction"]=="SHORT" else "⚪"
        recap += f"  {em} {rd}:  {rpct:+.2f}%  [{rs['direction']}]\n"

    # Rolling 10-session stats
    last10 = sess_df.tail(10)
    roll   = [compute_signal(r.to_dict()) for _, r in last10.iterrows()]
    nl = sum(1 for s in roll if s["direction"] == "LONG")
    ns = sum(1 for s in roll if s["direction"] == "SHORT")
    nn = sum(1 for s in roll if s["direction"] == "NEUTRAL")

    lr, sr = [], []
    for i, (_, r) in enumerate(last10.iterrows()):
        ro = sf(r.get("sess_open",0)); rc = sf(r.get("sess_close",0))
        rp = ((rc - ro) / ro * 100) if ro > 0 else 0
        if roll[i]["direction"] == "LONG":  lr.append(rp)
        elif roll[i]["direction"] == "SHORT": sr.append(-rp)

    al = f"{np.mean(lr):+.2f}%" if lr else "—"
    as_ = f"{np.mean(sr):+.2f}%" if sr else "—"

    now_s = datetime.now().strftime("%H:%M IST")

    return (
        f"📊 <b>GOLD AI — END OF DAY REPORT</b>\n\n"
        f"📅 <b>{date_str}</b>  |  🕐 {now_s}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{de} <b>TODAY'S SIGNAL:</b> {dl}\n"
        f"   💪 {tier}  |  🎯 {conf:.1f}% confidence\n\n"
        f"💰 <b>XAU/USD Session Result:</b>\n"
        f"  Open:  ${o:,.2f}  →  Close: ${c:,.2f}\n"
        f"  Day:   {dpct:+.2f}%  ({dpts:+.1f} pts)\n"
        f"  {peak}\n\n"
        f"📌 <b>Outcome:</b> {out}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>Last 5 Sessions:</b>\n{recap}"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>Rolling 10-Session Stats:</b>\n"
        f"  🟢 LONG days:    {nl}  (avg: {al})\n"
        f"  🔴 SHORT days:   {ns}  (avg: {as_})\n"
        f"  ⚪ NEUTRAL days: {nn}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Tomorrow prep:</b>\n"
        f"  → Check dashboard for fresh level zones\n"
        f"  → Morning alert at 08:00 IST\n\n"
        f"<i>Gold AI Antigravity | XAU/USD Paper Trading</i>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log("=" * 55)
    log("  Gold AI Telegram Notifier — Online")
    log(f"  Watching: {SESSION_FILE}")
    log("=" * 55)

    send_telegram(
        "🚀 <b>Gold AI Notifier — Online</b>\n\n"
        "✅ Watching XAU/USD signals live.\n\n"
        "📩 <b>You will receive:</b>\n"
        "  🕗 08:00 IST — London session signal\n"
        "  🕑 14:00 IST — NY session reminder\n"
        "  🕛 23:30 IST — End-of-Day report\n\n"
        "<i>Gold AI Antigravity | XAU/USD</i>"
    )

    state = load_state()

    while True:
        now   = datetime.now()
        today = now.date().isoformat()
        h, mn = now.hour, now.minute

        try:
            sess_df = pd.read_csv(SESSION_FILE)
            sess_df["date"] = pd.to_datetime(sess_df["date"])
            sess_df = sess_df.sort_values("date").reset_index(drop=True)
            latest  = sess_df.iloc[-1]
            sig     = compute_signal(latest.to_dict())
            date_s  = latest["date"].date().isoformat()

            # Morning — 08:00 IST
            if h == MORNING_HOUR and mn < 5 and state.get("morning") != today:
                is_dir = sig["direction"] in ("LONG", "SHORT")
                msg = (build_signal_msg(sig, latest.to_dict(), date_s, "London")
                       if is_dir else build_neutral_msg(sig, date_s, "London"))
                if send_telegram(msg):
                    state["morning"] = today
                    save_state(state)

            # NY session — 14:00 IST
            elif h == NY_HOUR and mn < 5 and state.get("ny") != today:
                if sig["direction"] in ("LONG", "SHORT"):
                    em  = "🟢" if sig["direction"] == "LONG" else "🔴"
                    msg = (f"{em} <b>Gold AI — NY Session Opens</b>\n\n"
                           f"📅 {date_s}  |  🕑 NY opens now (14:30 IST)\n"
                           f"Signal still: <b>{sig['direction']}</b> — {sig['confidence']:.1f}% confidence\n"
                           f"💡 Watch for entries as NY volume kicks in.\n\n"
                           f"<i>Gold AI Antigravity</i>")
                else:
                    msg = (f"⚪ <b>Gold AI — NY Session</b>\n\n"
                           f"📅 {date_s}  |  🕑 NY opens now\n"
                           f"🚫 No signal active.\n{sig['reason']}\n\n"
                           f"<i>Gold AI Antigravity</i>")
                if send_telegram(msg):
                    state["ny"] = today
                    save_state(state)

            # EOD report — 23:30 IST
            elif h == EOD_HOUR and mn < 5 and state.get("eod") != today:
                msg = build_eod_report(sess_df, date_s)
                if send_telegram(msg):
                    state["eod"] = today
                    save_state(state)

        except Exception as e:
            log(f"❌ Error: {e}")

        time.sleep(CHECK_EVERY)


def send_london_alert():
    sess_df = pd.read_csv(SESSION_FILE)
    sess_df["date"] = pd.to_datetime(sess_df["date"])
    sess_df = sess_df.sort_values("date").reset_index(drop=True)
    latest  = sess_df.iloc[-1]
    sig     = compute_signal(latest.to_dict())
    date_s  = latest["date"].date().isoformat()
    is_dir  = sig["direction"] in ("LONG", "SHORT")
    msg     = (build_signal_msg(sig, latest.to_dict(), date_s, "London")
               if is_dir else build_neutral_msg(sig, date_s, "London"))
    log("Sending London session alert...")
    return send_telegram(msg)


def send_ny_alert():
    sess_df = pd.read_csv(SESSION_FILE)
    sess_df["date"] = pd.to_datetime(sess_df["date"])
    sess_df = sess_df.sort_values("date").reset_index(drop=True)
    latest  = sess_df.iloc[-1]
    sig     = compute_signal(latest.to_dict())
    date_s  = latest["date"].date().isoformat()
    if sig["direction"] in ("LONG", "SHORT"):
        em      = "🟢" if sig["direction"] == "LONG" else "🔴"
        live_p  = get_live_gold_price()
        px_line = f"  Gold price: <b>${live_p:,.2f}</b>\n" if live_p else ""
        msg = (f"{em} <b>Gold AI — NY Session Opens</b>\n\n"
               f"📅 {date_s}  |  🕑 NY opens now (14:00 IST)\n"
               f"Signal: <b>{sig['direction']}</b> — {sig['confidence']:.1f}% confidence\n"
               f"{px_line}"
               f"💡 Watch for entries as NY volume kicks in.\n\n"
               f"<i>Gold AI Antigravity</i>")
    else:
        msg = (f"⚪ <b>Gold AI — NY Session</b>\n\n"
               f"📅 {date_s}  |  🕑 NY opens now\n"
               f"🚫 No signal active.\n{sig['reason']}\n\n"
               f"<i>Gold AI Antigravity</i>")
    log("Sending NY session alert...")
    return send_telegram(msg)


def send_eod_alert():
    sess_df = pd.read_csv(SESSION_FILE)
    sess_df["date"] = pd.to_datetime(sess_df["date"])
    sess_df = sess_df.sort_values("date").reset_index(drop=True)
    latest  = sess_df.iloc[-1]
    date_s  = latest["date"].date().isoformat()
    msg     = build_eod_report(sess_df, date_s)
    log("Sending EOD report...")
    return send_telegram(msg)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if "--london" in args or "--morning" in args:
        send_london_alert()
    elif "--ny" in args:
        send_ny_alert()
    elif "--eod" in args:
        send_eod_alert()
    elif "--test" in args:
        print("Running instant test...")
        send_london_alert()
        send_eod_alert()
        print("Done!")
    elif "--auto" in args:
        import datetime as dt
        try:
            from datetime import timezone
            now_ist = dt.datetime.now(timezone.utc) + dt.timedelta(hours=5, minutes=30)
        except Exception:
            now_ist = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
        h = now_ist.hour
        print(f"Auto mode: IST time is {h}:{now_ist.minute:02d}")
        if h < 12:
            send_london_alert()
        elif 12 <= h < 18:
            send_ny_alert()
        else:
            send_eod_alert()
    else:
        main()


