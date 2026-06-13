"""
dashboard/server.py — Gold AI Antigravity Dashboard Backend
============================================================
Run: python dashboard/server.py
Open: http://localhost:5050

Login: username=dakshit  password=gold2025
"""

import os, json, time, requests
from datetime import datetime, timedelta
from functools import wraps
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
from flask import (Flask, render_template, request, session,
                   redirect, url_for, jsonify)

app = Flask(__name__)
app.secret_key = "goldai_antigravity_7x9k2025"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USERS = {"dakshit": "gold2025"}

# ─── Signal weights (from OOS forward test) ───────────────────────────────
SIGNAL_META = {
    # LONG signals
    "U4 Overextension":       dict(dir="LONG",  tier="STRONG",  oos=84.6, desc="U4 broke but EMA bearish — extreme overshoot, snap back UP"),
    "U3 EMA Bearish":         dict(dir="LONG",  tier="STRONG",  oos=77.3, desc="U3 broke, EMA still bearish — overextension LONG"),
    "U2 EMA Bearish":         dict(dir="LONG",  tier="STRONG",  oos=75.0, desc="U2 broke but EMA bearish — bull exhaustion signal"),
    "U1 EMA Bearish":         dict(dir="LONG",  tier="STRONG",  oos=78.6, desc="U1 broke, EMA bearish — overextension entry zone"),
    "L1 False Break":         dict(dir="LONG",  tier="STRONG",  oos=78.3, desc="Hit L1 AND U1 same session — false break, look LONG"),
    "U3 Rebound":             dict(dir="LONG",  tier="STRONG",  oos=70.0, desc="Hit U3 then reversed in-session — LONG reversal setup"),
    "U1 + VWAP + DXY Fall":   dict(dir="LONG",  tier="MODERATE",oos=69.8, desc="U1 + VWAP bullish + DXY falling (weak dollar)"),
    "Pure Up + VWAP Bull":    dict(dir="LONG",  tier="MODERATE",oos=68.5, desc="Pure upside session, no L1 touch, VWAP bullish"),
    "U2 + EMA Bull":          dict(dir="LONG",  tier="MODERATE",oos=55.4, desc="Broke U2 + EMA bullish — trend continuation"),
    "U2 + VWAP Bull":         dict(dir="LONG",  tier="MODERATE",oos=57.8, desc="Broke U2 + VWAP bullish — moderate LONG setup"),
    # SHORT signals
    "L4 Overextension":       dict(dir="SHORT", tier="STRONG",  oos=72.7, desc="L4 broke with bullish EMA — extreme overshoot DOWN"),
    "L3 EMA Bull":            dict(dir="SHORT", tier="STRONG",  oos=56.2, desc="L3 broke, EMA still bullish — overextension SHORT"),
    "BEST Short Macro":       dict(dir="SHORT", tier="STRONG",  oos=50.0, desc="L2+ + EMA bearish + DXY rising — full macro SHORT"),
}

_news_cache = {"data": [], "ts": 0}


# ─── Helpers ──────────────────────────────────────────────────────────────
def require_login(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def load_sessions():
    path = os.path.join(ROOT, "data", "session_regime_v2.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _safe_float(val, default=0.0):
    """NaN-safe float conversion. float('nan') or 0 returns nan (nan is truthy),
    so we must explicitly check with pd.isna()."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def compute_signal(row):
    """Apply all 26 conditions to one session row. Returns signal dict."""
    mu  = _safe_float(row.get("max_up",  0))
    md  = _safe_float(row.get("max_down", 0))
    es  = _safe_float(row.get("ema_slope", 0))
    vs  = _safe_float(row.get("vwap_slope", 0))
    ds  = _safe_float(row.get("dxy_slope", 0))
    ibd = int(_safe_float(row.get("is_bidir", 0)))
    lfb = int(_safe_float(row.get("l1_false_brk", 0)))
    flp = _safe_float(row.get("roll_flip_rate", 0))

    # ── STAY OUT checks ───────────────────────────────────────────────────
    if flp >= 35:
        return _neutral("Flip rate ≥35% — market is choppy, no edge")
    if mu < 1 and md < 1:
        return _neutral("No level broken — price consolidating inside levels")
    if mu >= 1 and md >= 1 and mu < 2 and md < 2:
        return _neutral("Hit L1 AND U1 same session — ranging / trapped")

    active = {}

    # ── LONG conditions ───────────────────────────────────────────────────
    if mu >= 4 and es < -0.0005:
        active["U4 Overextension"] = SIGNAL_META["U4 Overextension"]
    if mu >= 3 and es < -0.0005:
        active["U3 EMA Bearish"]   = SIGNAL_META["U3 EMA Bearish"]
    if mu >= 2 and es < -0.0005:
        active["U2 EMA Bearish"]   = SIGNAL_META["U2 EMA Bearish"]
    if mu >= 1 and es < -0.0005:
        active["U1 EMA Bearish"]   = SIGNAL_META["U1 EMA Bearish"]
    if mu >= 3 and ibd == 1:
        active["U3 Rebound"]       = SIGNAL_META["U3 Rebound"]
    if lfb == 1 and mu >= 1:
        active["L1 False Break"]   = SIGNAL_META["L1 False Break"]
    if mu >= 1 and md < 1 and vs > 0:
        active["Pure Up + VWAP Bull"] = SIGNAL_META["Pure Up + VWAP Bull"]
    if mu >= 2 and vs > 0:
        active["U2 + VWAP Bull"]   = SIGNAL_META["U2 + VWAP Bull"]
    if mu >= 2 and es > 0.0005:
        active["U2 + EMA Bull"]    = SIGNAL_META["U2 + EMA Bull"]
    if mu >= 1 and vs > 0 and ds < 0:
        active["U1 + VWAP + DXY Fall"] = SIGNAL_META["U1 + VWAP + DXY Fall"]

    # ── SHORT conditions ──────────────────────────────────────────────────
    if md >= 4 and es > 0.0005:
        active["L4 Overextension"] = SIGNAL_META["L4 Overextension"]
    if md >= 3 and es > 0.0005:
        active["L3 EMA Bull"]      = SIGNAL_META["L3 EMA Bull"]
    if md >= 2 and es < -0.0005 and ds > 0:
        active["BEST Short Macro"] = SIGNAL_META["BEST Short Macro"]

    long_sigs  = {k:v for k,v in active.items() if v["dir"]=="LONG"}
    short_sigs = {k:v for k,v in active.items() if v["dir"]=="SHORT"}

    if not long_sigs and not short_sigs:
        return _neutral("No qualifying signal — wait for setup")

    long_conf  = np.mean([v["oos"] for v in long_sigs.values()])  if long_sigs  else 0
    short_conf = np.mean([v["oos"] for v in short_sigs.values()]) if short_sigs else 0

    # ── CONFLICTED: both directions active ────────────────────────────────
    # When is_bidir=1 AND both LONG+SHORT conditions fire, the session hit
    # upper AND lower levels — too chaotic, treat as NEUTRAL
    if long_sigs and short_sigs:
        return _neutral(f"Conflicting signals (L:{long_conf:.0f}% vs S:{short_conf:.0f}%) — CONFLICTED")

    if long_sigs and not short_sigs:
        return _directional("LONG", long_sigs, long_conf)
    elif short_sigs and not long_sigs:
        return _directional("SHORT", short_sigs, short_conf)
    else:
        return _neutral(f"Conflicting signals (L:{long_conf:.0f}% vs S:{short_conf:.0f}%)")


def _neutral(reason):
    return {"direction":"NEUTRAL","confidence":0,"tier":"RANGING",
            "reason":reason,"signals":[],"key_signal":None}


def _directional(direction, sigs, conf):
    strong = {k:v for k,v in sigs.items() if v["tier"]=="STRONG"}
    key    = max(sigs, key=lambda k: sigs[k]["oos"])
    tier   = "STRONG" if strong else "MODERATE"
    n      = len(sigs)
    reason = f"{n} signal{'s' if n>1 else ''} confirm {'bullish' if direction=='LONG' else 'bearish'} window"
    return {
        "direction":  direction,
        "confidence": round(float(conf), 1),
        "tier":       tier,
        "reason":     reason,
        "signals":    [{"name":k,"acc":v["oos"],"tier":v["tier"],"desc":v["desc"]}
                       for k,v in sigs.items()],
        "key_signal": {"name":key,"desc":sigs[key]["desc"],"acc":sigs[key]["oos"]}
    }


def fetch_gold_news():
    """Fetch gold news from Kitco RSS. Cache 30 min."""
    global _news_cache
    if time.time() - _news_cache["ts"] < 1800 and _news_cache["data"]:
        return _news_cache["data"]
    articles = []
    feeds = [
        "https://www.kitco.com/rss/news.xml",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC%3DF&region=US&lang=en-US",
    ]
    for url in feeds:
        try:
            r = requests.get(url, timeout=5,
                             headers={"User-Agent":"Mozilla/5.0"})
            root = ET.fromstring(r.content)
            ns = {"atom":"http://www.w3.org/2005/Atom"}
            items = root.findall(".//item")
            for item in items[:8]:
                title = (item.findtext("title") or "").strip()
                link  = (item.findtext("link")  or "").strip()
                pub   = (item.findtext("pubDate") or "").strip()
                desc  = (item.findtext("description") or "").strip()[:200]
                if title and link:
                    articles.append({"title":title,"link":link,
                                     "pub":pub,"desc":desc})
            if articles:
                break
        except Exception:
            pass
    if not articles:
        articles = [
            {"title":"Gold holds near all-time highs amid US tariff uncertainty",
             "link":"#","pub":"Today","desc":"Gold prices remain elevated as investors continue to seek safe-haven assets."},
            {"title":"DXY weakness supports bullion buyers",
             "link":"#","pub":"Today","desc":"The US Dollar Index continues its decline, providing tailwinds for gold."},
            {"title":"Fed minutes: rate cuts still on the table",
             "link":"#","pub":"Today","desc":"Federal Reserve minutes suggest rate cuts remain possible, supporting gold."},
            {"title":"Central banks add to gold reserves in Q1 2025",
             "link":"#","pub":"Today","desc":"Global central banks continue to diversify into gold at record pace."},
            {"title":"Geopolitical tensions drive safe-haven demand",
             "link":"#","pub":"Today","desc":"Ongoing global uncertainty keeps institutional demand for gold elevated."},
        ]
    _news_cache = {"data": articles[:10], "ts": time.time()}
    return articles[:10]


def get_live_gold_price():
    """Fetch current gold futures price from Yahoo Finance."""
    try:
        url  = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1d&range=1d"
        r    = requests.get(url, timeout=4,
                             headers={"User-Agent":"Mozilla/5.0"})
        data = r.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        prev  = data["chart"]["result"][0]["meta"]["chartPreviousClose"]
        chg   = price - prev
        chgp  = chg / prev * 100
        return {"price": round(price, 2), "change": round(chg, 2),
                "change_pct": round(chgp, 3), "symbol": "XAU/USD"}
    except Exception:
        return {"price": "—", "change": 0, "change_pct": 0, "symbol": "XAU/USD"}


def load_performance():
    """Load OOS forward test stats."""
    try:
        path = os.path.join(ROOT, "logs", "forward_test", "session_signals.csv")
        sigs = pd.read_csv(path)
        long_acc  = None; short_acc = None
        sess_path = os.path.join(ROOT, "data", "session_regime_v2.csv")
        sess = pd.read_csv(sess_path)
        sess["date_only"] = pd.to_datetime(sess["date"]).dt.date.astype(str)
        sess = sess.sort_values("date").reset_index(drop=True)
        N = len(sess)
        date_idx = {d:i for i,d in enumerate(sess["date_only"])}
        mfes_l, finals_l, mfes_s, finals_s = [], [], [], []
        for _, row in sigs.iterrows():
            d = str(row["date"])
            if d not in date_idx: continue
            idx = date_idx[d]
            if idx + 10 >= N: continue
            entry = sess.iloc[idx]["sess_open"]
            if pd.isna(entry) or entry <= 0: continue
            closes = [sess.iloc[idx+j]["sess_close"] for j in range(1,11)
                      if not pd.isna(sess.iloc[idx+j]["sess_close"])]
            if not closes: continue
            pcts_l = [(c-entry)/entry*100 for c in closes]
            pcts_s = [-(c-entry)/entry*100 for c in closes]
            if row["direction"] == "LONG":
                mfes_l.append(max(pcts_l)); finals_l.append(pcts_l[-1])
            elif row["direction"] == "SHORT":
                mfes_s.append(max(pcts_s)); finals_s.append(pcts_s[-1])
        total   = len(sigs)
        long_n  = len(mfes_l); short_n = len(mfes_s)
        neutral = total - long_n - short_n
        return {
            "total_sessions": total,
            "long_signals":   long_n,
            "short_signals":  short_n,
            "neutral":        neutral,
            "long_acc":       round((sum(1 for f in finals_l if f>0)/long_n*100) if long_n else 0, 1),
            "short_acc":      round((sum(1 for f in finals_s if f>0)/short_n*100) if short_n else 0, 1),
            "long_avg_gain":  round(np.mean(finals_l) if finals_l else 0, 3),
            "short_avg_gain": round(np.mean(finals_s) if finals_s else 0, 3),
            "long_mfe":       round(np.mean(mfes_l) if mfes_l else 0, 3),
            "short_mfe":      round(np.mean(mfes_s) if mfes_s else 0, 3),
            "signals_month":  round(total / 74, 1),  # dataset: Jan2020→Mar2026 ≈ 74 months
        }
    except Exception as e:
        return {"error": str(e)}


# ─── Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("landing.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","").strip()
        if USERS.get(u) == p:
            session["user"] = u
            return redirect(url_for("dashboard"))
        error = "Invalid credentials. Try again."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@require_login
def dashboard():
    return render_template("dashboard.html", user=session["user"])


# ─── API endpoints ────────────────────────────────────────────────────────
@app.route("/api/signal")
@require_login
def api_signal():
    try:
        df  = load_sessions()
        row = df.iloc[-1]
        sig = compute_signal(row)
        sig["date"] = str(row["date"].date())
        sig["levels"] = {
            "max_up":   float(row.get("max_up", 0) or 0),
            "max_down": float(row.get("max_down", 0) or 0),
            "ema_slope": float(row.get("ema_slope", 0) or 0),
            "vwap_slope": float(row.get("vwap_slope", 0) or 0),
            "dxy_slope": float(row.get("dxy_slope", 0) or 0),
        }
        return jsonify({"ok": True, "signal": sig})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/news")
@require_login
def api_news():
    articles = fetch_gold_news()
    return jsonify({"ok": True, "articles": articles})


@app.route("/api/price")
@require_login
def api_price():
    price_data = get_live_gold_price()
    return jsonify({"ok": True, **price_data})


@app.route("/api/performance")
@require_login
def api_performance():
    perf = load_performance()
    return jsonify({"ok": True, **perf})


@app.route("/api/history")
@require_login
def api_history():
    try:
        path = os.path.join(ROOT, "logs", "forward_test", "session_signals.csv")
        df   = pd.read_csv(path)
        df   = df.tail(60).fillna("")
        rows = df.to_dict("records")
        return jsonify({"ok": True, "history": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})



@app.route("/api/levels")
@require_login
def api_levels():
    """Compute U1-U8 and L1-L8 actual price levels for chart overlay."""
    try:
        df  = load_sessions()
        row = df.iloc[-1]
        ref_price = float(row.get("sess_open", 0) or 0)
        atr       = float(row.get("atr", 30) or 30)
        max_up    = int(float(row.get("max_up", 0) or 0))
        max_down  = int(float(row.get("max_down", 0) or 0))
        # Level spacing: ~0.5x ATR per level
        step = max(atr * 0.5, 15.0)
        upper_levels = []
        for i in range(1, 9):
            price = round(ref_price + i * step, 2)
            broken = i <= max_up
            upper_levels.append({"label": f"U{i}", "price": price, "broken": broken,
                "color": "#00E676" if broken else "rgba(0,230,118,0.3)",
                "width": 2 if broken else 1, "style": 0 if broken else 2})
        lower_levels = []
        for i in range(1, 9):
            price = round(ref_price - i * step, 2)
            broken = i <= max_down
            lower_levels.append({"label": f"L{i}", "price": price, "broken": broken,
                "color": "#FF4C4C" if broken else "rgba(255,76,76,0.3)",
                "width": 2 if broken else 1, "style": 0 if broken else 2})
        date_val = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10]
        return jsonify({"ok": True, "ref_price": ref_price, "atr": atr, "step": step,
                        "max_up": max_up, "max_down": max_down,
                        "upper": upper_levels, "lower": lower_levels, "date": date_val})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/candles")
@require_login
def api_candles():
    """Proxy Yahoo Finance 15-min Gold Futures candles (free, no API key)."""
    import urllib.request, json as _json
    interval = request.args.get("interval", "15m")
    period   = request.args.get("period", "1mo")
    try:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF"
               f"?interval={interval}&range={period}&includePrePost=false")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = _json.loads(resp.read().decode())
        result = raw["chart"]["result"][0]
        timestamps = result["timestamp"]
        ohlcv = result["indicators"]["quote"][0]
        candles = []
        for i, ts in enumerate(timestamps):
            o = ohlcv["open"][i]; h = ohlcv["high"][i]
            l = ohlcv["low"][i];  c = ohlcv["close"][i]
            v = (ohlcv.get("volume") or [None]*len(timestamps))[i]
            if o is None or c is None: continue
            candles.append({"time": int(ts), "open": round(float(o),2),
                            "high": round(float(h),2), "low": round(float(l),2),
                            "close": round(float(c),2), "volume": int(v) if v else 0})
        return jsonify({"ok": True, "candles": candles, "interval": interval})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "candles": []})


if __name__ == "__main__":
    print("=" * 55)
    print("  GOLD AI ANTIGRAVITY — Dashboard Server")
    print("  http://localhost:5050")
    print("  Login: dakshit / gold2025")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5050)
