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

# ─── Signal rules: shared with the notifier and forward test ────────────────
import sys
sys.path.insert(0, ROOT)
import signal_rules


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


def compute_signal(df, i=None):
    """Label session i (default: latest) with the shared rules in signal_rules.py."""
    recs = df.to_dict("records")
    return signal_rules.classify_at(recs, len(recs) - 1 if i is None else i)


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
        sig = compute_signal(df)
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
