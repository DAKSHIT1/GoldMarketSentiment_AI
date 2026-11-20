"""
session_builder.py
==================
Session-level feature construction shared by the batch pipeline
(regime_rules_v2.py) and the live refresher (refresh_data.py).

Moving these functions here means the live notifier builds sessions with
EXACTLY the same code as the backtest data — no drift.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  REGIME v2 THRESHOLDS (derived from short_regime_analysis.py findings)
# ─────────────────────────────────────────────────────────────────────────────
LONG_PU2_MIN      = 35.0
LONG_AVG_UP_RATIO = 1.0

ST_MIN_DOWN_LEVEL = 1.5
ST_EMA_MAX        = -0.0005
ST_PL2_MIN        = 20.0
ST_L1_FREQ_MIN    = 30.0

SE_MIN_DOWN_LEVEL = 3.0
SE_RECENT_WINDOW  = 5
SE_PL3_MIN        = 15.0

FLIP_RATE_MAX     = 25.0
NO_BREAK_MAX      = 60.0
FALSE_BREAK_GATE  = True
RETEST_CONT_BONUS = True

ROLL = 10

ROLL_COLS = ["roll_p_u2_u1", "roll_p_l2_l1", "roll_p_l3_l1",
             "roll_avg_max_up", "roll_avg_max_down",
             "roll_flip_rate", "roll_no_break_pct",
             "roll_avg_ema", "roll_avg_dxy",
             "roll_l3plus_pct", "roll_l1_freq"]


def tag_sessions(bars, start_id=0):
    """Session id increments at every 18:00 New York bar (same as regime_features.py)."""
    start = (bars["datetime"].dt.hour == 18) & (bars["datetime"].dt.minute == 0)
    bars = bars.copy()
    bars["session_id"] = start.cumsum() + start_id
    return bars


def build_session_db_v2(bars_df):
    records = []
    for sid, grp in bars_df.groupby("session_id"):
        grp = grp.reset_index(drop=True)
        max_up   = float(grp["seq_max_levels_up"].max())   if "seq_max_levels_up"   in grp.columns else 0.0
        max_down = float(grp["seq_max_levels_down"].max()) if "seq_max_levels_down" in grp.columns else 0.0
        first_up = first_dn = None
        for i in range(len(grp)):
            if first_up is None and grp["seq_levels_broken_up"].iloc[i]   >= 1: first_up = i
            if first_dn is None and grp["seq_levels_broken_down"].iloc[i] >= 1: first_dn = i
            if first_up is not None and first_dn is not None: break
        if   first_up is not None and first_dn is not None:
            dir_first = "up" if first_up < first_dn else "down"
        elif first_up is not None: dir_first = "up"
        elif first_dn is not None: dir_first = "down"
        else:                      dir_first = "neither"

        ema_slope  = float(grp["ema50_slope"].iloc[0])    if "ema50_slope"          in grp.columns else 0.0
        dxy_slope  = float(grp["dxy_ema_slope"].iloc[0])  if "dxy_ema_slope"        in grp.columns else 0.0
        macro_str  = float(grp["macro_trend_strength"].iloc[0]) if "macro_trend_strength" in grp.columns else 0.0
        vwap_slope = float(grp["vwap_slope"].iloc[0])     if "vwap_slope"           in grp.columns else 0.0
        atr        = float(grp["atr"].mean())              if "atr"                  in grp.columns else 5.0

        records.append({
            "sid"       : sid,
            "date"      : grp["datetime"].iloc[0],
            "max_up"    : max_up,
            "max_down"  : max_down,
            "dir_first" : dir_first,
            "sess_open" : float(grp["open"].iloc[0]),
            "sess_close": float(grp["close"].iloc[-1]),
            "ema_slope" : ema_slope,
            "dxy_slope" : dxy_slope,
            "macro_str" : macro_str,
            "vwap_slope": vwap_slope,
            "atr"       : atr,
            "broke_u1"  : int(max_up   >= 1),
            "broke_u2"  : int(max_up   >= 2),
            "broke_u3"  : int(max_up   >= 3),
            "broke_l1"  : int(max_down >= 1),
            "broke_l2"  : int(max_down >= 2),
            "broke_l3"  : int(max_down >= 3),
            "is_bidir"  : int(max_up >= 1 and max_down >= 1),
            "is_no_brk" : int(max_up < 1 and max_down < 1),
            "is_l1_only": int(max_down >= 1 and max_down < 2),
            "l1_false_brk": int(max_down >= 1 and max_down < 2 and max_up >= 1),
        })
    return pd.DataFrame(records)


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def add_rolling_features(sess_v2, start_idx=0):
    """Rolling 10-session features. Only rows >= start_idx are (re)computed."""
    for c in ROLL_COLS:
        if c not in sess_v2.columns:
            sess_v2[c] = 0.0
    for i in range(start_idx, len(sess_v2)):
        w = sess_v2.iloc[max(0, i-ROLL):i]
        n_w = len(w)
        if n_w == 0:
            for c in ROLL_COLS:
                sess_v2.loc[i, c] = 0.0
            continue

        n_u1 = max(w["broke_u1"].sum(), 1)
        n_l1 = max(w["broke_l1"].sum(), 1)
        sess_v2.loc[i, "roll_p_u2_u1"]      = safe_div(w["broke_u2"].sum(), n_u1) * 100
        sess_v2.loc[i, "roll_p_l2_l1"]      = safe_div(w["broke_l2"].sum(), n_l1) * 100
        sess_v2.loc[i, "roll_p_l3_l1"]      = safe_div(w["broke_l3"].sum(), n_l1) * 100
        sess_v2.loc[i, "roll_avg_max_up"]   = w["max_up"].mean()
        sess_v2.loc[i, "roll_avg_max_down"] = w["max_down"].mean()
        sess_v2.loc[i, "roll_flip_rate"]    = w["is_bidir"].mean() * 100
        sess_v2.loc[i, "roll_no_break_pct"] = w["is_no_brk"].mean() * 100
        sess_v2.loc[i, "roll_avg_ema"]      = w["ema_slope"].mean()
        sess_v2.loc[i, "roll_avg_dxy"]      = w["dxy_slope"].mean()
        sess_v2.loc[i, "roll_l3plus_pct"]   = (w["max_down"] >= 3).mean() * 100
        sess_v2.loc[i, "roll_l1_freq"]      = w["broke_l1"].mean() * 100
    return sess_v2


def classify_v2(row):
    p_u2  = row["roll_p_u2_u1"]
    p_l2  = row["roll_p_l2_l1"]
    p_l3  = row["roll_p_l3_l1"]
    mu    = row["roll_avg_max_up"]
    md    = row["roll_avg_max_down"]
    flip  = row["roll_flip_rate"]
    nb    = row["roll_no_break_pct"]
    ema   = row["ema_slope"]
    dxy   = row["dxy_slope"]
    l3pct = row["roll_l3plus_pct"]
    l1freq= row["roll_l1_freq"]
    max_d = row["max_down"]
    max_u = row["max_up"]
    l1_fb = row["l1_false_brk"]

    # ── HARD EXCLUSIONS (NEUTRAL override) ─────────────────────────────────
    if l1_fb == 1 and max_d < 2:
        return "NEUTRAL"
    if flip > FLIP_RATE_MAX and max_d < 2:
        return "NEUTRAL"
    if l1freq > 55 and p_l2 < 20:
        return "NEUTRAL"

    # ── LONG ───────────────────────────────────────────────────────────────
    if (p_u2 >= LONG_PU2_MIN and
        mu > md * LONG_AVG_UP_RATIO and
        p_u2 > p_l2):
        return "LONG"

    # ── SHORT_TREND ────────────────────────────────────────────────────────
    if (md >= ST_MIN_DOWN_LEVEL and
        p_l2 >= ST_PL2_MIN and
        ema  <= ST_EMA_MAX and
        l1freq >= ST_L1_FREQ_MIN):
        return "SHORT_TREND"

    # ── SHORT_EXHAUST ──────────────────────────────────────────────────────
    if (l3pct >= SE_PL3_MIN or max_d >= SE_MIN_DOWN_LEVEL):
        if max_u < 1:
            return "SHORT_EXHAUST"
        if max_d >= SE_MIN_DOWN_LEVEL:
            return "SHORT_EXHAUST"

    return "NEUTRAL"


def session_is_complete(sess_start, last_bar_in_session, next_session_exists):
    """
    A session (18:00 NY → 17:00 NY next day) is complete when the next session
    has started in the data, or its last stored bar is at/after 16:45 NY
    (the final 15m bar before the daily 17:00 break).
    """
    if next_session_exists:
        return True
    return last_bar_in_session >= sess_start + pd.Timedelta(hours=22, minutes=45)
