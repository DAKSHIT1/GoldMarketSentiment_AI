"""
signal_rules.py
===============
SINGLE SOURCE OF TRUTH for the session-level "WHEN TO LOOK FOR TRADES" rules.

Used by:
  * forward_test_signal.py  (validation / per-session labels)
  * signal_notifier.py      (Telegram alerts)
  * dashboard/server.py     (live dashboard)

ACTIVE STRATEGY (forward paper trading, since 2026-11-20)
  EMA_falling_DXY_falling_LONG — the ONLY enabled rule:
    at session K's open the EMA200 slope is falling (< -EMA_EPS) AND the dollar
    (UUP EMA slope) is falling (< 0). Evaluated once session K has CLOSED;
    paper entry = session K close, exit = close of session K + HOLD_SESSIONS.
    Validation (lookahead-free walk-forward, 10-session hold, 0.02% cost,
    weekend pseudo-sessions removed — see session_builder.tradable_bars):
      OOS 2024-07 → 2026-11: 32 trades, 62.5% win vs 60.5% base, +1.83%/trade
      Full 2020–2026:        87 trades, 59% win, +1.20%/trade, PF 2.63,
                             max DD -12.8%, Sharpe 1.21 (beats 99% of random entries)

ARCHIVED: every older level / sequence / stay-out rule is kept in RULES for
reporting (forward_test_signal.py) but is DISABLED — it never affects a live
label. Re-validation without lookahead showed no edge over chance for them.

Labelling logic (unchanged, mirrors forward_test_signal.get_label):
  1. Collect every ENABLED rule that fires on session K.
  2. Sort by tier, then by in-sample accuracy.
  3. STRONG LONG and STRONG SHORT both firing           -> CONFLICTED (no trade)
  4. Only stay-out rules firing                          -> STAY OUT
  5. Otherwise the top-ranked directional rule decides the direction.
  With only the macro rule enabled this reduces to: fires -> LONG, else NO SIGNAL.

Pure Python on purpose: rows may be dicts or pandas Series (anything with .get).
"""

WINDOW_SESSIONS = 10      # forward window / paper-trade hold (sessions)
HOLD_SESSIONS   = 10      # hold for the active macro rule
EMA_EPS         = 0.0005  # dead-zone for the EMA slope sign tests

ACTIVE_RULE_KEY = "EMA_falling_DXY_falling_LONG"
ACTIVE_RULES    = {ACTIVE_RULE_KEY}   # everything else is archived (disabled)

# Archived-rule stats: logs/forward_test/signal_report.txt (OOS 2025-08-08 →
# 2026-08-20, generated BEFORE the lookahead fix — inflated; kept for reference).


def _f(row, key, default=0.0):
    if row is None:
        return default
    try:
        v = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return default if v != v else v   # NaN -> default


# ─────────────────────────────────────────────────────────────────────────────
#  CONDITIONS  —  each takes (cur, p1, p2): session K, K-1, K-2
# ─────────────────────────────────────────────────────────────────────────────
def _mu(r): return _f(r, "max_up")
def _md(r): return _f(r, "max_down")
def _es(r): return _f(r, "ema_slope")
def _vs(r): return _f(r, "vwap_slope")
def _ds(r): return _f(r, "dxy_slope")
def _ibd(r): return int(_f(r, "is_bidir"))
def _lfb(r): return int(_f(r, "l1_false_brk"))


def _seq(fn):
    """Sequence rules need two previous sessions."""
    def wrapped(c, p1, p2):
        if p1 is None or p2 is None:
            return False
        return fn(c, p1, p2)
    return wrapped


RULES = [
    # ── ACTIVE: macro pullback (the only enabled rule) ────────────────────
    dict(key=ACTIVE_RULE_KEY, name="Macro Pullback — EMA Falling + Dollar Falling",
         dir="LONG", tier="STRONG", hold=HOLD_SESSIONS,
         is_acc=56.4, oos=62.5, oos_n=32,          # 10-session hold; discovery <2024-07, OOS 2024-07→2026-11
         desc="EMA200 slope falling and dollar (UUP) falling at session open — gold pullback in a weak-dollar regime",
         cond=lambda c, p1, p2: _es(c) < -EMA_EPS and _ds(c) < 0),

    # ── ARCHIVED (disabled) ───────────────────────────────────────────────
    # ── LONG ──────────────────────────────────────────────────────────────
    dict(key="U3_EMA_bearish_LONG", name="U3 EMA Bearish", dir="LONG", tier="STRONG",
         is_acc=77.8, oos=72.4, oos_n=29,
         desc="U3 broke, EMA still bearish — overextension, snap back UP",
         cond=lambda c, p1, p2: _mu(c) >= 3 and _es(c) < -EMA_EPS),
    dict(key="U4_EMA_bearish_LONG", name="U4 Overextension", dir="LONG", tier="STRONG",
         is_acc=84.6, oos=75.0, oos_n=12,
         desc="U4 broke, EMA still bearish — extreme overextension LONG",
         cond=lambda c, p1, p2: _mu(c) >= 4 and _es(c) < -EMA_EPS),
    dict(key="U2_EMA_bearish_LONG", name="U2 EMA Bearish", dir="LONG", tier="STRONG",
         is_acc=76.1, oos=65.1, oos_n=43,
         desc="U2 broke but EMA bearish — bull exhaustion LONG",
         cond=lambda c, p1, p2: _mu(c) >= 2 and _es(c) < -EMA_EPS),
    dict(key="U1_EMA_bearish_LONG", name="U1 EMA Bearish", dir="LONG", tier="STRONG",
         is_acc=74.2, oos=64.8, oos_n=71,
         desc="U1 broke, EMA bearish — overextension LONG signal",
         cond=lambda c, p1, p2: _mu(c) >= 1 and _es(c) < -EMA_EPS),
    dict(key="U3_REBOUND_LONG", name="U3 Rebound", dir="LONG", tier="STRONG",
         is_acc=76.9, oos=71.4, oos_n=14,
         desc="Hit U3 then reversed in same session — look for LONG entry",
         cond=lambda c, p1, p2: _mu(c) >= 3 and _ibd(c) == 1),
    dict(key="L3_L2_L1_rising_lows", name="L3→L2→L1 Rising Lows", dir="LONG", tier="STRONG",
         is_acc=65.0, oos=50.0, oos_n=2,
         desc="3-session rising low pattern: L3→L2→L1 — downside exhausting",
         cond=_seq(lambda c, p1, p2: _md(p2) >= 3 and 2 <= _md(p1) < 3 and 1 <= _md(c) < 2)),
    dict(key="L1_false_break_LONG", name="L1 False Break", dir="LONG", tier="STRONG",
         is_acc=69.2, oos=73.9, oos_n=23,
         desc="Same session hit L1 AND U1 (false break down) — LONG signal",
         cond=lambda c, p1, p2: _lfb(c) == 1 and _mu(c) >= 1),
    dict(key="Pure_up_VWAP_bull_LONG", name="Pure Up + VWAP Bull", dir="LONG", tier="MODERATE",
         is_acc=65.0, oos=64.8, oos_n=71,
         desc="Pure upside session with VWAP bullish slope — clean LONG",
         cond=lambda c, p1, p2: _mu(c) >= 1 and _md(c) < 1 and _vs(c) > 0),
    dict(key="U2_VWAP_bull_LONG", name="U2 + VWAP Bull", dir="LONG", tier="MODERATE",
         is_acc=57.8, oos=57.1, oos_n=63,
         desc="Broke U2 + VWAP bullish — moderate LONG signal",
         cond=lambda c, p1, p2: _mu(c) >= 2 and _vs(c) > 0),
    dict(key="U2_EMA_bull_LONG", name="U2 + EMA Bull", dir="LONG", tier="MODERATE",
         is_acc=56.5, oos=54.8, oos_n=62,
         desc="Broke U2 + EMA bullish — trend following LONG",
         cond=lambda c, p1, p2: _mu(c) >= 2 and _es(c) > EMA_EPS),
    dict(key="U2_VWAP_EMA_bull_LONG", name="U2 + VWAP + EMA Bull", dir="LONG", tier="MODERATE",
         is_acc=58.9, oos=54.9, oos_n=51,
         desc="Broke U2 + VWAP + EMA both bullish — high confluence LONG",
         cond=lambda c, p1, p2: _mu(c) >= 2 and _vs(c) > 0 and _es(c) > EMA_EPS),
    dict(key="U1_VWAP_DXY_fall_LONG", name="U1 + VWAP + DXY Fall", dir="LONG", tier="MODERATE",
         is_acc=66.5, oos=66.7, oos_n=36,
         desc="U1 broke + VWAP bullish + DXY falling (weak dollar) — LONG",
         cond=lambda c, p1, p2: _mu(c) >= 1 and _vs(c) > 0 and _ds(c) < 0),
    dict(key="U2_pullback_U3_LONG", name="U2 → Pullback → U3", dir="LONG", tier="MODERATE",
         is_acc=75.0, oos=37.5, oos_n=8,
         desc="U2 break → pullback to L1 → U3 continuation",
         cond=_seq(lambda c, p1, p2: _mu(p2) >= 2 and _md(p1) >= 1 and _mu(p1) < 2 and _mu(c) >= 3)),
    dict(key="Range_breakout_up_LONG", name="Range → Range → U2 Breakout", dir="LONG", tier="MODERATE",
         is_acc=69.2, oos=100.0, oos_n=5,
         desc="2 ranging sessions then clean U2+ breakout",
         cond=_seq(lambda c, p1, p2: _ibd(p2) == 1 and _ibd(p1) == 1 and _mu(c) >= 2 and _md(c) < 1)),

    # ── SHORT ─────────────────────────────────────────────────────────────
    dict(key="L3_EMA_bull_SHORT", name="L3 EMA Bull", dir="SHORT", tier="STRONG",
         is_acc=66.7, oos=73.1, oos_n=26,
         desc="L3 broke but EMA still bullish — overextension, snap back DOWN",
         cond=lambda c, p1, p2: _md(c) >= 3 and _es(c) > EMA_EPS),
    dict(key="L3_VWAP_bull_SHORT", name="L3 VWAP Bull", dir="SHORT", tier="STRONG",
         is_acc=68.0, oos=71.4, oos_n=21,
         desc="L3 broke but VWAP still bullish — extreme overextension SHORT",
         cond=lambda c, p1, p2: _md(c) >= 3 and _vs(c) > 0),
    dict(key="L4_EMA_bull_SHORT", name="L4 Overextension", dir="SHORT", tier="STRONG",
         is_acc=78.6, oos=76.5, oos_n=17,
         desc="L4 broke with bullish EMA — extreme overextension, look for SHORT",
         cond=lambda c, p1, p2: _md(c) >= 4 and _es(c) > EMA_EPS),
    dict(key="L2_VWAP_EMA_bear_SHORT", name="L2 VWAP+EMA Bear", dir="SHORT", tier="STRONG",
         is_acc=63.9, oos=61.8, oos_n=34,
         desc="L2+ broke + VWAP + EMA both bearish — high confluence SHORT",
         cond=lambda c, p1, p2: _md(c) >= 2 and _vs(c) < 0 and _es(c) < -EMA_EPS),
    dict(key="L2_EMA_bear_SHORT", name="L2 EMA Bear", dir="SHORT", tier="STRONG",
         is_acc=62.2, oos=59.0, oos_n=39,
         desc="L2+ broke + EMA bearish — trend following SHORT",
         cond=lambda c, p1, p2: _md(c) >= 2 and _es(c) < -EMA_EPS),
    dict(key="L2_VWAP_bear_SHORT", name="L2 VWAP Bear", dir="SHORT", tier="MODERATE",
         is_acc=60.3, oos=57.1, oos_n=49,
         desc="L2+ broke + VWAP bearish — moderate SHORT signal",
         cond=lambda c, p1, p2: _md(c) >= 2 and _vs(c) < 0),
    dict(key="U3_U2_U1_declining_SHORT", name="U3→U2→U1 Declining Highs", dir="SHORT", tier="MODERATE",
         is_acc=60.0, oos=100.0, oos_n=2,
         desc="3-session declining highs: U3→U2→U1 — upside exhausting",
         cond=_seq(lambda c, p1, p2: _mu(p2) >= 3 and 2 <= _mu(p1) < 3 and 1 <= _mu(c) < 2)),
    dict(key="U1_false_break_SHORT", name="U1 False Break", dir="SHORT", tier="MODERATE",
         is_acc=38.9, oos=54.5, oos_n=33,
         desc="Hit U1+L1 same sess (ranging), watch for next L2+ breakdown",
         cond=lambda c, p1, p2: _mu(c) >= 1 and _md(c) >= 1 and _mu(c) < 2),
    dict(key="L2_retest_L3_SHORT", name="L2 → Retest → L3", dir="SHORT", tier="MODERATE",
         is_acc=60.5, oos=0.0, oos_n=0,
         desc="L2 break → retest at L1 → L3 continuation",
         cond=_seq(lambda c, p1, p2: _md(p2) >= 2 and 1 <= _md(p1) < 2 and _mu(p1) < 1 and _md(c) >= 3)),
    dict(key="L2_pullback_L3_SHORT", name="L2 → Pullback → L3", dir="SHORT", tier="MODERATE",
         is_acc=75.0, oos=75.0, oos_n=4,
         desc="L2 break → pullback to U1 → L3 continuation",
         cond=_seq(lambda c, p1, p2: _md(p2) >= 2 and _mu(p1) >= 1 and _md(p1) < 2 and _md(c) >= 3)),
    dict(key="BEST_SHORT_macro_combo", name="BEST Short Macro", dir="SHORT", tier="STRONG",
         is_acc=60.0, oos=60.6, oos_n=33,
         desc="L2+ broke + EMA bearish + DXY rising — full macro SHORT",
         cond=lambda c, p1, p2: _md(c) >= 2 and _es(c) < -EMA_EPS and _ds(c) > 0),
    dict(key="Range_breakdown_SHORT", name="Range → Range → L2 Breakdown", dir="SHORT", tier="MODERATE",
         is_acc=60.0, oos=66.7, oos_n=3,
         desc="2 ranging sessions then clean L2+ breakdown",
         cond=_seq(lambda c, p1, p2: _ibd(p2) == 1 and _ibd(p1) == 1 and _md(c) >= 2 and _mu(c) < 1)),

    # ── STAY OUT (ranging) ────────────────────────────────────────────────
    dict(key="STAY_OUT_flip_rate", name="Flip rate ≥35%", dir="NEUTRAL", tier="RANGING",
         is_acc=None, oos=None, oos_n=None,
         desc="Flip rate ≥35% — market bidirectional and choppy",
         cond=lambda c, p1, p2: _f(c, "roll_flip_rate") >= 35),
    dict(key="STAY_OUT_l1u1_trapped", name="L1+U1 trapped", dir="NEUTRAL", tier="RANGING",
         is_acc=None, oos=None, oos_n=None,
         desc="Hit L1 AND U1 same session but NOT L2 or U2 — ranging/trapped",
         cond=lambda c, p1, p2: 1 <= _md(c) < 2 and 1 <= _mu(c) < 2),
    dict(key="STAY_OUT_no_break", name="No level broken", dir="NEUTRAL", tier="RANGING",
         is_acc=None, oos=None, oos_n=None,
         desc="No level broken at all — pure consolidation, stay out",
         cond=lambda c, p1, p2: _mu(c) < 1 and _md(c) < 1),
]

for _r in RULES:
    _r["enabled"] = _r["key"] in ACTIVE_RULES

RULES_BY_KEY = {r["key"]: r for r in RULES}
ENABLED_RULES = [r for r in RULES if r["enabled"]]
_TIER_ORDER = {"STRONG": 0, "MODERATE": 1, "RANGING": 2}


def firing_rules(cur, p1=None, p2=None, include_disabled=False):
    """All rules that fire on session `cur` (p1/p2 = the two previous sessions)."""
    pool = RULES if include_disabled else ENABLED_RULES
    return [r for r in pool if r["cond"](cur, p1, p2)]


def _signal_entry(r):
    return {"key": r["key"], "name": r["name"], "acc": r["oos"], "n": r["oos_n"],
            "tier": r["tier"], "desc": r["desc"]}


def classify(cur, p1=None, p2=None):
    """
    Label one completed session. Returns:
      direction  : LONG / SHORT / NEUTRAL
      label      : human label (same wording as forward_test_signal)
      tier       : STRONG / MODERATE / RANGING
      confidence : OOS accuracy of the deciding rule (0 for NEUTRAL)
      reason     : short explanation
      signals    : rules supporting the direction (or the stay-out rules)
      key_signal : the deciding rule
    """
    fired = sorted(firing_rules(cur, p1, p2),
                   key=lambda r: (_TIER_ORDER.get(r["tier"], 3), -(r["is_acc"] or 0)))

    def neutral(label, reason, sigs=()):
        return {"direction": "NEUTRAL", "label": label, "tier": "RANGING",
                "confidence": 0.0, "reason": reason,
                "signals": [_signal_entry(r) for r in sigs], "key_signal": None}

    if not fired:
        return neutral("NO SIGNAL", "No qualifying signal — wait for setup")

    strong = [r for r in fired if r["tier"] == "STRONG"]
    if any(r["dir"] == "LONG" for r in strong) and any(r["dir"] == "SHORT" for r in strong):
        return neutral("CONFLICTED", "STRONG LONG and STRONG SHORT signals both fired — conflicted", fired)

    directional = [r for r in fired if r["dir"] != "NEUTRAL"]
    if not directional:
        return neutral("STAY OUT — RANGING", fired[0]["desc"], fired)

    primary = fired[0]
    if primary["dir"] == "NEUTRAL":       # cannot happen given the sort, kept for safety
        primary = directional[0]
    direction = primary["dir"]
    active = [r for r in fired if r["dir"] == direction]
    base = "LOOK FOR LONGS" if direction == "LONG" else "LOOK FOR SHORTS"
    label = f"{base} [{primary['tier']}{'+ CONFLUENCE' if len(active) > 1 else ''}]"
    return {
        "direction": direction,
        "label": label,
        "tier": primary["tier"],
        "confidence": float(primary["oos"]),
        "reason": f"{len(active)} signal(s) confirm {'bullish' if direction == 'LONG' else 'bearish'} window",
        "signals": [_signal_entry(r) for r in active],
        "key_signal": _signal_entry(primary),
    }


def classify_at(rows, i):
    """Classify rows[i] using rows[i-1], rows[i-2] as history (rows: list-like of dict/Series)."""
    p1 = rows[i - 1] if i >= 1 else None
    p2 = rows[i - 2] if i >= 2 else None
    return classify(rows[i], p1, p2)
