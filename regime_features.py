"""
regime_features.py
==================
STEP 1 of Two-Layer System.

Builds session-level rolling statistics and Layer 1 labels:
  - For each 15m bar, computes rolling stats from the last 10 sessions
  - Labels each session as: LONG (0) / SHORT (1) / NEUTRAL (2)
  - Label logic: look forward 5 sessions, count which direction's setups
    actually won (hit 2R TP with 2:1 RR), label accordingly

Outputs:
  data/regime_features.csv   ← bar-level with regime features + label
  data/session_regime.csv    ← session-level summary
  logs/regime/regime_label_dist.txt
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import FEATURES_FILE

os.makedirs("logs/regime", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ROLL_N          = 10     # rolling window: last N sessions
LABEL_HORIZON   = 5      # look-forward N sessions to determine regime label
MIN_WINS_NEEDED = 2      # need >= 2 wins in next 5 sessions to label LONG/SHORT

ATR_SL_MULT     = 1.0
RR_RATIO        = 2.0
ATR_FALLBACK    = 5.0
SPREAD          = 0.40
MAX_HOLD_BARS   = 20

LONDON_HOURS    = set(range(3, 8))
NY_HOURS        = set(range(9, 17))
VALID_HOURS     = LONDON_HOURS | NY_HOURS

# Regime label encoding
LONG    = 0
SHORT   = 1
NEUTRAL = 2

print("=" * 65)
print("  GOLD AI — REGIME FEATURE ENGINE (Layer 1)")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading features dataset...")
df = pd.read_csv(FEATURES_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").fillna(0).reset_index(drop=True)
print(f"Bars: {len(df):,}  |  {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP A: TAG SESSIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\nTagging sessions (reset at 18:00 EST)...")
from session_builder import session_start_flags  # split by trading day, not by an 18:00 bar
df["_session_start"] = session_start_flags(df["datetime"])
df["session_id"]     = df["_session_start"].cumsum()
n_sessions           = df["session_id"].nunique()
print(f"Total sessions: {n_sessions:,}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP B: COMPUTE PER-SESSION LEVEL STATS
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing per-session level statistics...")

def session_level_stats(grp):
    """Compute level reach/direction stats for one session group."""
    mu = float(grp["seq_max_levels_up"].max())   if "seq_max_levels_up"   in grp.columns else 0.0
    md = float(grp["seq_max_levels_down"].max()) if "seq_max_levels_down" in grp.columns else 0.0

    first_up = first_dn = None
    for idx in range(len(grp)):
        if first_up is None and grp["seq_levels_broken_up"].iloc[idx]   >= 1: first_up = idx
        if first_dn is None and grp["seq_levels_broken_down"].iloc[idx] >= 1: first_dn = idx
        if first_up is not None and first_dn is not None: break

    if   first_up is not None and first_dn is not None:
        dir_first = "up_first" if first_up < first_dn else "down_first"
    elif first_up is not None: dir_first = "up_only"
    elif first_dn is not None: dir_first = "down_only"
    else:                      dir_first = "neither"

    return pd.Series({
        "session_date": grp["datetime"].iloc[0],
        "max_up"      : mu,
        "max_down"    : md,
        "dir_first"   : dir_first,
        "broke_u1"    : int(mu >= 1),
        "broke_u2"    : int(mu >= 2),
        "broke_u3"    : int(mu >= 3),
        "broke_l1"    : int(md >= 1),
        "broke_l2"    : int(md >= 2),
        "broke_l3"    : int(md >= 3),
        "is_bidirect" : int(mu >= 1 and md >= 1),
        "is_no_break" : int(mu < 1 and md < 1),
        "is_pure_up"  : int(dir_first == "up_only"),
        "is_pure_down": int(dir_first == "down_only"),
    })

has_seq = all(c in df.columns for c in
              ["seq_max_levels_up", "seq_max_levels_down",
               "seq_levels_broken_up", "seq_levels_broken_down"])

if not has_seq:
    print("❌  Sequence features missing — run feature_engine.py first!")
    exit()

sess_stats = (df.groupby("session_id", group_keys=False)
                .apply(session_level_stats)
                .reset_index(drop=False))
sess_stats.rename(columns={"session_id": "sid"}, inplace=True)
sess_stats["session_date"] = pd.to_datetime(sess_stats["session_date"])
print(f"Session stats computed: {len(sess_stats)} rows")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP C: COUNT PROFITABLE SETUPS PER SESSION
#  (how many actual BUY / SELL setups in this session hit 2R?)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCounting profitable setups per session (triple-barrier, 2:1 RR)...")

def count_session_wins(session_bars: pd.DataFrame):
    """Count profitable long and short setups in one session."""
    n      = len(session_bars)
    opens  = session_bars["open"].values
    highs  = session_bars["high"].values
    lows   = session_bars["low"].values
    atrs   = session_bars["atr"].values if "atr" in session_bars.columns else np.full(n, ATR_FALLBACK)
    hours  = session_bars["datetime"].dt.hour.values
    dists  = session_bars["dist_to_level"].values if "dist_to_level" in session_bars.columns else np.full(n, 0.0)

    long_wins = short_wins = 0
    for i in range(n - MAX_HOLD_BARS - 1):
        if int(hours[i]) not in VALID_HOURS:
            continue
        atr = float(atrs[i]) if float(atrs[i]) > 0 else ATR_FALLBACK
        if float(dists[i]) > atr * 1.5:
            continue

        next_open = float(opens[i + 1])
        sl_dist   = atr * ATR_SL_MULT
        tp_dist   = sl_dist * RR_RATIO

        # Long
        entry_l    = next_open + SPREAD / 2
        sl_l       = entry_l - sl_dist
        tp_l       = entry_l + tp_dist
        tgt1r_l    = entry_l + sl_dist
        lw = ll    = be_l = False; sl_l_cur = sl_l
        for j in range(1, MAX_HOLD_BARS):
            idx = i + j
            if idx >= n: break
            h, l = float(highs[idx]), float(lows[idx])
            if not be_l and h >= tgt1r_l: sl_l_cur = entry_l; be_l = True
            if l <= sl_l_cur: ll = True; break
            if h >= tp_l:     lw = True; break
        if lw and not ll: long_wins += 1

        # Short
        entry_s    = next_open - SPREAD / 2
        sl_s       = entry_s + sl_dist
        tp_s       = entry_s - tp_dist
        tgt1r_s    = entry_s - sl_dist
        sw = sl_s2 = be_s = False; sl_s_cur = sl_s
        for j in range(1, MAX_HOLD_BARS):
            idx = i + j
            if idx >= n: break
            h, l = float(highs[idx]), float(lows[idx])
            if not be_s and l <= tgt1r_s: sl_s_cur = entry_s; be_s = True
            if h >= sl_s_cur: sl_s2 = True; break
            if l <= tp_s:     sw    = True; break
        if sw and not sl_s2: short_wins += 1

    return long_wins, short_wins

# Get unique sessions and compute wins (use global df for accurate look-forward)
session_wins = {}
session_ids  = sorted(df["session_id"].unique())
total        = len(session_ids)
for si, sid in enumerate(session_ids):
    if si % 100 == 0:
        print(f"  Processing session {si+1}/{total}...", end="\r")
    grp = df[df["session_id"] == sid]
    lw, sw = count_session_wins(grp)
    session_wins[sid] = {"long_wins": lw, "short_wins": sw}

print(f"\n  Done. Win counts computed for {total} sessions.")
wins_df = pd.DataFrame.from_dict(session_wins, orient="index").reset_index()
wins_df.rename(columns={"index": "sid"}, inplace=True)
sess_stats = sess_stats.merge(wins_df, on="sid", how="left")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP D: LAYER 1 LABELS — forward-looking 5-session window
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Layer 1 labels (forward 5-session window)...")
labels = np.full(len(sess_stats), NEUTRAL, dtype=int)

for i in range(len(sess_stats) - LABEL_HORIZON):
    fwd    = sess_stats.iloc[i+1 : i+1+LABEL_HORIZON]
    lw_sum = int(fwd["long_wins"].sum())
    sw_sum = int(fwd["short_wins"].sum())
    if lw_sum >= MIN_WINS_NEEDED and lw_sum > sw_sum:
        labels[i] = LONG
    elif sw_sum >= MIN_WINS_NEEDED and sw_sum > lw_sum:
        labels[i] = SHORT
    else:
        labels[i] = NEUTRAL

sess_stats["regime_label"] = labels
label_counts = pd.Series(labels).value_counts().sort_index()
print(f"  LONG (0)    : {label_counts.get(0,0):,}")
print(f"  SHORT (1)   : {label_counts.get(1,0):,}")
print(f"  NEUTRAL (2) : {label_counts.get(2,0):,}")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP E: ROLLING REGIME FEATURES (last N sessions → per bar)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nComputing rolling {ROLL_N}-session statistics for each session...")

roll_records = []
for i, row in sess_stats.iterrows():
    window = sess_stats.iloc[max(0, i-ROLL_N):i]  # last N sessions
    n_w    = len(window)
    if n_w == 0:
        roll_records.append({})
        continue

    # Basic reach rates
    broke_u1 = window["broke_u1"].values
    broke_l1 = window["broke_l1"].values
    broke_u2 = window["broke_u2"].values
    broke_l2 = window["broke_l2"].values
    broke_u3 = window["broke_u3"].values
    broke_l3 = window["broke_l3"].values

    n_u1 = max(broke_u1.sum(), 1)
    n_l1 = max(broke_l1.sum(), 1)

    roll_records.append({
        "roll_pct_no_break_10s"   : window["is_no_break"].mean() * 100,
        "roll_pct_up_first_10s"   : (window["dir_first"].isin(["up_first","up_only"])).mean() * 100,
        "roll_pct_down_first_10s" : (window["dir_first"].isin(["down_first","down_only"])).mean() * 100,
        "roll_pct_pure_up_10s"    : window["is_pure_up"].mean() * 100,
        "roll_pct_pure_down_10s"  : window["is_pure_down"].mean() * 100,
        "roll_p_u2_given_u1_10s"  : broke_u2.sum() / n_u1 * 100,
        "roll_p_l2_given_l1_10s"  : broke_l2.sum() / n_l1 * 100,
        "roll_p_u3_given_u1_10s"  : broke_u3.sum() / n_u1 * 100,
        "roll_p_l3_given_l1_10s"  : broke_l3.sum() / n_l1 * 100,
        "roll_flip_rate_10s"      : window["is_bidirect"].mean() * 100,
        "roll_avg_max_up_10s"     : window["max_up"].mean(),
        "roll_avg_max_down_10s"   : window["max_down"].mean(),
        "roll_pct_u3plus_10s"     : window["broke_u3"].mean() * 100,
        "roll_pct_l3plus_10s"     : window["broke_l3"].mean() * 100,
        "roll_up_over_down_ratio" : (window["max_up"].mean() + 0.01)
                                   / (window["max_down"].mean() + 0.01),
        "roll_long_wins_5s"       : window.tail(5)["long_wins"].sum(),
        "roll_short_wins_5s"      : window.tail(5)["short_wins"].sum(),
    })

roll_df = pd.DataFrame(roll_records).fillna(0)
sess_stats = pd.concat([sess_stats.reset_index(drop=True),
                        roll_df.reset_index(drop=True)], axis=1)

sess_stats.to_csv("data/session_regime.csv", index=False)
print(f"  Saved: data/session_regime.csv ({len(sess_stats)} sessions)")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP F: JOIN REGIME FEATURES BACK TO BAR-LEVEL DF
# ─────────────────────────────────────────────────────────────────────────────
print("\nJoining regime features to bar-level dataset...")

REGIME_FEATURES = [
    "roll_pct_no_break_10s", "roll_pct_up_first_10s", "roll_pct_down_first_10s",
    "roll_pct_pure_up_10s", "roll_pct_pure_down_10s",
    "roll_p_u2_given_u1_10s", "roll_p_l2_given_l1_10s",
    "roll_p_u3_given_u1_10s", "roll_p_l3_given_l1_10s",
    "roll_flip_rate_10s", "roll_avg_max_up_10s", "roll_avg_max_down_10s",
    "roll_pct_u3plus_10s", "roll_pct_l3plus_10s",
    "roll_up_over_down_ratio",
    "roll_long_wins_5s", "roll_short_wins_5s",
    "regime_label",
]

sess_join = sess_stats[["sid"] + REGIME_FEATURES].copy()
sess_join.rename(columns={"sid": "session_id"}, inplace=True)

df_out = df.merge(sess_join, on="session_id", how="left")
df_out[REGIME_FEATURES] = df_out[REGIME_FEATURES].fillna(0)

print(f"  Merged. Output rows: {len(df_out):,}")
print(f"  Regime columns added: {REGIME_FEATURES}")

df_out.to_csv("data/regime_features.csv", index=False)
print(f"  Saved: data/regime_features.csv")

# ─────────────────────────────────────────────────────────────────────────────
#  STEP G: REPORT + CHART
# ─────────────────────────────────────────────────────────────────────────────
label_names = {0: "LONG", 1: "SHORT", 2: "NEUTRAL"}
report_lines = [
    "=" * 65,
    "  GOLD AI — REGIME LABEL DISTRIBUTION",
    "=" * 65,
    "",
    f"  Total sessions  : {len(sess_stats):,}",
    f"  LONG  (0)       : {label_counts.get(0,0):,}  ({label_counts.get(0,0)/len(sess_stats)*100:.1f}%)",
    f"  SHORT (1)       : {label_counts.get(1,0):,}  ({label_counts.get(1,0)/len(sess_stats)*100:.1f}%)",
    f"  NEUTRAL (2)     : {label_counts.get(2,0):,}  ({label_counts.get(2,0)/len(sess_stats)*100:.1f}%)",
    "",
    "  Quarterly breakdown:",
]
sess_stats["quarter"] = pd.to_datetime(sess_stats["session_date"]).dt.to_period("Q")
for q, grp in sess_stats.groupby("quarter"):
    lbl = grp["regime_label"].value_counts().to_dict()
    report_lines.append(
        f"  {q}  LONG={lbl.get(0,0):3d}  SHORT={lbl.get(1,0):3d}  NEUTRAL={lbl.get(2,0):3d}"
    )

report = "\n".join(report_lines)
print("\n" + report)
with open("logs/regime/regime_label_dist.txt", "w") as f:
    f.write(report)

# Simple chart
DARK = "#0d1117"; BG = "#161b22"; GOLD = "#FFD700"
GREEN = "#00E676"; RED = "#FF4C4C"; GRAY = "#8B949E"
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=DARK)

# 1. Label distribution pie
ax = axes[0]; ax.set_facecolor(BG)
vals = [label_counts.get(0,0), label_counts.get(1,0), label_counts.get(2,0)]
ax.pie(vals, labels=["LONG","SHORT","NEUTRAL"],
       colors=[GREEN, RED, GRAY], autopct="%1.1f%%",
       textprops={"color":"white","fontsize":10})
ax.set_title("Regime Label Distribution", color=GOLD, fontweight="bold")

# 2. Rolling avg max up vs down over time (session-level)
ax2 = axes[1]; ax2.set_facecolor(BG)
ax2.spines[:].set_color("#30363d"); ax2.tick_params(colors=GRAY, labelsize=8)
x = range(len(sess_stats))
ax2.plot(x, sess_stats["roll_avg_max_up_10s"],   color=GREEN, linewidth=1, label="Roll Avg Max Up")
ax2.plot(x, sess_stats["roll_avg_max_down_10s"],  color=RED,   linewidth=1, label="Roll Avg Max Down")
ax2.legend(facecolor=BG, labelcolor="white", fontsize=8)
ax2.set_title("Rolling 10-Session Avg Max Levels Broken", color=GOLD, fontweight="bold")
ax2.set_xlabel("Session #", color=GRAY)

# 3. Quarterly regime distribution stacked bar
ax3 = axes[2]; ax3.set_facecolor(BG)
ax3.spines[:].set_color("#30363d"); ax3.tick_params(colors=GRAY, labelsize=7)
qgroups = sess_stats.groupby("quarter")["regime_label"].value_counts(normalize=True).unstack(fill_value=0)
qs = [str(q) for q in qgroups.index]
xl = range(len(qs))
b0 = qgroups.get(0, pd.Series([0]*len(qs), index=qgroups.index)).values * 100
b1 = qgroups.get(1, pd.Series([0]*len(qs), index=qgroups.index)).values * 100
b2 = qgroups.get(2, pd.Series([0]*len(qs), index=qgroups.index)).values * 100
ax3.bar(xl, b0, color=GREEN, alpha=0.85, label="LONG")
ax3.bar(xl, b1, bottom=b0, color=RED, alpha=0.85, label="SHORT")
ax3.bar(xl, b2, bottom=b0+b1, color=GRAY, alpha=0.85, label="NEUTRAL")
ax3.set_xticks(xl); ax3.set_xticklabels(qs, rotation=60, ha="right", fontsize=6)
ax3.legend(facecolor=BG, labelcolor="white", fontsize=8)
ax3.set_title("Quarterly Regime Breakdown %", color=GOLD, fontweight="bold")
ax3.set_ylabel("% sessions", color=GRAY)

fig.suptitle("GOLD AI — Layer 1 Regime Labels", color=GOLD, fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("logs/regime/regime_labels.png", dpi=130, bbox_inches="tight", facecolor=DARK)
plt.close()

print("\n✅ Step 1 complete!")
print("  data/session_regime.csv       ← session-level stats + labels")
print("  data/regime_features.csv      ← bar-level with all regime features")
print("  logs/regime/regime_label_dist.txt")
print("  logs/regime/regime_labels.png")
print("\n→ Next: python3 regime_train.py")
