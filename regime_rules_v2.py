"""
regime_rules_v2.py
==================
FIXED SHORT REGIME CLASSIFIER

Incorporates exact findings from short_regime_analysis.py:

LONG  (unchanged, 55-71% accurate):
  rolling P(U2|U1) >= 35% AND avg_max_up > avg_max_down AND flip < 25%

SHORT — two verified flavours:
  SHORT_TREND    (60% accurate): max_down>=2 + EMA bearish + DXY rising
  SHORT_EXHAUST  (66-84% acc) : max_down>=3 in recent sessions EVEN with
                                 bullish EMA (overextension / snap-back)

NEUTRAL: everything else — no clear edge, stay flat

AVOID list (hardcoded exclusions):
  • L1 false break: max_down==1 AND same session hit max_up>=1 → noise
  • High L1 freq + no progression: lots of L1 breaks but never L2 → ranging
  • Decelerating downside: max_down shrinking each session
  • L1 retest FAIL: broke L1, bounced, reversed → skip

Outputs:
  data/regime_v2.csv           ← bar-level with new regime label
  data/session_regime_v2.csv   ← session-level with new regime label
  logs/regime_v2/accuracy.txt
  logs/regime_v2/accuracy.png
  logs/regime_v2/comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

os.makedirs("logs/regime_v2", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  THRESHOLDS (derived from short_regime_analysis.py findings)
# ─────────────────────────────────────────────────────────────────────────────
from session_builder import *   # thresholds, build_session_db_v2, rolling, classify_v2

FORWARD_SESSIONS = 10
print("=" * 70)
print("  GOLD AI — FIXED SHORT REGIME CLASSIFIER v2")
print("  Incorporating exact combinatorial analysis findings")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading session data...")
sess = pd.read_csv("data/session_regime.csv")
if "session_date" not in sess.columns:
    sess["session_date"] = pd.to_datetime(sess["datetime"])
else:
    sess["session_date"] = pd.to_datetime(sess["session_date"])
if "sid" not in sess.columns and "session_id" in sess.columns:
    sess["sid"] = sess["session_id"]
sess = sess.sort_values("session_date").reset_index(drop=True)
N = len(sess)
print(f"Sessions: {N:,}  |  {sess['session_date'].iloc[0].date()} → {sess['session_date'].iloc[-1].date()}")

print("Loading bar data for price measurement...")
bars = pd.read_csv("data/regime_features.csv")
bars["datetime"] = pd.to_datetime(bars["datetime"])
bars = bars.sort_values("datetime").fillna(0).reset_index(drop=True)
bars = tradable_bars(bars)  # drop weekend OTC quotes / 17:00 break bars (see session_builder)
bars = bars[bars.groupby("session_id")["datetime"].transform("size") >= MIN_SESSION_BARS].reset_index(drop=True)

sess_prices = bars.groupby("session_id").agg(
    sess_open  = ("open",  "first"),
    sess_close = ("close", "last"),
    sess_high  = ("high",  "max"),
    sess_low   = ("low",   "min"),
).reset_index()
sess = sess.merge(sess_prices, left_on="sid", right_on="session_id", how="left")
print(f"Price data joined.")

# ─────────────────────────────────────────────────────────────────────────────
#  REBUILD ROLLING FEATURES from scratch (using bar data)
#  — need rolling max_down, ema_slope, dxy_slope per session
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding session-level features from bar data...")

sess_v2 = build_session_db_v2(bars)
sess_v2["date"] = pd.to_datetime(sess_v2["date"])
sess_v2 = sess_v2.sort_values("date").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ROLLING FEATURES (10-session window)
# ─────────────────────────────────────────────────────────────────────────────
sess_v2 = add_rolling_features(sess_v2)

sess_v2 = sess_v2.fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
#  MEASURE FORWARD PRICE MOVEMENT
# ─────────────────────────────────────────────────────────────────────────────
print("Measuring forward price movement...")
fwd_ret = np.full(len(sess_v2), np.nan)
fwd_pts = np.full(len(sess_v2), np.nan)
fwd_up  = np.full(len(sess_v2), np.nan)

for i in range(len(sess_v2)):
    if i + FORWARD_SESSIONS >= len(sess_v2): continue
    start = sess_v2.iloc[i]["sess_open"]
    end   = sess_v2.iloc[i + FORWARD_SESSIONS]["sess_close"]
    if pd.isna(start) or start <= 0: continue
    ret = (end - start) / start * 100
    fwd_ret[i] = round(ret, 4)
    fwd_pts[i] = round(end - start, 2)
    fwd_up[i]  = 1.0 if end > start else 0.0

sess_v2["fwd_ret"]  = fwd_ret
sess_v2["fwd_pts"]  = fwd_pts
sess_v2["fwd_up"]   = fwd_up
sess_v2["fwd_down"] = 1.0 - fwd_up

# ─────────────────────────────────────────────────────────────────────────────
#  APPLY FIXED REGIME RULES
# ─────────────────────────────────────────────────────────────────────────────
print("\nApplying fixed regime rules v2...")

sess_v2["regime_v2"] = sess_v2.apply(classify_v2, axis=1)

rc = sess_v2["regime_v2"].value_counts()
print(f"  LONG         : {rc.get('LONG',0):>4}  ({rc.get('LONG',0)/N*100:.1f}%)")
print(f"  SHORT_TREND  : {rc.get('SHORT_TREND',0):>4}  ({rc.get('SHORT_TREND',0)/N*100:.1f}%)")
print(f"  SHORT_EXHAUST: {rc.get('SHORT_EXHAUST',0):>4}  ({rc.get('SHORT_EXHAUST',0)/N*100:.1f}%)")
print(f"  NEUTRAL      : {rc.get('NEUTRAL',0):>4}  ({rc.get('NEUTRAL',0)/N*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
#  ACCURACY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  REGIME v2 ACCURACY vs ACTUAL PRICE MOVEMENT")
print("=" * 70)

lines = []
def P(s=""): lines.append(s)

P("=" * 72)
P("  GOLD AI — FIXED REGIME v2 ACCURACY REPORT")
P(f"  Sessions: {N:,} | Forward: {FORWARD_SESSIONS} sessions")
P("=" * 72)
P()
P("  REGIME RULES:")
P(f"  LONG         → P(U2|U1) ≥ {LONG_PU2_MIN}% AND avg_up > avg_down AND no flip")
P(f"  SHORT_TREND  → avg_max_down ≥ {ST_MIN_DOWN_LEVEL} + P(L2|L1) ≥ {ST_PL2_MIN}% + EMA ≤ {ST_EMA_MAX} + L1 freq ≥ {ST_L1_FREQ_MIN}%")
P(f"  SHORT_EXHAUST→ L3+ hit recently OR this session max_down ≥ {SE_MIN_DOWN_LEVEL}")
P(f"  NEUTRAL      → everything else (false breaks, ranging, ambiguous)")
P()
P(f"  {'Regime':<15}  {'N':>5}  {'Correct%':>10}  {'Avg Move':>9}  {'Avg Ret':>9}  {'Status'}")
P("  " + "─" * 62)

results = {}
for regime in ["LONG", "SHORT_TREND", "SHORT_EXHAUST", "NEUTRAL"]:
    mask = (sess_v2["regime_v2"] == regime) & sess_v2["fwd_down"].notna()
    sub  = sess_v2[mask]
    n    = len(sub)
    if n == 0:
        P(f"  {regime:<15}  {'0':>5}  {'N/A':>10}")
        results[regime] = {"n": 0, "accuracy": 0}
        continue

    # For SHORT regimes: correct = price went DOWN (fwd_down==1)
    # For LONG: correct = price went UP (fwd_up==1)
    # For NEUTRAL: correct = price stayed flat (abs(ret) < 0.5%)
    if regime == "LONG":
        n_correct = (sub["fwd_up"] == 1).sum()
    elif regime in ["SHORT_TREND", "SHORT_EXHAUST"]:
        n_correct = (sub["fwd_down"] == 1).sum()
    else:
        n_correct = (sub["fwd_ret"].abs() < 0.5).sum()

    accuracy  = n_correct / n * 100
    avg_pts   = sub["fwd_pts"].mean()
    avg_ret   = sub["fwd_ret"].mean()

    if regime == "LONG":
        signed_pts = avg_pts          # positive = price went up = correct
    elif regime in ["SHORT_TREND", "SHORT_EXHAUST"]:
        signed_pts = -avg_pts         # negative pts = price went down = correct
    else:
        signed_pts = -abs(avg_pts)    # flat is good for neutral

    flag = "✅ EDGE" if accuracy >= 55 else "⚠️  MARGINAL" if accuracy >= 45 else "❌ WEAK"
    P(f"  {regime:<15}  {n:>5}  {accuracy:>9.1f}%  {avg_pts:>+8.1f}  {avg_ret:>+8.3f}%  {flag}")
    results[regime] = {
        "n": n, "accuracy": accuracy,
        "avg_pts": avg_pts, "avg_ret": avg_ret,
        "returns": sub["fwd_ret"].dropna().tolist(),
    }
    print(f"  {regime:<15}: n={n:>4}  accuracy={accuracy:.1f}%  avg_move={avg_pts:+.1f} pts  {flag}")

P()
P("─" * 72)
P("  QUARTERLY BREAKDOWN (forward 10 sessions)")
P("─" * 72)
P(f"  {'Quarter':<10}  {'LONG%':>8}  {'ST%':>8}  {'SE%':>8}  {'Distribution'}")
P("  " + "─" * 55)

sess_v2["quarter"] = pd.to_datetime(sess_v2["date"]).dt.to_period("Q")
for q, grp in sess_v2.groupby("quarter"):
    l  = grp[grp["regime_v2"] == "LONG"]
    st = grp[grp["regime_v2"] == "SHORT_TREND"]
    se = grp[grp["regime_v2"] == "SHORT_EXHAUST"]

    def qacc(sub, is_long=False):
        sub = sub.dropna(subset=["fwd_down"])
        if len(sub) == 0: return "N/A"
        nc = (sub["fwd_up"]==1).sum() if is_long else (sub["fwd_down"]==1).sum()
        a  = nc/len(sub)*100
        return f"{a:.0f}%{'✅' if a>=55 else '⚠️' if a>=45 else '❌'}"

    la  = qacc(l,  is_long=True)
    sta = qacc(st, is_long=False)
    sea = qacc(se, is_long=False)
    dist = f"L={len(l)}/ST={len(st)}/SE={len(se)}/N={len(grp)-len(l)-len(st)-len(se)}"
    P(f"  {q!s:<10}  {la:>10}  {sta:>10}  {sea:>10}  {dist}")

P()
P("─" * 72)
P("  COMPARISON: OLD (single regime) vs NEW v2 (4 states)")
P("─" * 72)
P()
P("  Old SHORT regime accuracy (from regime_validation.py):")
P("    Short (single class) fwd 10s: 38.1% ❌")
P()
P("  New v2 regime accuracy:")
for k in ["SHORT_TREND", "SHORT_EXHAUST"]:
    r = results.get(k, {})
    if r.get("n", 0) > 0:
        P(f"    {k:<16}: {r['accuracy']:.1f}% ({r['n']} sessions)")
P()
P("  Key improvements:")
P("  1. L1 false breaks excluded → removes ~19% accuracy drag")
P("  2. SHORT split into TREND + EXHAUST → each has different edge")
P("  3. EMA + DXY filter removes pullbacks misclassified as shorts")
P("  4. L3+ overextension catches counter-trend snaps (66-84%)")

report = "\n".join(lines)
print("\n" + report)
with open("logs/regime_v2/accuracy.txt", "w") as f:
    f.write(report)

# ─────────────────────────────────────────────────────────────────────────────
#  SAVE SESSION FILE WITH NEW LABELS
# ─────────────────────────────────────────────────────────────────────────────
sess_v2.to_csv("data/session_regime_v2.csv", index=False)
print(f"\n✅ data/session_regime_v2.csv saved ({len(sess_v2)} sessions)")

# ─────────────────────────────────────────────────────────────────────────────
#  JOIN BACK TO BAR-LEVEL + SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("Joining v2 regime labels to bar-level data...")
regime_join = sess_v2[["sid", "regime_v2"]].rename(columns={"sid": "session_id"})

bars_v2 = bars.merge(regime_join, on="session_id", how="left")
bars_v2["regime_v2"] = bars_v2["regime_v2"].fillna("NEUTRAL")

# Also add numeric encoding for model use
enc = {"LONG": 0, "SHORT_TREND": 1, "SHORT_EXHAUST": 2, "NEUTRAL": 3}
bars_v2["regime_v2_enc"] = bars_v2["regime_v2"].map(enc).fillna(3).astype(int)

bars_v2.to_csv("data/regime_v2.csv", index=False)
print(f"✅ data/regime_v2.csv saved ({len(bars_v2):,} bars)")

# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
DARK = "#0d1117"; BG = "#161b22"; GOLD = "#FFD700"
GREEN = "#00E676"; RED = "#FF4C4C"; GRAY = "#8B949E"
ORANGE = "#FFA726"; BLUE = "#42A5F5"; PURPLE = "#CE93D8"; TEAL = "#26C6DA"

regime_colors = {
    "LONG"         : GREEN,
    "SHORT_TREND"  : RED,
    "SHORT_EXHAUST": ORANGE,
    "NEUTRAL"      : GRAY,
}

def sx(ax, title, sub=""):
    ax.set_facecolor(BG); ax.spines[:].set_color("#30363d")
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.set_title(f"{title}\n{sub}" if sub else title,
                 color=GOLD, fontsize=9, fontweight="bold")

# ── Figure 1: Full accuracy and distribution chart ────────────────────────────
fig = plt.figure(figsize=(22, 18), facecolor=DARK)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

# 1. Accuracy bar chart
ax1 = fig.add_subplot(gs[0, :2])
regimes = ["LONG", "SHORT_TREND", "SHORT_EXHAUST", "NEUTRAL"]
accs = [results.get(r, {}).get("accuracy", 0) for r in regimes]
ns   = [results.get(r, {}).get("n", 0)        for r in regimes]
bars_plot = ax1.bar(regimes, accs,
                    color=[regime_colors[r] for r in regimes],
                    alpha=0.85, edgecolor=DARK, width=0.5)
for bar, acc, n in zip(bars_plot, accs, ns):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{acc:.1f}%\n(n={n})", ha="center", color="white",
             fontsize=11, fontweight="bold")
ax1.axhline(55, color=GOLD,  linestyle=":",  linewidth=2, label="55% edge threshold")
ax1.axhline(50, color=GRAY,  linestyle="--", linewidth=1, label="50% random")
ax1.axhline(38, color=RED,   linestyle="-.", linewidth=1.5,
            label="Old SHORT accuracy (38.1%) ← baseline")
ax1.set_ylim(0, 100); ax1.set_ylabel("Directional Accuracy %", color=GRAY)
ax1.legend(facecolor=BG, labelcolor="white", fontsize=9)
sx(ax1, "Regime v2 Directional Accuracy (Forward 10 Sessions)",
   "LONG: % price went UP  |  SHORT_*: % price went DOWN  |  NEUTRAL: % price stayed flat")

# 2. Comparison vs old
ax2 = fig.add_subplot(gs[0, 2])
comp_labels = ["Old SHORT\n(single class)", "SHORT_TREND\n(new)", "SHORT_EXHAUST\n(new)"]
comp_accs   = [38.1,
               results.get("SHORT_TREND",  {}).get("accuracy", 0),
               results.get("SHORT_EXHAUST",{}).get("accuracy", 0)]
comp_cols   = [RED, ORANGE, ORANGE]
ax2.bar(comp_labels, comp_accs, color=comp_cols, alpha=0.85, edgecolor=DARK)
for i, (lbl, acc) in enumerate(zip(comp_labels, comp_accs)):
    ax2.text(i, acc+0.5, f"{acc:.1f}%", ha="center", color="white",
             fontsize=11, fontweight="bold")
ax2.axhline(55, color=GOLD, linestyle=":", linewidth=2)
ax2.set_ylim(0, 100); ax2.set_ylabel("SHORT Accuracy %", color=GRAY)
sx(ax2, "SHORT Regime Improvement", "Old vs New v2")

# 3–6. Return distributions
for i, regime in enumerate(["LONG","SHORT_TREND","SHORT_EXHAUST","NEUTRAL"]):
    ax = fig.add_subplot(gs[1, i % 3] if i < 3 else fig.add_subplot(gs[2, 0]))
    r  = results.get(regime, {})
    rets = r.get("returns", [])
    col  = regime_colors[regime]
    if rets:
        ax.hist(rets, bins=30, color=col, alpha=0.75, edgecolor=DARK)
        mean_r = np.mean(rets)
        ax.axvline(0,      color=GRAY, linestyle="--", linewidth=1)
        ax.axvline(mean_r, color=GOLD, linestyle="-",  linewidth=1.5,
                   label=f"Mean {mean_r:+.2f}%")
        if regime == "LONG":
            ax.axvspan(0, max(rets+[1]), alpha=0.1, color=GREEN)
        elif regime in ["SHORT_TREND","SHORT_EXHAUST"]:
            ax.axvspan(min(rets+[-1]), 0, alpha=0.1, color=GREEN)
        ax.legend(facecolor=BG, labelcolor="white", fontsize=7)
    ax.set_xlabel("Price return % (fwd 10s)", color=GRAY)
    ax.set_ylabel("Count", color=GRAY)
    sx(ax, f"{regime} Return Distribution", "Green = correct direction for this regime")

# Fill remaining slots in row 1 and row 2
ax_dist4 = fig.add_subplot(gs[2, 0])
r4 = results.get("NEUTRAL", {}); rets4 = r4.get("returns", [])
if rets4:
    ax_dist4.hist(rets4, bins=30, color=GRAY, alpha=0.75, edgecolor=DARK)
    ax_dist4.axvline(np.mean(rets4), color=GOLD, linewidth=1.5,
                     label=f"Mean {np.mean(rets4):+.2f}%")
    ax_dist4.legend(facecolor=BG, labelcolor="white", fontsize=7)
ax_dist4.set_xlabel("Price return % (fwd 10s)", color=GRAY)
sx(ax_dist4, "NEUTRAL Return Distribution")

# 7. Quarterly accuracy heatmap
ax_q = fig.add_subplot(gs[2, 1:])
quarters = sorted(sess_v2["quarter"].unique())
q_labels = [str(q) for q in quarters]
x_q      = range(len(quarters))

def qarray(regimes_list, is_long=False):
    arr = []
    for q in quarters:
        grp = sess_v2[(sess_v2["quarter"] == q) & sess_v2["fwd_down"].notna()]
        sub = grp[grp["regime_v2"].isin(regimes_list)]
        if len(sub) == 0: arr.append(np.nan); continue
        nc  = (sub["fwd_up"]==1).sum() if is_long else (sub["fwd_down"]==1).sum()
        arr.append(nc/len(sub)*100)
    return arr

long_acc = qarray(["LONG"], is_long=True)
st_acc   = qarray(["SHORT_TREND"])
se_acc   = qarray(["SHORT_EXHAUST"])

def col_arr(accs):
    return [GREEN if (a is not None and not np.isnan(a) and a >= 55)
            else ORANGE if (a is not None and not np.isnan(a) and a >= 45)
            else RED for a in accs]

ax_q.bar([xi-0.25 for xi in x_q], long_acc, width=0.22,
         color=col_arr(long_acc), alpha=0.85, edgecolor=DARK, label="LONG")
ax_q.bar([xi+0.00 for xi in x_q], st_acc,   width=0.22,
         color=col_arr(st_acc),   alpha=0.75, edgecolor=DARK, label="SHORT_TREND",
         hatch="//")
ax_q.bar([xi+0.25 for xi in x_q], se_acc,   width=0.22,
         color=col_arr(se_acc),   alpha=0.65, edgecolor=DARK, label="SHORT_EXHAUST",
         hatch="xx")
ax_q.axhline(55, color=GOLD, linestyle=":", linewidth=1.5)
ax_q.axhline(50, color=GRAY, linestyle="--", linewidth=0.8)
ax_q.set_xticks(x_q); ax_q.set_xticklabels(q_labels, rotation=60, ha="right", fontsize=6)
ax_q.set_ylim(0, 105); ax_q.set_ylabel("Directional Accuracy %", color=GRAY)
ax_q.legend(facecolor=BG, labelcolor="white", fontsize=8)
sx(ax_q, "Quarterly Accuracy — v2 Regime",
   "Green ≥55% | Orange 45-55% | Red <45%")

# 8. Regime distribution over time + price path
ax_path = fig.add_subplot(gs[3, :])
fw_sess = FORWARD_SESSIONS

def avg_price_path(subset, fw):
    paths = []
    for i in subset.index.tolist():
        if i + fw >= len(sess_v2): continue
        sp = sess_v2.iloc[i]["sess_open"]
        if pd.isna(sp) or sp <= 0: continue
        path = []
        for j in range(fw + 1):
            if i+j >= len(sess_v2): break
            ep = sess_v2.iloc[i+j]["sess_close"]
            if pd.isna(ep): break
            path.append((ep - sp) / sp * 100)
        if len(path) == fw+1:
            paths.append(path)
    if not paths: return None
    arr = np.array(paths)
    return arr.mean(axis=0), arr.std(axis=0)

xp = range(fw_sess + 1)
for regime, col, lbl in [
    ("LONG",          GREEN,  f"LONG (n={rc.get('LONG',0)})"),
    ("SHORT_TREND",   RED,    f"SHORT_TREND (n={rc.get('SHORT_TREND',0)}) → INVERTED"),
    ("SHORT_EXHAUST", ORANGE, f"SHORT_EXHAUST (n={rc.get('SHORT_EXHAUST',0)}) → INVERTED"),
    ("NEUTRAL",       GRAY,   f"NEUTRAL (n={rc.get('NEUTRAL',0)})"),
]:
    sub  = sess_v2[sess_v2["regime_v2"] == regime]
    path = avg_price_path(sub, fw_sess)
    if path is None: continue
    mu, sd = path
    if regime in ["SHORT_TREND", "SHORT_EXHAUST"]:
        mu_plot = -mu  # invert so "correct short" also goes UP on chart
    else:
        mu_plot = mu
    ax_path.plot(xp, mu_plot, color=col, linewidth=2, label=lbl)
    ax_path.fill_between(xp, mu_plot - sd*0.3, mu_plot + sd*0.3, color=col, alpha=0.12)

ax_path.axhline(0, color=GRAY, linestyle="--", linewidth=0.8)
ax_path.set_xlabel("Sessions after regime signal", color=GRAY)
ax_path.set_ylabel("Avg % price change (SHORT lines inverted)", color=GRAY)
ax_path.legend(facecolor=BG, labelcolor="white", fontsize=9)
ax_path.tick_params(colors=GRAY)
sx(ax_path,
   f"Average Price Path After Regime Signal (forward {fw_sess} sessions)",
   "All lines should go UP = regime correctly predicted direction (SHORT lines are inverted)")

# Overall suptitle
long_acc_overall  = results.get("LONG",         {}).get("accuracy", 0)
st_acc_overall    = results.get("SHORT_TREND",  {}).get("accuracy", 0)
se_acc_overall    = results.get("SHORT_EXHAUST",{}).get("accuracy", 0)
fig.suptitle(
    f"GOLD AI — Fixed Regime v2  |  "
    f"LONG: {long_acc_overall:.1f}%  "
    f"SHORT_TREND: {st_acc_overall:.1f}%  "
    f"SHORT_EXHAUST: {se_acc_overall:.1f}%  "
    f"(was 38.1% single SHORT)",
    color=GOLD, fontsize=12, fontweight="bold", y=0.999
)
plt.savefig("logs/regime_v2/accuracy.png", dpi=140,
            bbox_inches="tight", facecolor=DARK)
plt.close()

print(f"\n✅ logs/regime_v2/accuracy.txt")
print(f"✅ logs/regime_v2/accuracy.png")
print(f"✅ data/session_regime_v2.csv")
print(f"✅ data/regime_v2.csv")
