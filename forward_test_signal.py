"""
forward_test_signal.py
=======================
FORWARD TEST — "WHEN TO LOOK FOR TRADES" SIGNALS

Takes the 26 high-confidence signals (14 LONG + 12 SHORT) discovered from
499 combinatorial tests, applies them ONLY to the OOS (out-of-sample) portion
of the data (last 20% of sessions = ~2025–2026), and measures:

  1. Per-signal accuracy on OOS data
  2. In-sample vs OOS accuracy (overfitting check)
  3. Per-session "LOOK FOR LONGS / LOOK FOR SHORTS / STAY OUT" label
  4. Signal co-occurrence (when multiple signals agree)
  5. Accuracy by signal strength tier (Strong / Moderate)

Output:
  logs/forward_test/signal_report.txt       ← full stats
  logs/forward_test/oos_accuracy.png        ← visual
  logs/forward_test/session_signals.csv     ← per-session label
  logs/forward_test/WHEN_TO_LOOK_live.txt  ← human-readable rules
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import signal_rules
import warnings
warnings.filterwarnings("ignore")

os.makedirs("logs/forward_test", exist_ok=True)

DARK  = "#0d1117"; BG = "#161b22"; GOLD = "#FFD700"
GREEN = "#00E676"; RED = "#FF4C4C"; GRAY = "#8B949E"
ORANGE= "#FFA726"; BLUE = "#42A5F5"; TEAL = "#26C6DA"

FW = 10     # primary forward window (sessions)
SPLIT = 0.80   # 80% in-sample, 20% OOS

lines = []
def P(s=""): lines.append(s); print(s)

P("=" * 72)
P("  GOLD AI — FORWARD TEST: 'WHEN TO LOOK FOR TRADES'")
P(f"  OOS = last {int((1-SPLIT)*100)}% of sessions | Forward window = {FW} sessions")
P("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
sess = pd.read_csv("data/session_regime_v2.csv")
sess["date"] = pd.to_datetime(sess["date"])
sess = sess.sort_values("date").reset_index(drop=True)
N    = len(sess)

split_idx = int(N * SPLIT)
insample  = sess.iloc[:split_idx].copy().reset_index(drop=True)
oos       = sess.iloc[split_idx:].copy().reset_index(drop=True)
N_oos     = len(oos)

P(f"\nTotal sessions  : {N:,}")
P(f"In-sample       : {len(insample):,}  ({insample['date'].iloc[0].date()} → {insample['date'].iloc[-1].date()})")
P(f"OOS (fwd test)  : {N_oos:,}  ({oos['date'].iloc[0].date()} → {oos['date'].iloc[-1].date()})")

# Build forward returns on BOTH datasets (using prices from oos itself)
def build_fwd(df, fw=FW):
    df = df.copy()
    fwd_up  = np.full(len(df), np.nan)
    fwd_pts = np.full(len(df), np.nan)
    for i in range(len(df) - fw):
        # Signal for session i is only known once session i has CLOSED, so the
        # forward return starts at its close (was: its open → included the
        # very move that produced the signal).
        s = df.iloc[i]["sess_close"]
        e = df.iloc[i + fw]["sess_close"]
        if pd.isna(s) or s <= 0: continue
        fwd_pts[i] = e - s
        fwd_up[i]  = 1.0 if e > s else 0.0
    df[f"fwd_up"]  = fwd_up
    df[f"fwd_pts"] = fwd_pts
    df[f"fwd_dn"]  = 1.0 - fwd_up
    return df

insample = build_fwd(insample)
oos      = build_fwd(oos)
P("Forward returns computed.")

# Helpers
def acc_up_df(df, mask):
    sub = df[mask].dropna(subset=["fwd_up"])
    n   = len(sub)
    if n == 0: return 0, 0, 0.0, 0.0
    nc  = (sub["fwd_up"] == 1).sum()
    return nc, n, nc/n*100, sub["fwd_pts"].mean()

def acc_dn_df(df, mask):
    sub = df[mask].dropna(subset=["fwd_dn"])
    n   = len(sub)
    if n == 0: return 0, 0, 0.0, 0.0
    nc  = (sub["fwd_dn"] == 1).sum()
    return nc, n, nc/n*100, sub["fwd_pts"].mean()

def flag(acc):
    return "✅ EDGE" if acc >= 55 else "⚠️  WEAK" if acc >= 45 else "❌ AVOID"

# ─────────────────────────────────────────────────────────────────────────────
#  DEFINE ALL 26 SIGNAL CONDITIONS
#  Each returns (mask_for_insample, mask_for_oos, direction, tier, description)
# ─────────────────────────────────────────────────────────────────────────────

# Rules come from signal_rules.py (shared with the notifier and dashboard).
# Masks are evaluated on the FULL session table so sequence rules at the start
# of the OOS split can see their two previous sessions.
ALL_RECS = sess.to_dict("records")
ALL_FIRED = [set(r["key"] for r in signal_rules.firing_rules(
                 ALL_RECS[i], ALL_RECS[i-1] if i >= 1 else None,
                 ALL_RECS[i-2] if i >= 2 else None, include_disabled=True))
             for i in range(N)]

def define_signals(df, offset=0):
    signals = {}
    for r in signal_rules.RULES:
        mask = np.array([r["key"] in ALL_FIRED[offset + i] for i in range(len(df))], dtype=bool)
        signals[r["key"]] = {
            "mask": mask, "dir": r["dir"], "tier": r["tier"],
            "insample_acc": r["is_acc"] if r["is_acc"] is not None else np.nan,
            "enabled": r["enabled"], "desc": r["desc"],
        }
    return signals


# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTE IN-SAMPLE AND OOS ACCURACY FOR EVERY SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
sigs_is  = define_signals(insample, 0)
sigs_oos = define_signals(oos, split_idx)

P("\n" + "=" * 72)
P("  FORWARD TEST RESULTS — OOS vs IN-SAMPLE COMPARISON")
P(f"  OOS: {oos['date'].iloc[0].date()} → {oos['date'].iloc[-1].date()}  ({N_oos} sessions)")
P("=" * 72)
P()
P(f"  {'Signal':<30}  {'IS Acc':>7}  {'IS N':>5}  {'OOS Acc':>8}  {'OOS N':>6}  {'Avg Pts':>8}  Status")
P("  " + "─" * 82)

results = []
for sig_name, is_sig in sigs_is.items():
    oos_sig = sigs_oos.get(sig_name, {})
    direction = is_sig["dir"]
    tier      = is_sig["tier"]
    is_acc_stated = is_sig.get("insample_acc", np.nan)

    if direction == "NEUTRAL":
        is_mask_v = is_sig["mask"]
        if isinstance(is_mask_v, np.ndarray): is_mask_s = pd.Series(is_mask_v, index=insample.index)
        else: is_mask_s = is_mask_v
        n_is_actual = int(is_mask_s.sum())
        oos_mask_v = oos_sig["mask"]
        if isinstance(oos_mask_v, np.ndarray): oos_mask_s = pd.Series(oos_mask_v, index=oos.index)
        else: oos_mask_s = oos_mask_v
        n_oos = int(oos_mask_s.sum())
        P(f"  {sig_name:<30}  {'N/A':>7}  {n_is_actual:>5}  {'N/A':>8}  {n_oos:>6}  {'—':>8}  ⚪ STAY OUT")
        results.append({"signal":sig_name,"dir":direction,"tier":tier,
                         "is_acc":np.nan,"oos_acc":np.nan,"oos_n":n_oos,"oos_pts":0,
                         "desc":is_sig["desc"]})
        continue

    # In-sample
    is_mask_v = is_sig["mask"]
    if isinstance(is_mask_v, np.ndarray):
        is_mask_s = pd.Series(is_mask_v, index=insample.index)
    else:
        is_mask_s = is_mask_v
    if direction == "LONG":
        _, n_is_a, is_acc_a, _ = acc_up_df(insample, is_mask_s)
    else:
        _, n_is_a, is_acc_a, _ = acc_dn_df(insample, is_mask_s)

    # OOS
    oos_mask_v = oos_sig["mask"]
    if isinstance(oos_mask_v, np.ndarray):
        oos_mask_s = pd.Series(oos_mask_v, index=oos.index)
    else:
        oos_mask_s = oos_mask_v

    if direction == "LONG":
        _, n_oos, oos_acc, oos_pts = acc_up_df(oos, oos_mask_s)
    else:
        _, n_oos, oos_acc, oos_pts = acc_dn_df(oos, oos_mask_s)
        oos_pts = -oos_pts  # invert so positive = correct direction

    holds = oos_acc >= 55
    drift = oos_acc - is_acc_a
    status = ("✅ HOLDS" if holds else "⚠️  WEAKER" if oos_acc >= 45 else "❌ FAILED")

    P(f"  {sig_name:<30}  {is_acc_a:>6.1f}%  {n_is_a:>5}  {oos_acc:>7.1f}%  {n_oos:>6}  {oos_pts:>+8.1f}  {status}  (Δ{drift:+.1f}%)")
    results.append({"signal":sig_name,"dir":direction,"tier":tier,
                     "is_acc":is_acc_a,"oos_acc":oos_acc,"oos_n":n_oos,"oos_pts":oos_pts,
                     "drift":drift,"holds":holds,"desc":is_sig["desc"]})

rdf = pd.DataFrame(results)
P()

# Summary statistics
long_res  = rdf[(rdf["dir"]=="LONG")  & rdf["oos_acc"].notna()]
short_res = rdf[(rdf["dir"]=="SHORT") & rdf["oos_acc"].notna()]
P("  ── SUMMARY ─────────────────────────────────────────────────────────")
P(f"  LONG signals  tested : {len(long_res)}")
if len(long_res)>0:
    P(f"  LONG avg OOS acc     : {long_res['oos_acc'].mean():.1f}%")
    P(f"  LONG signals ≥55%    : {(long_res['oos_acc']>=55).sum()}/{len(long_res)}")
P(f"  SHORT signals tested : {len(short_res)}")
if len(short_res)>0:
    P(f"  SHORT avg OOS acc    : {short_res['oos_acc'].mean():.1f}%")
    P(f"  SHORT signals ≥55%   : {(short_res['oos_acc']>=55).sum()}/{len(short_res)}")
P()

# ─────────────────────────────────────────────────────────────────────────────
#  PER-SESSION SIGNAL ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
P("  ── PER-SESSION SIGNAL LABELS (OOS) ────────────────────────────────")

session_output = oos[["date","max_up","max_down"]].copy()
session_output["sess_start"] = session_output["date"]
session_output["date"] = session_output["date"].dt.date

# Priority: STRONG > MODERATE | LONG vs SHORT | STAY_OUT last
oos_sigs = define_signals(oos, split_idx)

def get_label(i):
    """Shared label logic (signal_rules.classify) — identical to the live notifier."""
    c = signal_rules.classify_at(ALL_RECS, split_idx + i)
    return c["label"], c["direction"], [x["key"] for x in c["signals"]], c["confidence"]


labels = []; dirs = []; active_sigs = []; confs = []
for i in range(N_oos):
    l, d, a, c = get_label(i)
    labels.append(l); dirs.append(d); active_sigs.append("|".join(a)); confs.append(c)

session_output["signal"]      = labels
session_output["direction"]   = dirs
session_output["active_sigs"] = active_sigs
session_output["confidence"]  = confs
session_output["fwd_10s_up"]  = oos["fwd_up"].values
session_output["fwd_10s_pts"] = oos["fwd_pts"].values

session_output.to_csv("logs/forward_test/session_signals.csv", index=False)

# Print last 30 sessions
P(f"\n  Last 30 OOS sessions signal output:")
P(f"  {'Date':<12}  {'Signal':<45}  {'Fwd Correct?':>13}")
P("  " + "─" * 75)
for _, row in session_output.tail(30).iterrows():
    fwd_ok = row["fwd_10s_up"]
    if row["direction"] == "LONG":
        correct = "✅ UP" if fwd_ok==1 else "❌ DOWN"
    elif row["direction"] == "SHORT":
        correct = "✅ DOWN" if fwd_ok==0 else "❌ UP"
    else:
        correct = "⚪ N/A"
    if pd.isna(fwd_ok): correct = "⏳ pending"
    P(f"  {str(row['date']):<12}  {row['signal']:<45}  {correct:>13}")

# Signal distribution
P()
dist = session_output["direction"].value_counts()
P("  Signal distribution across OOS sessions:")
for d, c in dist.items():
    P(f"    {d:<15}: {c:>4} sessions ({c/N_oos*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFLUENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
P("\n  ── CONFLUENCE ANALYSIS ─────────────────────────────────────────────")
P("  When multiple signals agree, is accuracy higher?")
P()

for direction in ["LONG","SHORT"]:
    dir_sessions = session_output[session_output["direction"]==direction].copy()
    dir_sessions["n_signals"] = dir_sessions["active_sigs"].apply(
        lambda x: len(x.split("|")) if x else 0
    )
    for n_sigs in [1, 2, 3]:
        sub = dir_sessions[dir_sessions["n_signals"] >= n_sigs].dropna(subset=["fwd_10s_up"])
        if len(sub) < 3: continue
        if direction == "LONG":
            acc = (sub["fwd_10s_up"]==1).mean()*100
        else:
            acc = (sub["fwd_10s_up"]==0).mean()*100
        P(f"  {direction} with ≥{n_sigs} signal(s): {len(sub):>4} sessions → accuracy {acc:.1f}%  {flag(acc)}")

# ─────────────────────────────────────────────────────────────────────────────
#  FINAL RULES FILE
# ─────────────────────────────────────────────────────────────────────────────
rules_lines = []
def R(s=""): rules_lines.append(s)

R("=" * 68)
R("  GOLD AI — WHEN TO LOOK FOR TRADES (Forward Tested Rules)")
R(f"  OOS validation: {oos['date'].iloc[0].date()} → {oos['date'].iloc[-1].date()}")
R("=" * 68)
R()
R("  STRONG SIGNALS (>60% OOS accuracy):")
R("  ─────────────────────────────────────────────────────────────────")
strong_results = rdf[(rdf["tier"]=="STRONG") & (rdf["oos_acc"]>=55) & rdf["oos_acc"].notna()]
for _, row in strong_results.sort_values("oos_acc", ascending=False).iterrows():
    d = "▲ LONG" if row["dir"]=="LONG" else "▼ SHORT"
    R(f"  {d}  {row['signal']:<35} OOS: {row['oos_acc']:.1f}% (n={row['oos_n']})")
    R(f"       → {row['desc']}")

R()
R("  MODERATE SIGNALS (55-60% OOS accuracy):")
R("  ─────────────────────────────────────────────────────────────────")
mod_results = rdf[(rdf["tier"]=="MODERATE") & (rdf["oos_acc"]>=55) & rdf["oos_acc"].notna()]
for _, row in mod_results.sort_values("oos_acc", ascending=False).iterrows():
    d = "▲ LONG" if row["dir"]=="LONG" else "▼ SHORT"
    R(f"  {d}  {row['signal']:<35} OOS: {row['oos_acc']:.1f}% (n={row['oos_n']})")
    R(f"       → {row['desc']}")

R()
R("  STAY OUT — RANGING conditions:")
R("  ─────────────────────────────────────────────────────────────────")
for _, row in rdf[rdf["dir"]=="NEUTRAL"].iterrows():
    R(f"  ⚪ {row['signal']}")
    R(f"     → {row['desc']}")

R()
R("  SIGNALS THAT FAILED OOS (avoid these):")
R("  ─────────────────────────────────────────────────────────────────")
failed = rdf[(rdf["dir"]!="NEUTRAL") & (rdf["oos_acc"]<50) & rdf["oos_acc"].notna()]
for _, row in failed.sort_values("oos_acc").iterrows():
    d = "▲ LONG" if row["dir"]=="LONG" else "▼ SHORT"
    R(f"  {d}  {row['signal']:<35} OOS: {row['oos_acc']:.1f}%  ❌ AVOID")

rules_text = "\n".join(rules_lines)
with open("logs/forward_test/WHEN_TO_LOOK_live.txt", "w") as f:
    f.write(rules_text)
P("\n" + rules_text)

# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
def sx(ax, title, sub=""):
    ax.set_facecolor(BG); ax.spines[:].set_color("#30363d")
    ax.tick_params(colors=GRAY, labelsize=7)
    ax.set_title(f"{title}\n{sub}" if sub else title,
                 color=GOLD, fontsize=8.5, fontweight="bold")

fig = plt.figure(figsize=(24, 20), facecolor=DARK)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.40)

# ── Panel 1: IS vs OOS accuracy (LONG signals) ───────────────────────────
ax1 = fig.add_subplot(gs[0, :])
long_r  = rdf[(rdf["dir"]=="LONG")  & rdf["oos_acc"].notna()].sort_values("oos_acc", ascending=False)
short_r = rdf[(rdf["dir"]=="SHORT") & rdf["oos_acc"].notna()].sort_values("oos_acc", ascending=False)
combined = pd.concat([long_r, short_r])

x = np.arange(len(combined))
is_vals  = combined["is_acc"].values
oos_vals = combined["oos_acc"].values
sig_lbls = [f"{'L' if r=='LONG' else 'S'} {n[:22]}" for n, r in zip(combined["signal"], combined["dir"])]
tier_cols = [GREEN if r=="LONG" else RED for r in combined["dir"]]

ax1.bar(x - 0.2, is_vals,  0.35, color=[c+"88" for c in tier_cols],
        edgecolor=DARK, label="In-Sample acc", alpha=0.6)
ax1.bar(x + 0.2, oos_vals, 0.35, color=tier_cols,
        edgecolor=DARK, label="OOS acc", alpha=0.9)
ax1.axhline(55, color=GOLD, linestyle=":", linewidth=2, label="55% edge")
ax1.axhline(50, color=GRAY, linestyle="--", linewidth=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(sig_lbls, rotation=45, ha="right", fontsize=6.5)
ax1.set_ylim(0, 100); ax1.set_ylabel("Accuracy %", color=GRAY)
ax1.legend(facecolor=BG, labelcolor="white", fontsize=8)
sx(ax1, "In-Sample vs OOS Accuracy — All 26 Signals",
   "Faded bars = in-sample | Solid bars = OOS forward test | Green=LONG | Red=SHORT")

# Add drift arrows / markers for large drops
for xi, (is_v, oos_v, n) in enumerate(zip(is_vals, oos_vals, combined["oos_n"])):
    drift = oos_v - is_v
    col = GREEN if drift >= 0 else RED
    ax1.annotate(f"Δ{drift:+.0f}", xy=(xi+0.2, oos_v+1), ha="center",
                 color=col, fontsize=5.5, fontweight="bold")

# ── Panel 2: OOS signal distribution ──────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
dist_vals = session_output["direction"].value_counts()
cols_pie  = [GREEN if d=="LONG" else RED if d=="SHORT" else GRAY
             for d in dist_vals.index]
ax2.pie(dist_vals.values, labels=dist_vals.index,
        colors=cols_pie, autopct="%1.1f%%",
        textprops={"color":"white","fontsize":9},
        wedgeprops={"edgecolor":DARK,"linewidth":2})
sx(ax2, "OOS Session Distribution", "LOOK FOR LONGS / SHORTS / STAY OUT / None")

# ── Panel 3: Confluence accuracy ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
conf_data = {"1+ LONG":[], "2+ LONG":[], "3+ LONG":[],
             "1+ SHORT":[], "2+ SHORT":[]}
session_output_plot = session_output.copy()
session_output_plot["n_sigs"] = session_output_plot["active_sigs"].apply(
    lambda x: len(x.split("|")) if x and x!="" else 0)

for direction in ["LONG","SHORT"]:
    ds = session_output_plot[session_output_plot["direction"]==direction].dropna(subset=["fwd_10s_up"])
    ds["n_sigs_num"] = ds["active_sigs"].apply(lambda x: len(x.split("|")) if x else 0)
    for n_sigs in [1, 2, 3]:
        sub = ds[ds["n_sigs_num"] >= n_sigs]
        if len(sub) < 2: continue
        if direction == "LONG":
            acc = (sub["fwd_10s_up"]==1).mean()*100
        else:
            acc = (sub["fwd_10s_up"]==0).mean()*100
        key = f"{n_sigs}+ {direction}"
        conf_data[key] = (acc, len(sub))

conf_keys = [k for k,v in conf_data.items() if v]
conf_accs = [conf_data[k][0] for k in conf_keys]
conf_ns   = [conf_data[k][1] for k in conf_keys]
conf_cols = [GREEN if "LONG" in k else RED for k in conf_keys]
bars3 = ax3.bar(conf_keys, conf_accs, color=conf_cols, alpha=0.85, edgecolor=DARK)
ax3.axhline(55, color=GOLD, linestyle=":", linewidth=1.5, label="55% edge")
for bar, acc, n in zip(bars3, conf_accs, conf_ns):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{acc:.1f}%\n(n={n})", ha="center", color="white", fontsize=8)
ax3.set_ylim(0, 100); ax3.legend(facecolor=BG, labelcolor="white", fontsize=7)
ax3.set_xticklabels(conf_keys, rotation=20, ha="right")
sx(ax3, "Confluence Effect on OOS Accuracy",
   "Does having more signals agreeing improve accuracy?")

# ── Panel 4: Monthly accuracy ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
session_output["month"]    = pd.to_datetime(session_output["date"]).dt.to_period("M")
session_output["fwd_ok"]   = np.where(
    session_output["direction"]=="LONG",  session_output["fwd_10s_up"],
    np.where(session_output["direction"]=="SHORT", 1-session_output["fwd_10s_up"], np.nan)
)
monthly = session_output.dropna(subset=["fwd_ok","direction"]).groupby("month").agg(
    acc=("fwd_ok","mean"), n=("fwd_ok","count")
).reset_index()
monthly = monthly[monthly["n"]>=2]
monthly["acc_pct"] = monthly["acc"]*100
cols4 = [GREEN if a>=55 else ORANGE if a>=45 else RED for a in monthly["acc_pct"]]
ax4.bar(range(len(monthly)), monthly["acc_pct"], color=cols4, alpha=0.85, edgecolor=DARK)
ax4.axhline(55, color=GOLD, linestyle=":", linewidth=1.5)
ax4.axhline(50, color=GRAY, linestyle="--", linewidth=0.8)
ax4.set_xticks(range(len(monthly)))
ax4.set_xticklabels([str(m) for m in monthly["month"]], rotation=45, ha="right", fontsize=7)
sx(ax4, "Monthly Signal Accuracy (OOS)",
   "Green ≥55% | Orange 45-55% | Red <45%")

# ── Panel 5: Signal detail table (top 15) ─────────────────────────────────
ax5 = fig.add_subplot(gs[2, :])
ax5.set_facecolor(BG); ax5.axis("off")
top_sigs = rdf[rdf["dir"]!="NEUTRAL"].dropna(subset=["oos_acc"]).sort_values("oos_acc", ascending=False)
tbl_data = []
for _, row in top_sigs.head(20).iterrows():
    d = "▲ LONG" if row["dir"]=="LONG" else "▼ SHORT"
    holds = "✅" if row["oos_acc"]>=55 else "⚠️" if row["oos_acc"]>=45 else "❌"
    drift = f"{row['drift']:+.1f}%"
    tbl_data.append([d, row["signal"][:32], row["tier"],
                     f"{row['is_acc']:.1f}%", f"{row['oos_acc']:.1f}% {holds}",
                     f"{row['oos_n']}", drift])
tbl_cols = ["Dir","Signal","Tier","IS Acc","OOS Acc","OOS N","IS→OOS Drift"]
tbl5 = ax5.table(cellText=tbl_data, colLabels=tbl_cols, loc="center", cellLoc="center",
                 colWidths=[0.07,0.3,0.1,0.09,0.12,0.07,0.11])
tbl5.auto_set_font_size(False); tbl5.set_fontsize(8)
for (r, c), cell in tbl5.get_celld().items():
    cell.set_facecolor(BG if r%2==0 else "#1c2128")
    cell.set_text_props(color="white"); cell.set_edgecolor("#30363d")
    if r==0: cell.set_text_props(color=GOLD, fontweight="bold")
ax5.set_title("★ Forward Test Results — Top 20 Signals (OOS)",
              color=GOLD, fontsize=10, fontweight="bold")

# ── Panel 6: OOS session timeline ────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, :])
dates_plot  = pd.to_datetime(session_output["date"])
dir_arr     = session_output["direction"].values
fwd_ok_arr  = session_output["fwd_ok"].values

for i, (d, ok) in enumerate(zip(dir_arr, fwd_ok_arr)):
    if d == "LONG":
        col = GREEN if ok == 1.0 else RED if ok == 0.0 else GRAY
        ax6.axvspan(i-0.4, i+0.4, ymin=0.5, ymax=1.0, color=col, alpha=0.6)
    elif d == "SHORT":
        col = GREEN if ok == 0.0 else RED if ok == 1.0 else GRAY
        ax6.axvspan(i-0.4, i+0.4, ymin=0.0, ymax=0.5, color=col, alpha=0.6)
    elif d == "NEUTRAL":
        ax6.axvspan(i-0.4, i+0.4, alpha=0.15, color=GRAY)

ax6.axhline(0.5, color=GOLD, linewidth=1.5, linestyle="--")
ax6.set_ylim(0, 1); ax6.set_xlim(-1, N_oos+1)
ax6.set_yticks([0.25, 0.75])
ax6.set_yticklabels(["LOOK SHORT", "LOOK LONG"], color=GRAY, fontsize=9)
tick_step = max(1, N_oos // 20)
ax6.set_xticks(range(0, N_oos, tick_step))
ax6.set_xticklabels([str(dates_plot.iloc[i].date()) if i<N_oos else ""
                      for i in range(0, N_oos, tick_step)],
                    rotation=45, ha="right", fontsize=6.5, color=GRAY)
sx(ax6, "OOS Session Timeline — WHEN TO LOOK FOR TRADES",
   "Top half = LOOK LONG | Bottom half = LOOK SHORT | Green=Correct | Red=Wrong | Gray=Ranging/No signal")

fig.suptitle(
    f"GOLD AI — Forward Test: 26 Signals on OOS Data ({oos['date'].iloc[0].date()} → {oos['date'].iloc[-1].date()})",
    color=GOLD, fontsize=12, fontweight="bold", y=0.999
)
plt.savefig("logs/forward_test/oos_accuracy.png", dpi=140,
            bbox_inches="tight", facecolor=DARK)
plt.close()

report_text = "\n".join(lines)
with open("logs/forward_test/signal_report.txt", "w") as f:
    f.write(report_text)

print(f"\n✅ logs/forward_test/signal_report.txt")
print(f"✅ logs/forward_test/oos_accuracy.png")
print(f"✅ logs/forward_test/session_signals.csv")
print(f"✅ logs/forward_test/WHEN_TO_LOOK_live.txt")
