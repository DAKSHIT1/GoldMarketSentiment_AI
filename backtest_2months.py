"""
backtest_2months.py
===================
Runs the EXACT same strategy as backtest_final.py but specifically
on the LAST 2 MONTHS of available features data.

Data range: ~Jan 23, 2026 → Mar 24, 2026
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import FEATURES_FILE

# ── Parameters (identical to backtest_final.py) ───────────────
INITIAL_BALANCE    = 10_000.0
RISK_PCT           = 0.02
SPREAD             = 0.40
ATR_FALLBACK       = 5.0
ATR_SL_FALLBACK    = 0.8
ATR_TP_FALLBACK    = 1.2
SL_BUFFER_ATR      = 0.15
MIN_META_CONF      = 0.52
MIN_DIR_CONF       = 0.55
MAX_HOLD_BARS      = 20
MAX_CONSEC_LOSSES  = 3
MAX_DAILY_DD_PCT   = 0.04

MIN_STRUCT_DIST    = 0.0002
MAX_STRUCT_DIST    = 0.05

LONDON_HOURS = set(range(3, 8))
NY_HOURS     = set(range(9, 17))
VALID_HOURS  = LONDON_HOURS | NY_HOURS

FEATURES = [
    "dist_to_level_norm", "dist_prev_level_norm", "dist_next_level_norm",
    "level_position", "level_rank", "nearest_level_type", "nearest_level_id",
    "level_tier_strength", "nearest_to_next_ratio",
    "level_reaction_5", "level_reaction_10",
    "level_held_as_support", "level_held_as_resistance",
    "level_bounce_strength", "level_break_confirmed",
    "level_retest_signal",
    "levels_above_count", "levels_below_count",
    "breakout_room_up", "breakout_room_down",
    "price_vs_vwap", "dist_to_vwap_norm", "vwap_slope",
    "volatility", "momentum", "momentum_slope_change",
    "is_pin_bar_up", "is_pin_bar_down",
    "engulf_bull_at_level", "engulf_bear_at_level",
    "level_compression_count", "level_crossed",
    "liquidity_sweep_signal", "sweep_strength",
    "relative_volume", "vol_spike_at_level",
    "bear_divergence", "bull_divergence",
    "bb_width", "ema50_slope", "macro_trend_strength",
    "prior_day_range_pct",
    "dxy_return", "dxy_ema_slope", "dxy_volatility",
    "yield_return", "yield_ema_slope", "yield_volatility",
]


def session_tag(hour: int) -> str:
    if hour in LONDON_HOURS: return "London"
    if hour in NY_HOURS:     return "NY"
    return "Off"


def max_drawdown(equity: list) -> float:
    peak, mdd = 0.0, 0.0
    for v in equity:
        if v > peak: peak = v
        dd = peak - v
        if dd > mdd: mdd = dd
    return mdd


def get_structural_targets(row, direction: str, atr: float) -> tuple:
    close_price = float(row["close"])
    dist_prev   = float(row["dist_prev_level"])
    dist_next   = float(row["dist_next_level"])

    def is_valid(d):
        return (MIN_STRUCT_DIST * close_price) < d < (MAX_STRUCT_DIST * close_price)

    if direction == "LONG":
        sl_dist = (dist_prev + atr * SL_BUFFER_ATR) if is_valid(dist_prev) else (atr * ATR_SL_FALLBACK)
        tp_dist = dist_next if is_valid(dist_next) else (atr * ATR_TP_FALLBACK)
    else:
        sl_dist = (dist_next + atr * SL_BUFFER_ATR) if is_valid(dist_next) else (atr * ATR_SL_FALLBACK)
        tp_dist = dist_prev if is_valid(dist_prev) else (atr * ATR_TP_FALLBACK)

    if tp_dist < sl_dist * 0.8:
        tp_dist = sl_dist * 1.0

    return sl_dist, tp_dist


def backtest_2months():
    print("=" * 70)
    print("  GOLD AI — 2-MONTH BACKTEST (Structure-Based SL/TP)")
    print("  Period : Last 2 months of available data")
    print("  Session: London (03-08 EST)  +  NY (09-17 EST)")
    print("  Rule   : ONE TRADE AT A TIME")
    print(f"  Risk   : {RISK_PCT*100:.0f}% per trade  |  BE@1R  |  Structure SL/TP")
    print(f"  Meta≥  : {MIN_META_CONF}  |  Dir≥: {MIN_DIR_CONF}")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────
    print("\nLoading features...")
    df = pd.read_csv(FEATURES_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.fillna(0).reset_index(drop=True)

    # ── Filter to last 2 months ────────────────────────────────
    data_end = df["datetime"].iloc[-1]
    cutoff   = data_end - pd.Timedelta(days=60)
    test_df  = df[df["datetime"] >= cutoff].copy().reset_index(drop=True)

    print(f"Full data : {df['datetime'].iloc[0]}  →  {data_end}")
    print(f"2-month   : {test_df['datetime'].iloc[0]}  →  {test_df['datetime'].iloc[-1]}")
    print(f"Bars      : {len(test_df):,}  |  Avg ATR: {test_df['atr'].mean():.2f}")

    # ── Load models ───────────────────────────────────────────
    meta_path = "models/cb_meta_v2.pkl"
    dir_path  = "models/cb_dir_v2.pkl"
    if not os.path.exists(meta_path) or not os.path.exists(dir_path):
        print("❌  Run model_train_v2.py first!")
        return

    print("\nLoading v2 models...")
    model_meta = joblib.load(meta_path)
    model_dir  = joblib.load(dir_path)

    X_test     = test_df[FEATURES]
    meta_preds = model_meta.predict(X_test)
    meta_proba = model_meta.predict_proba(X_test)[:, 1]
    dir_proba  = model_dir.predict_proba(X_test)
    dir_preds  = model_dir.predict(X_test)

    print("Running simulation...\n")

    # ── State ─────────────────────────────────────────────────
    balance      = INITIAL_BALANCE
    equity_curve = [balance]
    equity_dates = [test_df["datetime"].iloc[0]]
    results      = []

    trade_open        = False
    current_day       = None
    day_start_bal     = balance
    day_consec_losses = 0
    day_paused        = False
    skipped_session   = 0
    skipped_conf      = 0
    skipped_pa        = 0
    skipped_mid       = 0

    n = len(test_df) - MAX_HOLD_BARS
    i = 0
    while i < n:
        row  = test_df.iloc[i]
        dt   = row["datetime"]
        hour = dt.hour
        day  = dt.date()

        if day != current_day:
            current_day       = day
            day_start_bal     = balance
            day_consec_losses = 0
            day_paused        = False

        if day_paused:               i += 1; continue
        if hour not in VALID_HOURS:  skipped_session += 1; i += 1; continue
        if trade_open:               i += 1; continue

        if meta_preds[i] != 1:              skipped_conf += 1; i += 1; continue
        if meta_proba[i] < MIN_META_CONF:   skipped_conf += 1; i += 1; continue

        dir_conf = float(max(dir_proba[i]))
        if dir_conf < MIN_DIR_CONF:         skipped_conf += 1; i += 1; continue

        is_london = bool(row["is_london_open"])
        is_ny     = bool(row["is_ny_open"])
        if not (is_london or is_ny):        skipped_session += 1; i += 1; continue

        is_pin     = bool(row["is_pin_bar_up"]) or bool(row["is_pin_bar_down"])
        level_x    = bool(row["level_crossed"])
        if not (is_pin or level_x):         skipped_pa += 1; i += 1; continue

        if "Mid" in str(row["nearest_level_id"]): skipped_mid += 1; i += 1; continue

        # ── Build trade ───────────────────────────────────────
        direction = "LONG" if int(dir_preds[i]) == 0 else "SHORT"
        atr       = float(row["atr"]) if float(row["atr"]) > 0 else ATR_FALLBACK
        next_open = float(test_df.iloc[i + 1]["open"])

        sl_dist, tp_dist = get_structural_targets(row, direction, atr)

        if direction == "LONG":
            entry     = next_open + SPREAD / 2
            sl        = entry - sl_dist
            tp        = entry + tp_dist
            target_1r = entry + sl_dist
        else:
            entry     = next_open - SPREAD / 2
            sl        = entry + sl_dist
            tp        = entry - tp_dist
            target_1r = entry - sl_dist

        rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

        # ── Simulate ──────────────────────────────────────────
        trade_open   = True
        win = loss   = False
        be_triggered = False
        sl_cur       = sl
        exit_bar     = i + 1

        for j in range(1, MAX_HOLD_BARS):
            idx  = i + j
            high = float(test_df.iloc[idx]["high"])
            low  = float(test_df.iloc[idx]["low"])

            if direction == "LONG":
                if not be_triggered and high >= target_1r:
                    sl_cur = entry; be_triggered = True
                if low  <= sl_cur: loss = True; exit_bar = idx; break
                if high >= tp:     win  = True; exit_bar = idx; break
            else:
                if not be_triggered and low <= target_1r:
                    sl_cur = entry; be_triggered = True
                if high >= sl_cur: loss = True; exit_bar = idx; break
                if low  <= tp:     win  = True; exit_bar = idx; break

        timed_out = not win and not loss
        if timed_out:
            exit_bar    = i + MAX_HOLD_BARS - 1
            exit_price  = float(test_df.iloc[exit_bar]["close"])
            pnl_pts     = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            exit_reason = "TIMEOUT"
        else:
            if direction == "LONG":
                pnl_pts = (tp - entry) if win else (sl_cur - entry)
            else:
                pnl_pts = (entry - tp) if win else (entry - sl_cur)
            if win:
                exit_reason = "TP"
            elif abs(pnl_pts) < 0.01:
                exit_reason = "BE"
            else:
                exit_reason = "SL"

        risk_amt = balance * RISK_PCT
        pnl_usd  = (risk_amt / sl_dist) * pnl_pts if sl_dist > 0 else 0.0
        balance  += pnl_usd

        if pnl_usd < -0.01:
            day_consec_losses += 1
        elif pnl_usd > 0.01:
            day_consec_losses = 0

        daily_dd = (day_start_bal - balance) / day_start_bal
        if day_consec_losses >= MAX_CONSEC_LOSSES or daily_dd >= MAX_DAILY_DD_PCT:
            day_paused = True

        rr_act = abs(pnl_pts / sl_dist) if sl_dist > 0 else 0.0

        results.append({
            "entry_datetime" : test_df.iloc[i]["datetime"],
            "exit_datetime"  : test_df.iloc[exit_bar]["datetime"],
            "session"        : session_tag(hour),
            "direction"      : direction,
            "entry"          : round(entry, 2),
            "sl"             : round(sl, 2),
            "tp"             : round(tp, 2),
            "sl_dist"        : round(sl_dist, 3),
            "tp_dist"        : round(tp_dist, 3),
            "rr_planned"     : round(rr, 3),
            "exit_reason"    : exit_reason,
            "be_triggered"   : be_triggered,
            "pnl_pts"        : round(pnl_pts, 3),
            "pnl_usd"        : round(pnl_usd, 2),
            "balance"        : round(balance, 2),
            "rr_actual"      : round(rr_act, 3),
            "meta_conf"      : round(float(meta_proba[i]), 4),
            "dir_conf"       : round(dir_conf, 4),
        })
        equity_curve.append(balance)
        equity_dates.append(test_df.iloc[exit_bar]["datetime"])

        trade_open = False
        i          = exit_bar + 1

    # ─────────────────────────────────────────────────────────
    #  METRICS
    # ─────────────────────────────────────────────────────────
    if not results:
        print("❌ No trades fired in the 2-month window.")
        return

    res = pd.DataFrame(results)
    res["is_win"]  = res["pnl_usd"] > 0.01
    res["is_loss"] = res["pnl_usd"] < -0.01
    res["is_be"]   = res["pnl_usd"].abs() <= 0.01

    wins   = int(res["is_win"].sum())
    losses = int(res["is_loss"].sum())
    bes    = int(res["is_be"].sum())
    total  = len(res)

    gp  = res[res["is_win"]]["pnl_usd"].sum()
    gl  = abs(res[res["is_loss"]]["pnl_usd"].sum())
    pf  = gp / gl if gl > 0 else 999.0
    mdd = max_drawdown(equity_curve)
    mdd_pct      = (mdd / INITIAL_BALANCE) * 100
    total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100

    # ── Weekly breakdown ──────────────────────────────────────
    res["week"] = res["entry_datetime"].dt.isocalendar().week.astype(int)
    res["year_week"] = res["entry_datetime"].dt.strftime("%Y-W%U")

    def session_stats(df_s, name):
        if len(df_s) == 0: print(f"  {name:7s}: No trades"); return
        w=int(df_s["is_win"].sum()); l=int(df_s["is_loss"].sum()); b=int(df_s["is_be"].sum()); t=len(df_s)
        pnl=df_s["pnl_usd"].sum(); avg_rr=df_s["rr_planned"].mean()
        print(f"  {name:7s}: {t:3d} trades | WR {w/t*100:5.1f}% ({w}W/{b}BE/{l}L) | AvgRR {avg_rr:.2f} | PnL ${pnl:+,.2f}")

    print("\n" + "=" * 70)
    print("   2-MONTH BACKTEST RESULTS — Structure SL/TP")
    print("=" * 70)
    print(f"  Period       : {res['entry_datetime'].iloc[0].date()} → {res['entry_datetime'].iloc[-1].date()}")
    print(f"  Total Trades : {total}")
    print(f"  Win Rate     : {wins/total*100:.1f}%  ({wins}W / {bes}BE / {losses}L)")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg RR       : {res['rr_planned'].mean():.2f}  (planned)  |  {res['rr_actual'].mean():.2f} (actual)")
    print(f"  Total Return : {total_return:+.2f}%  (${balance:,.2f})")
    print(f"  Max Drawdown : ${mdd:,.2f}  ({mdd_pct:.1f}%)")
    print(f"  Starting Bal : ${INITIAL_BALANCE:,.2f}")
    print("-" * 70)
    print("  SESSION BREAKDOWN:")
    session_stats(res[res["session"] == "London"], "London")
    session_stats(res[res["session"] == "NY"],     "NY    ")
    print("-" * 70)
    print("  EXIT BREAKDOWN:")
    for reason, grp in res.groupby("exit_reason"):
        print(f"    {reason:8s}: {len(grp):3d} trades | PnL ${grp['pnl_usd'].sum():+,.2f}")
    print("-" * 70)
    print("  DIRECTION BREAKDOWN:")
    for d, grp in res.groupby("direction"):
        w_d = int(grp["is_win"].sum()); t_d = len(grp)
        print(f"    {d:5s}: {t_d:3d} trades | WR {w_d/t_d*100:5.1f}% | PnL ${grp['pnl_usd'].sum():+,.2f}")
    print("=" * 70)

    # ── Weekly breakdown ──────────────────────────────────────
    print("\n📅 WEEKLY BREAKDOWN:")
    print(f"  {'Week':12s} | {'Trades':>6s} | {'WR':>6s} | {'PnL':>10s} | {'Cum.Bal':>10s}")
    print("  " + "-" * 55)
    running_bal = INITIAL_BALANCE
    for wk, grp in res.groupby("year_week", sort=True):
        w_wk = int(grp["is_win"].sum()); t_wk = len(grp)
        pnl_wk = grp["pnl_usd"].sum()
        running_bal += pnl_wk
        wr_wk = w_wk/t_wk*100 if t_wk > 0 else 0
        print(f"  {wk:12s} | {t_wk:6d} | {wr_wk:5.1f}% | ${pnl_wk:+9,.2f} | ${running_bal:>9,.2f}")

    # ── All trades ────────────────────────────────────────────
    print(f"\n📋 ALL {total} TRADES:")
    print(res[["entry_datetime","session","direction","entry","sl","tp",
               "sl_dist","tp_dist","rr_planned","exit_reason","be_triggered",
               "pnl_pts","pnl_usd","balance","meta_conf","dir_conf"]].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/backtest_2months_results.csv"
    res.to_csv(csv_path, index=False)
    print(f"\n✅ Trade log → {csv_path}")

    # ── Equity Chart ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values(): spine.set_color("#30363d")
        ax.tick_params(colors="#c9d1d9")

    eq_arr = np.array(equity_curve)
    ax1.plot(equity_dates, eq_arr, color="#2dba76", linewidth=1.8, zorder=3)
    ax1.fill_between(equity_dates, eq_arr, INITIAL_BALANCE,
                     where=(eq_arr >= INITIAL_BALANCE), alpha=0.25, color="#2dba76")
    ax1.fill_between(equity_dates, eq_arr, INITIAL_BALANCE,
                     where=(eq_arr < INITIAL_BALANCE),  alpha=0.30, color="#f23645")
    ax1.axhline(INITIAL_BALANCE, color="#8b949e", linewidth=0.8, linestyle="--")

    title_str = (
        f"Gold AI — 2-Month Backtest  |  {res['entry_datetime'].iloc[0].date()} → {res['entry_datetime'].iloc[-1].date()}  |  "
        f"Return: {total_return:+.1f}%  |  MaxDD: {mdd_pct:.1f}%  |  WR: {wins/total*100:.1f}%  |  "
        f"PF: {pf:.2f}  |  Trades: {total}"
    )
    ax1.set_title(title_str, color="#f0f6fc", fontsize=11, pad=10)
    ax1.set_ylabel("Balance ($)", color="#8b949e")

    clrs = []
    for _, r in res.iterrows():
        if r["pnl_usd"] > 0.01:   clrs.append("#2dba76")
        elif abs(r["pnl_usd"]) <= 0.01: clrs.append("#f5a623")
        else:                       clrs.append("#f23645")
    ax2.bar(res["entry_datetime"], res["pnl_usd"], color=clrs, width=0.01, alpha=0.85)
    ax2.axhline(0, color="#8b949e", linewidth=0.6)
    ax2.set_ylabel("Trade PnL ($)", color="#8b949e")
    ax2.set_xlabel("Date", color="#8b949e")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2dba76", label="Win"),
        Patch(facecolor="#f5a623", label="BE"),
        Patch(facecolor="#f23645", label="Loss"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", facecolor="#21262d", labelcolor="#c9d1d9", framealpha=0.8)

    plt.tight_layout(pad=1.5)
    img_path = "logs/equity_2months.png"
    plt.savefig(img_path, dpi=160, facecolor="#0d1117")
    plt.close()
    print(f"✅ Equity curve → {img_path}")

    # Print signal funnel
    print(f"\n📊 Signal Funnel:")
    print(f"  Total bars in 2-month window: {len(test_df):,}")
    print(f"  Session bars skipped (off-hours): {skipped_session}")
    print(f"  Confidence filtered:              {skipped_conf}")
    print(f"  No PA confirmation:               {skipped_pa}")
    print(f"  Mid-level filtered:               {skipped_mid}")
    print(f"  Trades executed:                  {total}")


if __name__ == "__main__":
    backtest_2months()
