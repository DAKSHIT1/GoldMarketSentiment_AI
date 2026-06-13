"""
model_train_v2.py
=================
Retrained models with:
  1. CONSISTENT labels: same ATR-based SL/TP as the backtest uses (ATR*1.0 SL, ATR*2.0 TP)
  2. CORRECT direction mapping: 0=Long, 1=Short (aligns with backtest + strategy)
  3. Session-gated training: only London (3-8) and NY (9-17) candles labelled
  4. Break-Even simulation in label generation (realistic triple-barrier)
  5. Direction model trained with class_weights to handle imbalance
  6. Saves: models/cb_meta_v2.pkl  models/cb_dir_v2.pkl

v2.1 additions:
  7. VWAP features: price_vs_vwap, dist_to_vwap_norm, vwap_slope
  8. Level interaction features: support/resistance hold, bounce strength,
     break confirmation, retest signal, level stack counts, breakout room
"""

import pandas as pd
import numpy as np
import joblib
import os
from catboost import CatBoostClassifier
from config import FEATURES_FILE

# ── Match EXACTLY with backtest_optimized.py ──────────────
SPREAD        = 0.40
ATR_SL_MULT   = 1.0
RR_RATIO      = 2.0
ATR_FALLBACK  = 5.0
WINDOW        = 20    # Max bars to hold (same as backtest)

LONDON_HOURS  = set(range(3, 8))
NY_HOURS      = set(range(9, 17))
VALID_HOURS   = LONDON_HOURS | NY_HOURS

# NOTE: session_hour / is_london_open / is_ny_open are EXCLUDED from model features
# because the backtest already gates by session manually. Including them causes the
# model to learn "trade during session hours" (84% importance!) instead of actual
# price-action setups at levels — destroying out-of-sample edge.
FEATURES = [
    # ── Core level geometry ───────────────────────────────────────────────
    "dist_to_level_norm", "dist_prev_level_norm", "dist_next_level_norm",
    "level_position", "level_rank", "nearest_level_type", "nearest_level_id",
    "level_tier_strength", "nearest_to_next_ratio",

    # ── Level interaction & reaction (key for buy/sell classification) ────
    "level_reaction_5", "level_reaction_10",
    "level_held_as_support", "level_held_as_resistance",
    "level_bounce_strength", "level_break_confirmed",
    "level_retest_signal",

    # ── Level stack / breakout room ───────────────────────────────────────
    "levels_above_count", "levels_below_count",
    "breakout_room_up", "breakout_room_down",

    # ── VWAP filter features ──────────────────────────────────────────────
    "price_vs_vwap", "dist_to_vwap_norm", "vwap_slope",

    # ── Candle & momentum ─────────────────────────────────────────────────
    "volatility", "momentum", "momentum_slope_change",
    "is_pin_bar_up", "is_pin_bar_down",
    "engulf_bull_at_level", "engulf_bear_at_level",
    "level_compression_count", "level_crossed",

    # ── Liquidity sweeps ──────────────────────────────────────────────────
    "liquidity_sweep_signal", "sweep_strength",

    # ── Volume & oscillators ──────────────────────────────────────────────
    "relative_volume", "vol_spike_at_level",
    "bear_divergence", "bull_divergence",

    # ── Trend & volatility regime ─────────────────────────────────────────
    "bb_width", "ema50_slope", "macro_trend_strength",
    "prior_day_range_pct",

    # ── Macro (DXY + Yields) ──────────────────────────────────────────────
    "dxy_return", "dxy_ema_slope", "dxy_volatility",
    "yield_return", "yield_ema_slope", "yield_volatility",
]
CAT_FEATURES = ["nearest_level_type", "nearest_level_id"]


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Triple-barrier labelling that EXACTLY mirrors the backtest execution:
      • Entry  = next_open ± spread/2
      • SL     = ATR * ATR_SL_MULT
      • TP     = SL * RR_RATIO
      • BE     = move SL to entry when price hits 1R (SL distance)
    Label conditions:
      label_meta  : 1 = either direction wins (tradeable setup)
      label_dir   : 0 = Long wins, 1 = Short wins  (only set when label_meta=1)
    Only labels candles inside London / NY session hours AND within 1.5x ATR of
    a level (dist_to_level < ATR * 1.5) to ensure we're only labelling near-level setups.
    """
    n       = len(df)
    opens   = df["open"].values
    highs   = df["high"].values
    lows    = df["low"].values
    atrs    = df["atr"].values
    hours   = df["datetime"].dt.hour.values if "datetime" in df.columns else np.zeros(n, dtype=int)
    dists   = df["dist_to_level"].values

    label_meta = np.zeros(n, dtype=np.int8)
    label_dir  = np.full(n, -1, dtype=np.int8)   # -1 = no valid label

    for i in range(n - WINDOW - 1):
        if int(hours[i]) not in VALID_HOURS:
            continue

        atr       = float(atrs[i]) if float(atrs[i]) > 0 else ATR_FALLBACK

        # Only label bars that are near a level (within 1.5x ATR)
        # This ensures the model learns level-reaction patterns, not random bars
        if float(dists[i]) > atr * 1.5:
            continue
        next_open = float(opens[i + 1])
        sl_dist   = atr * ATR_SL_MULT
        tp_dist   = sl_dist * RR_RATIO

        # ── Long setup ───────────────────────────────────────
        entry_l   = next_open + (SPREAD / 2)
        sl_l      = entry_l - sl_dist
        tp_l      = entry_l + tp_dist
        target1r_l = entry_l + sl_dist   # BE trigger

        long_win  = False
        long_loss = False
        sl_l_cur  = sl_l
        be_l      = False

        for j in range(1, WINDOW):
            idx  = i + j
            high = float(highs[idx])
            low  = float(lows[idx])
            if not be_l and high >= target1r_l:
                sl_l_cur = entry_l
                be_l     = True
            if low  <= sl_l_cur: long_loss = True; break
            if high >= tp_l:     long_win  = True; break

        # ── Short setup ──────────────────────────────────────
        entry_s    = next_open - (SPREAD / 2)
        sl_s       = entry_s + sl_dist
        tp_s       = entry_s - tp_dist
        target1r_s = entry_s - sl_dist

        short_win  = False
        short_loss = False
        sl_s_cur   = sl_s
        be_s       = False

        for j in range(1, WINDOW):
            idx  = i + j
            high = float(highs[idx])
            low  = float(lows[idx])
            if not be_s and low <= target1r_s:
                sl_s_cur = entry_s
                be_s     = True
            if high >= sl_s_cur: short_loss = True; break
            if low  <= tp_s:     short_win  = True; break

        # ── Assign labels ────────────────────────────────────
        # Meta: at least one clean win without hitting SL
        if long_win and not long_loss:
            label_meta[i] = 1
            label_dir[i]  = 0   # Long
        elif short_win and not short_loss:
            label_meta[i] = 1
            label_dir[i]  = 1   # Short
        # Both-win ambiguity → label_meta=0 (skip noisy bar)

    df = df.copy()
    df["label_meta"] = label_meta
    df["label_dir"]  = label_dir
    return df


def train():
    print("=" * 60)
    print("  GOLD AI — MODEL TRAINING v2")
    print("  Consistent ATR labels, correct direction mapping")
    print("=" * 60)

    # ── Load features ─────────────────────────────────────────
    print("\nLoading features dataset...")
    df = pd.read_csv(FEATURES_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.fillna(0).reset_index(drop=True)
    print(f"Total rows : {len(df):,}")

    # ── Walk-forward split (80 / 20) ──────────────────────────
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].copy().reset_index(drop=True)
    oos_df    = df.iloc[split_idx:].copy().reset_index(drop=True)
    print(f"Train      : {train_df['datetime'].iloc[0].date()} → {train_df['datetime'].iloc[-1].date()}  ({len(train_df):,} rows)")
    print(f"OOS        : {oos_df['datetime'].iloc[0].date()} → {oos_df['datetime'].iloc[-1].date()}  ({len(oos_df):,} rows)")

    # ── Generate labels on TRAIN only ─────────────────────────
    print("\nGenerating triple-barrier labels (train set)...")
    train_df = generate_labels(train_df)

    print(f"\nMeta label distribution (train):")
    print(train_df["label_meta"].value_counts())

    tradeable = train_df[train_df["label_meta"] == 1]
    print(f"\nDirection label distribution (tradeable train):")
    print(tradeable["label_dir"].value_counts())

    # ── Model 1: Meta-filter ──────────────────────────────────
    print("\n" + "─" * 50)
    print("Training Model 1 — Meta-Filter (tradeable vs noise)")
    print("─" * 50)
    train_clean = train_df.dropna(subset=FEATURES + ["label_meta"])

    X_meta = train_clean[FEATURES]
    y_meta = train_clean["label_meta"]

    vc = y_meta.value_counts()
    spw = float(vc.get(0, 1)) / float(vc.get(1, 1))
    print(f"Class weights — 0:{1:.2f}  1:{spw:.2f}")

    model_meta = CatBoostClassifier(
        iterations       = 800,
        depth            = 6,
        learning_rate    = 0.04,
        loss_function    = "Logloss",
        scale_pos_weight = spw,
        cat_features     = CAT_FEATURES,
        eval_metric      = "AUC",
        random_seed      = 42,
        verbose          = 200,
    )
    model_meta.fit(X_meta, y_meta)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model_meta, "models/cb_meta_v2.pkl")
    print("✅ cb_meta_v2.pkl saved")

    # ── Model 2: Direction ────────────────────────────────────
    print("\n" + "─" * 50)
    print("Training Model 2 — Direction Predictor (Long vs Short)")
    print("─" * 50)
    dir_df = train_clean[train_clean["label_meta"] == 1].copy()
    dir_df = dir_df[dir_df["label_dir"].isin([0, 1])]

    X_dir = dir_df[FEATURES]
    y_dir = dir_df["label_dir"].astype(int)

    vc_dir = y_dir.value_counts()
    spw_dir = float(vc_dir.get(0, 1)) / float(vc_dir.get(1, 1))
    print(f"Class weights — Long:{1:.2f}  Short:{spw_dir:.2f}")

    model_dir = CatBoostClassifier(
        iterations       = 600,
        depth            = 5,
        learning_rate    = 0.04,
        loss_function    = "Logloss",
        scale_pos_weight = spw_dir,
        cat_features     = CAT_FEATURES,
        eval_metric      = "AUC",
        random_seed      = 42,
        verbose          = 200,
    )
    model_dir.fit(X_dir, y_dir)
    joblib.dump(model_dir, "models/cb_dir_v2.pkl")
    print("✅ cb_dir_v2.pkl saved")

    # ── Feature importances ───────────────────────────────────
    print("\n--- Top 15 Feature Importances (Meta-Filter) ---")
    fi_meta = pd.Series(model_meta.get_feature_importance(), index=FEATURES).sort_values(ascending=False)
    print(fi_meta.head(15).to_string())

    print("\n--- Top 15 Feature Importances (Direction) ---")
    fi_dir = pd.Series(model_dir.get_feature_importance(), index=FEATURES).sort_values(ascending=False)
    print(fi_dir.head(15).to_string())

    print("\n✅ Training complete. Models saved: models/cb_meta_v2.pkl  models/cb_dir_v2.pkl")
    print("   Run backtest_optimized_v2.py to evaluate on OOS data.")


if __name__ == "__main__":
    train()
