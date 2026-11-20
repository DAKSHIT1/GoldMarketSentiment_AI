import pandas as pd
import numpy as np
from config import *
from level_engine import MathLevelEngine
from session_builder import session_start_flags


def _new_session(df):
    """First bar of each trading-day session (handles sessions that open after 18:00)."""
    if "datetime" in df.columns:
        return session_start_flags(df["datetime"])
    f = np.zeros(len(df), dtype=bool)
    if len(f):
        f[0] = True
    return f

# ──────────────────────────────────────────────────────────────
#  VWAP HELPERS
# ──────────────────────────────────────────────────────────────

def _compute_session_vwap(df: pd.DataFrame) -> np.ndarray:
    """
    Session-anchored VWAP using typical price (H+L+C)/3.
    Volume = 0 for XAUUSD tick data, so equal-weight price VWAP is used.
    Resets every day at 18:00 EST (NY session open).
    """
    typical = ((df["high"] + df["low"] + df["close"]) / 3).values
    new_sess = _new_session(df)

    vwap_arr    = np.full(len(df), np.nan)
    cum_tp      = 0.0
    count       = 0

    for i in range(len(df)):
        # Reset at the session open (first bar of the trading day)
        if new_sess[i]:
            cum_tp = 0.0
            count  = 0
        cum_tp += typical[i]
        count  += 1
        vwap_arr[i] = cum_tp / count

    return vwap_arr


# ──────────────────────────────────────────────────────────────
#  LEVEL INTERACTION HELPERS
# ──────────────────────────────────────────────────────────────

def _level_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes how price *has reacted* after touching a level (look-back only — no leakage).

    Features:
      level_reaction_5        Net close move over previous 5 bars relative to ATR
      level_reaction_10       Net close move over previous 10 bars relative to ATR
      level_held_as_support   Price crossed level upward and stayed above for 3 bars
      level_held_as_resistance Price crossed level downward and stayed below for 3 bars
      level_bounce_strength   Bounce size off nearest level relative to ATR
      level_break_confirmed   2 consecutive closes firmly outside the level
      level_retest_signal     Returned within 0.5 ATR of a broken level after 1 ATR move
      nearest_to_next_ratio   dist_to_level / dist_next_level (relative position between levels)
      levels_above_count      How many active resistance levels sit above price
      levels_below_count      How many active support levels sit below price
      breakout_room_up        Dist to next 3 levels above, normalised by ATR (avg)
      breakout_room_down      Dist to next 3 levels below, normalised by ATR (avg)
    """
    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    levels  = df["nearest_level"].values
    atrs    = df["atr"].values.clip(1e-6)
    dist    = df["dist_to_level"].values
    dist_n  = df["dist_next_level"].values
    dist_p  = df["dist_prev_level"].values
    n       = len(df)

    # --- Look-back reaction over windows ---
    react5  = np.zeros(n)
    react10 = np.zeros(n)
    for i in range(10, n):
        react5[i]  = (closes[i] - closes[i - 5])  / atrs[i]
        react10[i] = (closes[i] - closes[i - 10]) / atrs[i]

    # --- Support / Resistance hold (3-bar confirmation look-back) ---
    held_sup = np.zeros(n, dtype=np.int8)
    held_res = np.zeros(n, dtype=np.int8)
    for i in range(3, n):
        lvl = levels[i]
        # Price is currently above the level (potential support)
        if closes[i] > lvl:
            if closes[i-1] > lvl and closes[i-2] > lvl:
                # Check if there was a cross from below within last 5 bars
                for k in range(3, min(6, i)):
                    if closes[i - k] < lvl:
                        held_sup[i] = 1
                        break
        # Price is currently below the level (potential resistance)
        elif closes[i] < lvl:
            if closes[i-1] < lvl and closes[i-2] < lvl:
                for k in range(3, min(6, i)):
                    if closes[i - k] > lvl:
                        held_res[i] = 1
                        break

    # --- Bounce strength ---
    bounce_str = np.zeros(n)
    for i in range(5, n):
        lvl    = levels[i]
        atr    = atrs[i]
        # Recent low close to a support level → measure bounce
        min_low  = lows[i - 5 : i].min()
        max_high = highs[i - 5 : i].max()
        if abs(min_low - lvl) < atr * 0.5:
            bounce_str[i] = (closes[i] - min_low) / atr          # upward bounce
        elif abs(max_high - lvl) < atr * 0.5:
            bounce_str[i] = (max_high - closes[i]) / atr          # downward bounce

    # --- Break confirmed: 2 consecutive closes clearly outside level ---
    brk_conf = np.zeros(n, dtype=np.int8)
    for i in range(2, n):
        lvl = levels[i]
        atr = atrs[i]
        thresh = atr * 0.25
        if closes[i] > lvl + thresh and closes[i-1] > lvl + thresh:
            brk_conf[i] = 1   # broken upward
        elif closes[i] < lvl - thresh and closes[i-1] < lvl - thresh:
            brk_conf[i] = -1  # broken downward

    # --- Retest signal: price returns to broken level after 1 ATR away ---
    retest = np.zeros(n, dtype=np.int8)
    for i in range(10, n):
        lvl = levels[i]
        atr = atrs[i]
        # Only if a break was confirmed in the recent past
        was_broken = False
        for k in range(2, 8):
            if brk_conf[i - k] != 0:
                was_broken = True
                break
        if was_broken and dist[i] < atr * 0.5:
            retest[i] = 1

    # --- Nearest-to-next ratio ---
    nn_ratio = dist / (dist_n + 1e-6)

    df = df.copy()
    df["level_reaction_5"]        = react5
    df["level_reaction_10"]       = react10
    df["level_held_as_support"]   = held_sup.astype(int)
    df["level_held_as_resistance"]= held_res.astype(int)
    df["level_bounce_strength"]   = bounce_str
    df["level_break_confirmed"]   = brk_conf.astype(int)
    df["level_retest_signal"]     = retest.astype(int)
    df["nearest_to_next_ratio"]   = nn_ratio.clip(0, 10)

    return df


# ──────────────────────────────────────────────────────────────
#  LEVEL SEQUENCE PROGRESSION ENGINE
# ──────────────────────────────────────────────────────────────

def _level_sequence_features(df: pd.DataFrame, engine: MathLevelEngine) -> pd.DataFrame:
    """
    Tracks HOW price moves through consecutive levels after a break.

    Example logic:
      - Price breaks U1 → does it reach U2? U3? U4?
      - Or does it retrace back to U1 after touching U3?
      - Same logic for downside: L1 → L2 → L3 → retrace?

    All features are LOOK-BACK only (no data leakage). They describe
    what happened in the recent past, which the model uses to predict
    what will happen next.

    Features produced:
      seq_levels_broken_up       : consecutive upper levels broken above Open (current streak)
      seq_levels_broken_down     : consecutive lower levels broken below Open (current streak)
      seq_max_levels_up          : max upper levels ever broken in this session
      seq_max_levels_down        : max lower levels ever broken in this session
      seq_retrace_depth          : how many levels price has GIVEN BACK from its session extreme
      seq_retrace_fraction       : retrace_depth / max_levels_reached  (0=no retrace, 1=full retrace)
      seq_level_velocity_up      : avg bars spent between consecutive upward level breaks (speed)
      seq_level_velocity_down    : avg bars spent between consecutive downward level breaks
      seq_at_level_N             : which numbered level price is currently at (e.g. 3 = U3 or L3)
      seq_continuation_signal    : +1 if breaking a level with momentum (velocity < 4 bars/level)
      seq_retrace_signal         : +1 if currently retracing after breaking ≥2 levels
      seq_failed_break_up        : price broke U_N but closed BACK below it (false breakout up)
      seq_failed_break_down      : price broke L_N but closed BACK above it (false breakout down)
      seq_level_gap_to_next_up   : distance to next unbroken upper level, normalised by ATR
      seq_level_gap_to_next_down : distance to next unbroken lower level, normalised by ATR
    """
    n       = len(df)
    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    opens_  = df["open"].values
    atrs    = df["atr"].values.clip(1e-6)
    new_sess = _new_session(df)

    # Output arrays
    broken_up       = np.zeros(n)
    broken_down     = np.zeros(n)
    max_up          = np.zeros(n)
    max_down        = np.zeros(n)
    retrace_depth   = np.zeros(n)
    retrace_frac    = np.zeros(n)
    vel_up          = np.zeros(n)
    vel_down        = np.zeros(n)
    at_level_n      = np.zeros(n)
    cont_signal     = np.zeros(n, dtype=np.int8)
    retrace_signal  = np.zeros(n, dtype=np.int8)
    failed_up       = np.zeros(n, dtype=np.int8)
    failed_down     = np.zeros(n, dtype=np.int8)
    gap_next_up     = np.zeros(n)
    gap_next_down   = np.zeros(n)

    # Session state — resets at 18:00
    sess_open        = None          # session open price
    upper_lvls       = []            # ordered upper level prices [U1, U2, U3, ...]
    lower_lvls       = []            # ordered lower level prices [L1, L2, L3, ...]
    sess_max_up      = 0             # max upper level index broken this session (0 = none)
    sess_max_down    = 0             # max lower level index broken this session (0 = none)
    bar_when_broke_up   = {}         # level_idx -> bar index when it was first broken
    bar_when_broke_down = {}

    for i in range(n):
        # ── Session reset ──────────────────────────────────────────────────
        is_new_session = engine.period_open is None or new_sess[i]
        if is_new_session:
            engine.reset_levels(opens_[i])
            sess_open       = opens_[i]
            all_lvls        = engine.get_all_levels()
            upper_lvls      = sorted(
                [l["price"] for l in all_lvls if l["id"].startswith("U") and "_" not in l["id"]]
            )
            lower_lvls      = sorted(
                [l["price"] for l in all_lvls if l["id"].startswith("L") and "_" not in l["id"]],
                reverse=True   # L1 is closest to open
            )
            sess_max_up     = 0
            sess_max_down   = 0
            bar_when_broke_up   = {}
            bar_when_broke_down = {}

        if not upper_lvls or sess_open is None:
            continue

        close   = closes[i]
        high    = highs[i]
        low     = lows[i]
        atr     = atrs[i]

        # ── Count how many upper levels price has closed above ─────────────
        cur_up = 0
        for idx, lvl_price in enumerate(upper_lvls):
            if close > lvl_price:
                cur_up = idx + 1
                if (idx + 1) > sess_max_up:
                    sess_max_up = idx + 1
                    bar_when_broke_up[idx + 1] = i
            else:
                break

        # ── Count how many lower levels price has closed below ─────────────
        cur_down = 0
        for idx, lvl_price in enumerate(lower_lvls):
            if close < lvl_price:
                cur_down = idx + 1
                if (idx + 1) > sess_max_down:
                    sess_max_down = idx + 1
                    bar_when_broke_down[idx + 1] = i
            else:
                break

        broken_up[i]   = cur_up
        broken_down[i] = cur_down
        max_up[i]      = sess_max_up
        max_down[i]    = sess_max_down

        # ── Retrace depth: how many levels were given back from session peak ─
        if cur_up > 0:
            # Currently in upward breakout: retrace = how many levels below peak
            rd  = sess_max_up - cur_up
            rf  = rd / sess_max_up if sess_max_up > 0 else 0.0
            retrace_depth[i] = rd
            retrace_frac[i]  = rf
            at_level_n[i]    = cur_up
        elif cur_down > 0:
            # Currently in downward breakout
            rd  = sess_max_down - cur_down
            rf  = rd / sess_max_down if sess_max_down > 0 else 0.0
            retrace_depth[i] = rd
            retrace_frac[i]  = rf
            at_level_n[i]    = -cur_down   # negative = downside level
        else:
            # Price between Open and U1/L1
            at_level_n[i] = 0

        # ── Velocity: avg bars per level break (upward) ─────────────────────
        if cur_up >= 2:
            bar_times = [bar_when_broke_up.get(k, i) for k in range(1, cur_up + 1)]
            diffs = [bar_times[j] - bar_times[j-1] for j in range(1, len(bar_times)) if bar_times[j] > bar_times[j-1]]
            vel_up[i] = float(np.mean(diffs)) if diffs else 0.0
        if cur_down >= 2:
            bar_times = [bar_when_broke_down.get(k, i) for k in range(1, cur_down + 1)]
            diffs = [bar_times[j] - bar_times[j-1] for j in range(1, len(bar_times)) if bar_times[j] > bar_times[j-1]]
            vel_down[i] = float(np.mean(diffs)) if diffs else 0.0

        # ── Continuation signal: fast multi-level break (velocity ≤ 4 bars/level) ─
        if cur_up >= 2 and vel_up[i] > 0 and vel_up[i] <= 4:
            cont_signal[i] = 1
        elif cur_down >= 2 and vel_down[i] > 0 and vel_down[i] <= 4:
            cont_signal[i] = -1

        # ── Retrace signal: gave back ≥1 level after breaking ≥2 ────────────
        if sess_max_up >= 2 and retrace_depth[i] >= 1:
            retrace_signal[i] = 1     # retracing from upside extension
        elif sess_max_down >= 2 and retrace_depth[i] >= 1:
            retrace_signal[i] = -1    # retracing from downside extension

        # ── Failed break: high pierced next level but close came back ────────
        # Upside: high went above the next unbroken upper level but close stayed below it
        if cur_up < len(upper_lvls):
            next_up_price = upper_lvls[cur_up]   # the first unbroken level
            if high >= next_up_price and close < next_up_price:
                failed_up[i] = 1

        # Downside: low went below the next unbroken lower level but close stayed above it
        if cur_down < len(lower_lvls):
            next_down_price = lower_lvls[cur_down]
            if low <= next_down_price and close > next_down_price:
                failed_down[i] = 1

        # ── Gap to next unbroken level (normalised by ATR) ──────────────────
        if cur_up < len(upper_lvls):
            gap_next_up[i]   = (upper_lvls[cur_up] - close) / atr
        if cur_down < len(lower_lvls):
            gap_next_down[i] = (close - lower_lvls[cur_down]) / atr

    df = df.copy()
    df["seq_levels_broken_up"]       = broken_up
    df["seq_levels_broken_down"]     = broken_down
    df["seq_max_levels_up"]          = max_up
    df["seq_max_levels_down"]        = max_down
    df["seq_retrace_depth"]          = retrace_depth
    df["seq_retrace_fraction"]       = retrace_frac
    df["seq_level_velocity_up"]      = vel_up
    df["seq_level_velocity_down"]    = vel_down
    df["seq_at_level_n"]             = at_level_n
    df["seq_continuation_signal"]    = cont_signal.astype(int)
    df["seq_retrace_signal"]         = retrace_signal.astype(int)
    df["seq_failed_break_up"]        = failed_up.astype(int)
    df["seq_failed_break_down"]      = failed_down.astype(int)
    df["seq_gap_to_next_up"]         = gap_next_up.clip(0, 20)
    df["seq_gap_to_next_down"]       = gap_next_down.clip(0, 20)

    return df


def _level_stack_features(df: pd.DataFrame, engine: MathLevelEngine) -> pd.DataFrame:
    """
    For each bar, count levels above/below price and compute breakout room
    (avg distance to next 3 levels in each direction), normalised by ATR.
    """
    closes = df["close"].values
    atrs   = df["atr"].values.clip(1e-6)
    n      = len(df)

    lv_above   = np.zeros(n)
    lv_below   = np.zeros(n)
    room_up    = np.zeros(n)
    room_down  = np.zeros(n)

    new_sess = _new_session(df)

    for i in range(n):
        if new_sess[i]:
            engine.reset_levels(df["open"].values[i])

        all_levels = engine.get_all_levels()
        if not all_levels:
            continue

        prices = np.array([lvl["price"] for lvl in all_levels])
        price  = closes[i]
        atr    = atrs[i]

        above = prices[prices > price]
        below = prices[prices < price]

        lv_above[i] = len(above)
        lv_below[i] = len(below)

        if len(above) >= 3:
            room_up[i] = np.sort(above)[:3].mean() / atr - price / atr
        elif len(above) > 0:
            room_up[i] = (above.min() - price) / atr

        if len(below) >= 3:
            room_down[i] = price / atr - np.sort(below)[-3:].mean() / atr
        elif len(below) > 0:
            room_down[i] = (price - below.max()) / atr

    df = df.copy()
    df["levels_above_count"] = lv_above
    df["levels_below_count"] = lv_below
    df["breakout_room_up"]   = room_up.clip(0, 20)
    df["breakout_room_down"] = room_down.clip(0, 20)
    return df


# ──────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def generate_features(df15, is_live=False):
    """
    Core Feature Engineering Pipeline.
    Designed to process either a massive historical 15m dataset (for training),
    or a 100-candle live websocket array (for real-time inference).
    """

    if "datetime" in df15.columns and not pd.api.types.is_datetime64_any_dtype(df15["datetime"]):
        df15["datetime"] = pd.to_datetime(df15["datetime"])

    for col in ["open", "high", "low", "close"]:
        df15[col] = df15[col].astype(float)

    if "volume" not in df15.columns:
        df15["volume"] = 0.0
    else:
        df15["volume"] = df15["volume"].astype(float)

    # =============================
    # NATIVE 15M PROXIES FOR 1H MTF
    # 1H = 4x 15m. So 50-period 1H = 200-period 15m
    # =============================
    df15["ema200"]          = df15["close"].ewm(span=200, adjust=False).mean()
    df15["ema50_slope"]     = (df15["ema200"] - df15["ema200"].shift(20)) / 20

    df15["std80"]           = df15["close"].rolling(80).std()
    df15["sma80"]           = df15["close"].rolling(80).mean()
    df15["bb_width"]        = (df15["std80"] * 2) / (df15["sma80"] + 1e-6)

    df15["macro_trend_strength"] = abs(df15["ema50_slope"]) / (df15["bb_width"] + 1e-6)

    # =============================
    # PINE SCRIPT MATH LEVEL ENGINE
    # =============================
    engine = MathLevelEngine(mode="Daily")

    nearest_levels = []
    level_dist     = []
    nearest_ids    = []
    nearest_types  = []
    dist_prev      = []
    dist_next      = []
    level_position = []
    level_ranks    = []

    new_sess = _new_session(df15)

    opens  = df15["open"].values
    closes = df15["close"].values

    for i in range(len(df15)):
        # Session reset at the first bar of each trading day
        if engine.period_open is None or new_sess[i]:
            engine.reset_levels(opens[i])

        res = engine.find_nearest_context(closes[i])

        nearest_levels.append(res["nearest_price"])
        level_dist.append(res["dist_to_level"])
        nearest_ids.append(res["nearest_id"])
        nearest_types.append(res["nearest_type"])
        dist_prev.append(res["dist_prev_level"])
        dist_next.append(res["dist_next_level"])
        level_position.append(res["level_position"])
        level_ranks.append(res["nearest_rank"])

    df15["nearest_level"]      = nearest_levels
    df15["dist_to_level"]      = level_dist
    df15["nearest_level_id"]   = nearest_ids
    df15["nearest_level_type"] = nearest_types
    df15["dist_prev_level"]    = dist_prev
    df15["dist_next_level"]    = dist_next
    df15["level_position"]     = level_position
    df15["level_rank"]         = level_ranks

    # =============================
    # CANDLE & MOMENTUM FEATURES
    # =============================
    df15["body"]       = abs(df15["close"] - df15["open"])
    df15["upper_wick"] = df15["high"] - df15[["open", "close"]].max(axis=1)
    df15["lower_wick"] = df15[["open", "close"]].min(axis=1) - df15["low"]

    df15["volatility"] = df15["high"] - df15["low"]
    df15["momentum"]   = df15["close"].diff()
    df15["momentum_3"] = df15["close"].diff(3)

    # ATR
    df15["tr1"] = df15["high"] - df15["low"]
    df15["tr2"] = abs(df15["high"] - df15["close"].shift()).fillna(0)
    df15["tr3"] = abs(df15["low"]  - df15["close"].shift()).fillna(0)
    df15["tr"]  = df15[["tr1", "tr2", "tr3"]].max(axis=1)
    df15["atr"] = df15["tr"].rolling(14).mean()

    # Ratios
    df15["upper_wick_ratio"] = df15["upper_wick"] / (df15["volatility"] + 1e-6)
    df15["lower_wick_ratio"] = df15["lower_wick"] / (df15["volatility"] + 1e-6)
    df15["body_ratio"]       = df15["body"]        / (df15["volatility"] + 1e-6)

    # =============================
    # PRICE PATTERNS
    # =============================
    df15["is_pin_bar_up"]   = ((df15["lower_wick_ratio"] > 0.65) & (df15["dist_to_level"] < 0.25)).astype(int)
    df15["is_pin_bar_down"] = ((df15["upper_wick_ratio"] > 0.65) & (df15["dist_to_level"] < 0.25)).astype(int)

    prev_body  = df15["body"].shift(1)
    prev_close = df15["close"].shift(1)
    prev_open  = df15["open"].shift(1)

    is_bull_engulf = (
        (df15["close"] > df15["open"]) & (prev_close < prev_open) &
        (df15["close"] > prev_open)    & (df15["open"] < prev_close)
    )
    is_bear_engulf = (
        (df15["close"] < df15["open"]) & (prev_close > prev_open) &
        (df15["close"] < prev_open)    & (df15["open"] > prev_close)
    )

    df15["engulf_bull_at_level"] = (is_bull_engulf & (df15["dist_to_level"] < 0.3)).astype(int)
    df15["engulf_bear_at_level"] = (is_bear_engulf & (df15["dist_to_level"] < 0.3)).astype(int)

    is_tiny_candle               = (df15["body_ratio"] < 0.4) & (df15["dist_to_level"] < 0.4)
    df15["level_compression_count"] = is_tiny_candle.rolling(5).sum()

    # LEVEL REACTION FEATURES (normalised)
    df15["dist_to_level_norm"]  = df15["dist_to_level"]  / (df15["atr"] + 1e-6)
    df15["dist_next_level_norm"]= df15["dist_next_level"] / (df15["atr"] + 1e-6)
    df15["dist_prev_level_norm"]= df15["dist_prev_level"] / (df15["atr"] + 1e-6)

    df15["level_crossed"] = (
        ((df15["open"] < df15["nearest_level"]) & (df15["close"] > df15["nearest_level"])) |
        ((df15["open"] > df15["nearest_level"]) & (df15["close"] < df15["nearest_level"]))
    ).astype(int)

    # =============================
    # VOLUME & OSCILLATORS
    # =============================
    df15["relative_volume"]    = df15["volume"] / (df15["volume"].rolling(20).mean() + 1e-6)
    df15["vol_spike_at_level"] = (
        (df15["relative_volume"] > 2.0) & (df15["dist_to_level_norm"] < 0.5)
    ).astype(int)

    delta = df15["close"].diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs    = gain / (loss + 1e-6)
    df15["rsi_14"] = 100 - (100 / (1 + rs))

    df15["bear_divergence"] = (
        (df15["high"] > df15["high"].shift(3)) & (df15["rsi_14"] < df15["rsi_14"].shift(3))
    ).astype(int)
    df15["bull_divergence"] = (
        (df15["low"] < df15["low"].shift(3)) & (df15["rsi_14"] > df15["rsi_14"].shift(3))
    ).astype(int)

    # =============================
    # LIQUIDITY SWEEPS
    # =============================
    df15["local_high_15"] = df15["high"].shift(1).rolling(15).max()
    df15["local_low_15"]  = df15["low"].shift(1).rolling(15).min()

    df15["bull_sweep"]            = (
        (df15["high"] > df15["local_high_15"]) & (df15["close"] < df15["local_high_15"])
    ).astype(int)
    df15["bear_sweep"]            = (
        (df15["low"] < df15["local_low_15"]) & (df15["close"] > df15["local_low_15"])
    ).astype(int)
    df15["sweep_strength"]        = df15["upper_wick_ratio"] + df15["lower_wick_ratio"]
    df15["liquidity_sweep_signal"]= df15["bull_sweep"] + df15["bear_sweep"]

    # =============================
    # SESSION FEATURES
    # =============================
    if "datetime" in df15.columns:
        df15["session_hour"]   = df15["datetime"].dt.hour.astype(int)
        df15["is_london_open"] = df15["session_hour"].between(3, 5).astype(int)
        df15["is_ny_open"]     = df15["session_hour"].between(9, 11).astype(int)
    else:
        df15["session_hour"]   = 0
        df15["is_london_open"] = 0
        df15["is_ny_open"]     = 0

    # =============================
    # VWAP (Session-Anchored, Price-Based)
    # =============================
    if "datetime" in df15.columns:
        vwap_vals            = _compute_session_vwap(df15)
        df15["vwap"]         = vwap_vals
        df15["price_vs_vwap"]= np.sign(df15["close"].values - vwap_vals)  # +1 above, -1 below
        df15["dist_to_vwap_norm"] = abs(df15["close"].values - vwap_vals) / (df15["atr"].values + 1e-6)
        # VWAP slope over last 4 bars (1 hour equivalent)
        df15["vwap_slope"]   = pd.Series(vwap_vals).diff(4).values / (df15["atr"].values + 1e-6)
    else:
        df15["vwap"]              = df15["close"]
        df15["price_vs_vwap"]     = 0
        df15["dist_to_vwap_norm"] = 0
        df15["vwap_slope"]        = 0

    # =============================
    # LEVEL TIER STRENGTH
    # =============================
    df15["level_tier_strength"] = 1.0 / (df15["level_rank"].replace(0, 1))

    # Prior 24-hour range position (96 bars = 24h of 15m)
    df15["pd_high"] = df15["high"].shift(1).rolling(96).max()
    df15["pd_low"]  = df15["low"].shift(1).rolling(96).min()
    pd_range = (df15["pd_high"] - df15["pd_low"]).replace(0, np.nan)
    df15["prior_day_range_pct"] = ((df15["close"] - df15["pd_low"]) / pd_range).clip(0, 1).fillna(0.5)

    # Momentum slope change
    df15["momentum_slope_change"] = df15["momentum"] - df15["momentum_3"]

    # =============================
    # LEVEL INTERACTION FEATURES (look-back, no leakage)
    # =============================
    df15 = _level_interaction_features(df15)

    # =============================
    # LEVEL STACK / BREAKOUT ROOM
    # =============================
    # Re-use a fresh engine instance for stack counting (stateless pass)
    engine2 = MathLevelEngine(mode="Daily")
    df15 = _level_stack_features(df15, engine2)

    # =============================
    # LEVEL SEQUENCE PROGRESSION
    # How price moves through L1→L2→L3 or U1→U2→U3 and retraces
    # =============================
    engine3 = MathLevelEngine(mode="Daily")
    df15 = _level_sequence_features(df15, engine3)

    # =============================
    # MACRO REGIME AUGMENTATION
    # =============================
    if not is_live:
        try:
            df_uup = pd.read_csv("data/uup_1d.csv")
            df_ief = pd.read_csv("data/ief_1d.csv")
            df_uup["datetime"] = pd.to_datetime(df_uup["datetime"])
            df_ief["datetime"] = pd.to_datetime(df_ief["datetime"])
            df_uup = df_uup.sort_values("datetime").reset_index(drop=True)
            df_ief = df_ief.sort_values("datetime").reset_index(drop=True)

            def macro_features(df, prefix):
                df[f"{prefix}_return"]     = df["close"].pct_change(1)
                ema = df["close"].ewm(span=10, adjust=False).mean()
                df[f"{prefix}_ema_slope"]  = (ema - ema.shift(5)) / 5
                df[f"{prefix}_volatility"] = df["close"].rolling(10).std()
                return df

            df_uup = macro_features(df_uup, "dxy")
            df_ief = macro_features(df_ief, "yield")
            uup_cols = ["datetime", "dxy_return", "dxy_ema_slope", "dxy_volatility"]
            ief_cols = ["datetime", "yield_return", "yield_ema_slope", "yield_volatility"]

            df15 = pd.merge_asof(
                df15.sort_values("datetime"),
                df_uup[uup_cols].sort_values("datetime"),
                on="datetime", direction="backward"
            )
            df15 = pd.merge_asof(
                df15.sort_values("datetime"),
                df_ief[ief_cols].sort_values("datetime"),
                on="datetime", direction="backward"
            )
        except FileNotFoundError:
            for c in ["dxy_return", "dxy_ema_slope", "dxy_volatility",
                      "yield_return", "yield_ema_slope", "yield_volatility"]:
                df15[c] = 0
    else:
        # Live mode: Macro features injected by paper_trade fetcher
        pass

    df15 = df15.fillna(0)

    # Drop intermediate columns not needed downstream
    drop_cols = [
        "tr1", "tr2", "tr3", "tr",
        "local_high_15", "local_low_15",
        "ema200", "std80", "sma80",
        "pd_high", "pd_low",
        "prev_body",
    ]
    df15 = df15.drop(columns=drop_cols, errors="ignore")

    return df15


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(DATA_15M_FILE)
    print(f"  Rows: {len(df):,}")
    print("Generating features (this takes a few minutes)...")
    df_features = generate_features(df, is_live=False)
    df_features.to_csv(FEATURES_FILE, index=False)
    new_cols = [c for c in df_features.columns if c not in
                ["datetime","open","high","low","close","volume"]]
    print(f"Features generated: {len(new_cols)} feature columns")
    print("New features added:")
    new_feature_names = [
        "vwap", "price_vs_vwap", "dist_to_vwap_norm", "vwap_slope",
        "level_reaction_5", "level_reaction_10",
        "level_held_as_support", "level_held_as_resistance",
        "level_bounce_strength", "level_break_confirmed",
        "level_retest_signal", "nearest_to_next_ratio",
        "levels_above_count", "levels_below_count",
        "breakout_room_up", "breakout_room_down",
    ]
    for f in new_feature_names:
        if f in df_features.columns:
            print(f"  ✅ {f:35s} mean={df_features[f].mean():.4f}")
        else:
            print(f"  ❌ MISSING: {f}")
    print(f"\nSaved → {FEATURES_FILE}")