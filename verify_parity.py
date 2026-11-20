"""
verify_parity.py
================
Checks that the LIVE session builder (refresh_data.build_sessions_from, which
runs generate_features on a trailing window) reproduces the BATCH session
rows in data/session_regime_v2.csv, and that the shared signal rules give the
same label. Run before trusting live alerts, and in CI before sending.

  python verify_parity.py [N_SESSIONS]   (default 30)

Exit code 1 if any session's level counts / flags / signal label differ.
"""

import sys
import numpy as np
import pandas as pd

from refresh_data import build_sessions_from, SESSION_FILE, DATA_15M_FILE
from session_builder import add_rolling_features, MIN_SESSION_BARS
import signal_rules

EXACT = ["max_up", "max_down", "is_bidir", "l1_false_brk", "broke_u1", "broke_l1"]
CLOSE = ["ema_slope", "vwap_slope", "sess_open", "sess_close"]
SIGN  = ["ema_slope", "vwap_slope", "dxy_slope"]


def main(n=30):
    stored = pd.read_csv(SESSION_FILE)
    stored["date"] = pd.to_datetime(stored["date"])
    stored = stored.sort_values("date").reset_index(drop=True)
    bars = pd.read_csv(DATA_15M_FILE, parse_dates=["datetime"])

    # Compare only sessions that are complete in the stored bars (skip the last stored row if partial)
    end_idx = len(stored) - 1
    start_idx = max(0, end_idx - n)
    from_date = stored["date"].iloc[start_idx]
    live = build_sessions_from(bars, from_date)
    live = live[live["complete"] & (live["nbars"] >= MIN_SESSION_BARS)].set_index("date")

    fails = 0
    rows = []
    for i in range(start_idx, end_idx):
        s = stored.iloc[i]
        d = s["date"]
        if d not in live.index:
            print(f"  {d}: missing from live rebuild"); fails += 1; continue
        l = live.loc[d]
        bad = [c for c in EXACT if float(s[c]) != float(l[c])]
        bad += [c for c in CLOSE if not np.isclose(float(s[c]), float(l[c]), rtol=1e-3, atol=1e-4)
                and c not in ("ema_slope", "vwap_slope")]
        bad += [f"sign({c})" for c in SIGN
                if np.sign(round(float(s[c]), 6)) != np.sign(round(float(l[c]), 6))]
        rows.append((d, bad, float(s["ema_slope"]) - float(l["ema_slope"]),
                     float(s["vwap_slope"]) - float(l["vwap_slope"])))
        if bad:
            fails += 1

    # Signal label parity: rebuild rolling features on the live rows with stored history
    hist = stored.iloc[:start_idx].copy()
    live_rows = live.reset_index()
    live_rows = live_rows[live_rows["date"] < stored["date"].iloc[end_idx]]
    combo = pd.concat([hist, live_rows.drop(columns=["complete", "nbars"])], ignore_index=True, sort=False)
    combo = add_rolling_features(combo, start_idx=len(hist))
    st_recs = stored.to_dict("records")
    lv_recs = combo.to_dict("records")
    label_fails = 0
    for i in range(start_idx, end_idx):
        a = signal_rules.classify_at(st_recs, i)
        b = signal_rules.classify_at(lv_recs, i)
        if a["label"] != b["label"]:
            label_fails += 1
            print(f"  LABEL MISMATCH {st_recs[i]['date']}: stored={a['label']}  live={b['label']}")

    print(f"Parity over {end_idx - start_idx} sessions ({from_date.date()} → {stored['date'].iloc[end_idx - 1].date()}):")
    for d, bad, de, dv in rows:
        if bad:
            print(f"  ✗ {d}  {bad}  Δema={de:+.2e} Δvwap={dv:+.2e}")
    max_de = max(abs(r[2]) for r in rows) if rows else 0
    max_dv = max(abs(r[3]) for r in rows) if rows else 0
    print(f"  feature mismatches: {fails}   label mismatches: {label_fails}   "
          f"max|Δema_slope|={max_de:.2e}  max|Δvwap_slope|={max_dv:.2e}")
    return 0 if (fails == 0 and label_fails == 0) else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
