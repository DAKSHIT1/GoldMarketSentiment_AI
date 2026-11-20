"""
Macro regime data via yfinance: UUP (dollar index proxy) and IEF (10Y Treasury proxy),
daily closes. UUP drives dxy_slope, half of the active macro rule, so a bad download must
never silently replace good data:
  * closes are rounded to 4 dp, so yfinance's float noise does not rewrite every row daily;
  * a download with fewer rows or an older last date than the stored file is rejected
    and the stored file is kept.
"""

import os

import pandas as pd
import yfinance as yf


def _load_existing(output_file):
    if not os.path.exists(output_file):
        return None
    try:
        return pd.read_csv(output_file, parse_dates=["datetime"])
    except Exception:
        return None


def download_macro_yahoo(symbol, output_file):
    """Download full daily history from Yahoo Finance. Returns True if the file is up to date."""
    print(f"Downloading {symbol} daily data via Yahoo Finance...")
    try:
        df = yf.download(symbol, start="2014-01-01", interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"❌ {symbol}: download error: {e} — keeping stored file")
        return False
    if df is None or len(df) == 0:
        print(f"❌ {symbol}: empty download — keeping stored file")
        return False

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):          # yfinance >= 0.2 returns (field, ticker)
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={"Date": "datetime", "Close": "close"})
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df[["datetime", "close"]].dropna().sort_values("datetime").reset_index(drop=True)
    df["close"] = df["close"].round(4)

    old = _load_existing(output_file)
    if old is not None and len(old):
        if len(df) < len(old) or df["datetime"].max() < old["datetime"].max():
            print(f"❌ {symbol}: download looks truncated ({len(df)} rows to {df['datetime'].max().date()} "
                  f"vs stored {len(old)} to {old['datetime'].max().date()}) — keeping stored file")
            return False

    tmp = output_file + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, output_file)
    print(f"Saved {symbol} -> {output_file} ({len(df)} bars, {df['datetime'].min().date()} to "
          f"{df['datetime'].max().date()})")
    return True


if __name__ == "__main__":
    download_macro_yahoo("UUP", "data/uup_1d.csv")     # Dollar Index Proxy
    download_macro_yahoo("IEF", "data/ief_1d.csv")     # 10-Year Yield Proxy
