"""
Phase 16: Macro Regime Data Collection via yfinance
Downloads UUP (Dollar Index proxy) and IEF (10Y Treasury proxy) 
at daily resolution (full history available), then forward-fills
to align with 1H Gold candles.
"""

import yfinance as yf
import pandas as pd

def download_macro_yahoo(symbol, output_file):
    """Download full history at daily resolution from Yahoo Finance (free)."""
    print(f"Downloading {symbol} daily data via Yahoo Finance...")
    
    try:
        df = yf.download(symbol, start="2014-01-01", end="2026-12-31", interval="1d", progress=False)
        
        if df is None or len(df) == 0:
            print(f"No data for {symbol}")
            return
        
        df = df.reset_index()
        df = df.rename(columns={"Date": "datetime", "Close": "close"})
        
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
            df = df.rename(columns={"datetime_": "datetime"})
            close_col = [c for c in df.columns if "close" in c.lower() or "Close" in c]
            if close_col:
                df = df.rename(columns={close_col[0]: "close"})
        
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        df = df[["datetime", "close"]].dropna()
        df = df.sort_values("datetime").reset_index(drop=True)
        
        df.to_csv(output_file, index=False)
        print(f"Saved {symbol} -> {output_file} ({len(df)} bars, {df['datetime'].min()} to {df['datetime'].max()})")
        
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")

if __name__ == "__main__":
    download_macro_yahoo("UUP", "data/uup_1d.csv")     # Dollar Index Proxy  
    download_macro_yahoo("IEF", "data/ief_1d.csv")     # 10-Year Yield Proxy
