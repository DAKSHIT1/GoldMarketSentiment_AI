"""
download_data.py
Paginates TwelveData to download the full 6-year XAU/USD 15m history (Jan 2020 -> now).
Saves to data/gold_15m_full.csv.
"""
import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from config import API_KEY

os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data/gold_15m_full.csv"
START_DATE = "2020-01-24"  # Earliest available from TwelveData

def fetch_batch(end_date_str):
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol=XAU/USD"
        f"&interval=15min"
        f"&outputsize=5000"
        f"&end_date={end_date_str}"
        f"&timezone=America/New_York"
        f"&apikey={API_KEY}"
    )
    res = requests.get(url).json()
    if "values" not in res:
        print("API Error:", res)
        return None
    df = pd.DataFrame(res["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
    return df.sort_values("datetime").reset_index(drop=True)

def download_full_history():
    print(f"Starting paginated download from {START_DATE} to now...")
    all_batches = []
    # Start from today and paginate backwards
    current_end = datetime.now()
    earliest = datetime.strptime(START_DATE, "%Y-%m-%d")
    batch_num = 0

    while current_end > earliest:
        end_str = current_end.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  Batch {batch_num+1}: fetching 5000 bars ending {end_str}...")
        batch = fetch_batch(end_str)
        if batch is None or len(batch) == 0:
            break
        all_batches.append(batch)
        # Move the window: next end_date = oldest bar in this batch - 1 minute
        oldest_in_batch = batch["datetime"].iloc[0]
        current_end = oldest_in_batch - timedelta(minutes=1)
        batch_num += 1
        if oldest_in_batch <= earliest:
            print(f"  Reached start date {START_DATE}. Stopping.")
            break
        time.sleep(0.3)  # Gentle rate-limit respect

    if not all_batches:
        print("No data downloaded.")
        return None

    df = pd.concat(all_batches).drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDownload complete!")
    print(f"Total bars:   {len(df):,}")
    print(f"Date range:   {df['datetime'].iloc[0]}  -->  {df['datetime'].iloc[-1]}")
    print(f"Price range:  ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Saved to:     {OUTPUT_FILE}")
    return df

if __name__ == "__main__":
    download_full_history()
