import requests
import pandas as pd
import time
from config import *

BASE_URL = "https://api.twelvedata.com/time_series"

def download(interval, output_file):

    all_data = []
    end_time = None

    # Max possible batches to prevent infinite loops, aim for ~10+ years
    max_batches = 300 

    for i in range(max_batches):

        print(f"Downloading {interval} batch:", i+1)

        url = f"{BASE_URL}?symbol={SYMBOL}&interval={interval}&outputsize={OUTPUT_SIZE}&apikey={API_KEY}"

        if end_time:
            url += f"&end_date={end_time}"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=15)
                data = r.json()
                if "values" in data:
                    break
                else:
                    print(f"API returned no values (Attempt {attempt+1}/{max_retries}):", data)
                    time.sleep(15)
            except Exception as e:
                print(f"Network error on attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(15)
                data = {}

        if "values" not in data:
            print("API error / Rate limit / End of data:", data)
            break

        candles = data["values"]

        df = pd.DataFrame(candles)
        df = df[["datetime","open","high","low","close"]]

        all_data.append(df)

        end_time = candles[-1]["datetime"]
        
        # Stop fetching if we've reached 2014 or earlier (Ensuring solid 2015-2024 data)
        if int(end_time[:4]) <= 2014:
            print(f"Reached {end_time}, which fulfills the 10-year requirement. Halting {interval} loop.")
            break

        time.sleep(8)

    if not all_data:
        print("No data downloaded")
        return

    full = pd.concat(all_data)
    full = full.drop_duplicates()

    full.to_csv(output_file, index=False)

    print(f"Saved {interval} data ->", output_file)
    print("Total candles:", len(full))


def fetch_all():

    download("5min", DATA_5M_FILE)
    download("15min", DATA_15M_FILE)
    download("1h", DATA_1H_FILE)


if __name__ == "__main__":
    fetch_all()