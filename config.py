import os

# TwelveData key: env var (GitHub Actions secret) or gitignored config_local.py
API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
if not API_KEY:
    try:
        from config_local import API_KEY
    except ImportError:
        pass

SYMBOL = "XAU/USD"

# timeframes
INTERVAL_5M = "5min"
INTERVAL_15M = "15min"
INTERVAL_1H = "1h"

# number of candles to download
OUTPUT_SIZE = 5000
HISTORICAL_BATCHES = 70

# data files
DATA_5M_FILE = "data/gold_5m.csv"
DATA_15M_FILE = "data/gold_15m_full.csv"
DATA_1H_FILE = "data/gold_1h.csv"

# features dataset
FEATURES_FILE = "data/gold_features.csv"

# trading engine
LEVEL_COUNT = 10

NY_SESSION_OPEN = "18:00"

