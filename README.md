# Gold.ai: Algorithmic Market Sentiment & Trading Engine

An advanced, quantitative trading system designed specifically for the Gold (XAU/USD) market. It computes structural price levels and session regimes from 15-minute data and forward paper-trades a single validated macro rule, with Telegram alerts.

## 🚀 Key Features

*   **Algorithmic Level Generation**: Automatically calculates daily structural levels (U1-U8, L1-L8) based on Average True Range (ATR) and session open prices.
*   **Intraday Regime Classification**: Analyzes price action across London and New York sessions to categorize the market into distinct "regimes" (e.g., trend, range, exhaustion).
*   **Macro Pullback Rule (live, paper trading)**: LONG when the EMA200 slope and the dollar (UUP) are both falling at the session open; held 10 sessions, no stop (`signal_rules.py`, `signal_notifier.py`).
*   **Rules-Based Forward Test**: `forward_test_signal.py` reports the older level / sequence rules, which are archived (disabled) after lookahead-free re-validation showed no edge.
*   **Live Interactive Dashboard**: A Flask-based web application with real-time TradingView charts overlaid with dynamic algorithmic levels, live signal feeds, and historical performance tracking.

## 📊 Strategy Highlights

The only active rule is the **Macro Pullback** LONG (EMA falling + dollar falling), entered at the signal session's close and held 10 sessions. Lookahead-free validation, 0.02% cost: 2020–2026 about 87 trades, 59% win, +1.20% per trade; see `signal_rules.py` for the current figures. It is paper traded only.

## 🛠️ Tech Stack

*   **Backend**: Python, Flask, Pandas, NumPy
*   **Frontend**: HTML/CSS/JS, Lightweight Charts, TradingView Widget
*   **Data APIs**: TwelveData (XAU/USD 15m), Yahoo Finance (UUP / IEF daily)
*   **Automation**: GitHub Actions (`.github/workflows/telegram_alerts.yml`) + Telegram

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DAKSHIT1/GoldMarketSentiment_AI.git
    cd GoldMarketSentiment_AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Live Dashboard:**
    ```bash
    python dashboard/server.py
    ```
4.  Open `http://localhost:5050/dashboard` in your browser and sign in with your dashboard credentials.

## 📁 Repository Structure

*   `/dashboard`: Flask application, HTML templates, and API endpoints.
*   `/data`: Historical OHLCV datasets and processed features.
*   `forward_test_signal.py`: Core rule-based strategy execution and validation.
*   `signal_rules.py` / `signal_notifier.py`: shared rule definitions and the Telegram paper trader.
*   `refresh_data.py` / `session_builder.py` / `verify_parity.py`: live data refresh, session construction, parity check.
*   `feature_engine.py`: Complex feature generation from raw price data.

---
*Developed as an advanced algorithmic sentiment prediction system.*
