# Gold.ai: Algorithmic Market Sentiment & Trading Engine

An advanced, quantitative trading system designed specifically for the Gold (XAU/USD) market. This system uses structural price levels, session volatility regimes, and machine learning models to identify high-probability intraday trading opportunities.

## 🚀 Key Features

*   **Algorithmic Level Generation**: Automatically calculates daily structural levels (U1-U8, L1-L8) based on Average True Range (ATR) and session open prices.
*   **Intraday Regime Classification**: Analyzes price action across London and New York sessions to categorize the market into distinct "regimes" (e.g., trend, range, exhaustion).
*   **Machine Learning Signal Engine**: Uses CatBoost to predict market direction and trade viability based on level interactions, VWAP, and macro factors (DXY, Yields).
*   **Rules-Based Forward Test**: Employs a battle-tested rule engine focusing on overextension reversals (e.g., "U4 Overextension", "L4 Overextension") that achieved a 72.5% win rate in out-of-sample forward testing.
*   **Live Interactive Dashboard**: A Flask-based web application with real-time TradingView charts overlaid with dynamic algorithmic levels, live signal feeds, and historical performance tracking.

## 📊 Strategy Highlights

The core edge of this system relies on **Overextension Reversals**. When the gold price stretches too far from its underlying moving averages and structural baselines within a single session, the system anticipates a snapback.

*   **LONG Edge**: Highly accurate in identifying bullish reversals after false breakdowns (72.5% OOS accuracy).
*   **STAY COOL Filter**: Intelligently identifies choppy, bidirectional markets and advises staying out of the market.

## 🛠️ Tech Stack

*   **Backend**: Python, Flask, Pandas, NumPy
*   **Machine Learning**: CatBoost, scikit-learn
*   **Frontend**: HTML/CSS/JS, Lightweight Charts, TradingView Widget
*   **Data APIs**: Yahoo Finance (Market Data)

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
4.  Open `http://localhost:5050/dashboard` in your browser. (Login: dakshit / gold2025)

## 📁 Repository Structure

*   `/dashboard`: Flask application, HTML templates, and API endpoints.
*   `/data`: Historical OHLCV datasets and processed features.
*   `/models`: Pre-trained CatBoost machine learning models.
*   `forward_test_signal.py`: Core rule-based strategy execution and validation.
*   `backtest_final.py`: ML-based backtesting pipeline.
*   `feature_engine.py`: Complex feature generation from raw price data.

---
*Developed as an advanced algorithmic sentiment prediction system.*
