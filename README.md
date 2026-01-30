# AI-Based Crypto-Currency Trading System 📈 ₿

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📋 Project Overview
This project implements an end-to-end Machine Learning pipeline designed to forecast **Bitcoin (BTC)** price trends and generate actionable **Buy/Sell/Hold** signals.

The system utilizes **XGBoost** for classification, engineered with technical indicators (RSI, MACD, Bollinger Bands) derived from raw Binance OHLCV data. It features a comprehensive backtesting engine and an interactive **Streamlit Dashboard** for visualizing equity growth and trade performance.

### 🚀 Key Results
* **Win Rate:** 64.44%
* **Total Return:** 93.75% (in backtesting)
* **Model:** XGBoost Classifier with Sample Weighting for class imbalance.

---

## 📂 Project Structure

The repository is organized into modular components reflecting the algorithmic trading lifecycle:

```bash
AI-Based-Crypto-Currency-Trading-System/
├── 1. Dataset Extraction/   # Data mining scripts
│   ├── data.py              # Script to fetch OHLCV data from Binance API
│   └── btc.csv              # Processed historical Bitcoin data
│
├── 2. AI Model Trainer/     # ML pipeline
│   ├── model.py             # Feature engineering & XGBoost training logic
│   └── xgb_model.pkl        # Serialized trained model file
│
├── 3. Back Testing/         # Performance evaluation
│   ├── dash.py              # Streamlit dashboard for visualization
│   └── trades.csv           # Log of executed backtest trades
│
├── 4. Live Trader/          # Deployment scripts
│   ├── live.py              # Real-time trading execution
│   └── paper.py             # Paper trading simulation script
│
├── 5. Documentation/        # Reports & Analysis
│   ├── Final Black Book_merged.pdf  # Full project report and methodology
│   └── Final Project Report.docx    # Supplementary documentation
│
└── requirements.txt         # Project dependencies
