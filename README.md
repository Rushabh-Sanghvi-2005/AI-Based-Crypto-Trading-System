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

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   
       git clone https://github.com/RushabhSanghvi-2005/AI-Based-Crypto-Trading-System.git
       cd AI-Based-Crypto-Trading-System

2. **Install Dependencies**
   Ensure you have Python installed, then run:
   
       pip install -r requirements.txt

   *Key libraries included:* `pandas`, `numpy`, `xgboost`, `streamlit`, `ta`, `scikit-learn`, `matplotlib`, `plotly`.

---

## 💻 Usage Guide

### Step 1: Data Extraction
Fetch the latest market data using the extraction script.

    cd "1. Dataset Extraction"
    python data.py

### Step 2: Model Training
Train the XGBoost model on the fetched data. This script handles feature engineering (RSI, MACD, ATR) and saves the model as `xgb_model.pkl`.

    cd "../2. AI Model Trainer"
    python model.py

### Step 3: Backtesting & Visualization
Run the Streamlit dashboard to visualize the model's performance against historical data.

    cd "../3. Back Testing"
    streamlit run dash.py

### Step 4: Live/Paper Trading
For real-time simulation or execution (use with caution):

    cd "../4. Live Trader"
    python paper.py

---

## 📊 Features
* **Technical Analysis:** Automated calculation of TA-Lib indicators (RSI, MACD, Bollinger Bands, ATR).
* **Class Imbalance Handling:** Uses sample weighting to prevent model bias toward the majority class.
* **Equity Curve Visualization:** Interactive charts plotting portfolio growth over time using Plotly/Altair.
* **Deployment Ready:** Modular scripts designed for easy deployment to cloud environments.

---

## 📜 License & Disclaimer
This project is for **educational and research purposes only**. Algorithmic trading involves significant risk. The past performance shown (93.75% return) is based on backtesting and does not guarantee future results.

**Author:** Rushabh Sanghvi
*Aspiring AI & FinTech Specialist*
