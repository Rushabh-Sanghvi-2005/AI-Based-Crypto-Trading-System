import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ta
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

st.set_page_config(page_title="Backtest Dashboard", layout="wide")

# ---------------- Config ----------------
MODEL_PATH = r"C:\Users\Dell\Desktop\VIVA College\Practicals - Projects - Assignments\Project\TYDS-36 - Rushabh_Sanghvi - AI-Based_Crypto-Currency_Trading-System\2. AI Model Trainer\xgb_model.pkl"
DATA_PATH  = r"C:\Users\Dell\Desktop\VIVA College\Practicals - Projects - Assignments\Project\TYDS-36 - Rushabh_Sanghvi - AI-Based_Crypto-Currency_Trading-System\1. Dataset Extraction\btc.csv"
HORIZON_MIN = 60
THRESH_BUY, THRESH_SELL = 0.005, -0.005


# ---------------- Header ----------------
st.title("Backtest Dashboard")
st.write(
    """
We used a **trained AI model (XGBoost)** and the **last one month of BTC/USDT data** to evaluate if the model is  
**accurate, precise, and profitable**.  

- **Dataset attributes:** time, open, high, low, close, volume  
- **Features:** ret1, ret5, MA(5)-MA(20), RSI(14), MACD, volume z-score, Bollinger %B, ATR, OBV  
- **Algorithm:** XGBoost  
- **Evaluation metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix  
- **Backtesting:** One-trade-at-a-time, 60 min horizon, no fees  
"""
)

# ---------------- Load model & data ----------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(DATA_PATH)
c, v, h, l = df["close"], df["volume"], df["high"], df["low"]

# ---------------- Features ----------------
X = pd.DataFrame({
    "ret1": c.pct_change(),
    "ret5": c.pct_change(5),
    "ma5_20": c.rolling(5).mean() - c.rolling(20).mean(),
    "rsi": ta.momentum.RSIIndicator(c, 14).rsi(),
    "macd": ta.trend.MACD(c).macd_diff(),
    "vol20": (v - v.rolling(20).mean()) / v.rolling(20).std(),
    "bb": ta.volatility.BollingerBands(c).bollinger_pband(),
    "atr": ta.volatility.AverageTrueRange(h, l, c).average_true_range(),
    "obv": ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
}).dropna()

close = c.iloc[-len(X):].reset_index(drop=True)
future_ret = (c.shift(-HORIZON_MIN) / c - 1).reindex(X.index)
y = pd.Series(1, index=X.index)
y[future_ret > THRESH_BUY] = 2
y[future_ret < THRESH_SELL] = 0
y = y.reset_index(drop=True)

# ---------------- Predictions ----------------
pred = pd.Series(model.predict(X), index=X.index).reset_index(drop=True)

# ---------------- Classification metrics ----------------
acc = accuracy_score(y, pred)
report = classification_report(y, pred, target_names=["SELL","HOLD","BUY"], output_dict=True, zero_division=0)
cm = confusion_matrix(y, pred, labels=[0,1,2])
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# ---------------- Backtest ----------------
profits, logs = [], []
t = 0
while t < len(close) - HORIZON_MIN:
    sig = pred.iloc[t]
    if sig == 2:  # BUY
        entry, exitp = close.iloc[t], close.iloc[t+HORIZON_MIN]
        ret = (exitp - entry) / entry
        profits.append(ret)
        logs.append(["BUY", t, t+HORIZON_MIN, entry, exitp, ret])
        t += HORIZON_MIN
    elif sig == 0:  # SELL
        entry, exitp = close.iloc[t], close.iloc[t+HORIZON_MIN]
        ret = (entry - exitp) / entry
        profits.append(ret)
        logs.append(["SELL", t, t+HORIZON_MIN, entry, exitp, ret])
        t += HORIZON_MIN
    else:
        t += 1

trades = pd.DataFrame(logs, columns=["side","entry_idx","exit_idx","entry_px","exit_px","ret"])
trades.to_csv("trades.csv", index=False)

# ---------------- Performance summary ----------------
st.subheader("Performance Summary")
if not trades.empty:
    eq = (1+trades["ret"]).cumprod()
    total_return = eq.iloc[-1]
    total_return_pct = (total_return-1)*100


    minutes_total = len(close)
    days = minutes_total/1440
    months = days/30 if days>0 else 1
    years = days/365 if days>0 else 1

    daily_cagr   = (total_return**(1/max(days,1e-9))-1)*100
    monthly_cagr = (total_return**(1/months)-1)*100
    yearly_cagr  = (total_return**(1/years)-1)*100

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Trades", len(trades))
    c2.metric("Win rate", f"{(trades['ret']>0).mean()*100:.2f}%", delta=None)
    c3.metric("Avg trade", f"{trades['ret'].mean()*100:.3f}%")
    c4.metric("Total return", f"{total_return_pct:.2f}%", delta="↑" if total_return_pct>0 else "↓")
    c5.metric("Daily CAGR", f"{daily_cagr:.2f}%")
    c6.metric("Monthly CAGR", f"{monthly_cagr:.2f}%")
    c7.metric("Yearly CAGR", f"{yearly_cagr:.2f}%")


# ---------------- Charts in Grid ----------------

# Row 1: Equity Growth + Profit per Day
st.subheader("Trading Performance Charts")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Equity Growth (Cumulative Profit %)**")
    if not trades.empty:
        equity_df = pd.DataFrame({"Trade #": np.arange(1,len(trades)+1), "Equity %": (eq-1)*100})
        line = alt.Chart(equity_df).mark_area(color="lightblue", opacity=0.5).encode(
            x="Trade #", y="Equity %"
        ) + alt.Chart(equity_df).mark_line(color="blue").encode(
            x="Trade #", y="Equity %"
        )
        st.altair_chart(line, use_container_width=True)

with col2:
    st.markdown("**Profit per Day**")
    if not trades.empty:
        trades["day"] = trades["entry_idx"] // 1440
        daily_profit = trades.groupby("day")["ret"].sum() * 100
        daily_df = pd.DataFrame({
            "Day": daily_profit.index + 1,
            "Daily Profit %": daily_profit.values,
            "Outcome": np.where(daily_profit >= 0, "Profit", "Loss")
        })
        bars = alt.Chart(daily_df).mark_bar().encode(
            x="Day",
            y="Daily Profit %",
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Profit","Loss"], range=["green","red"])),
            tooltip=["Day","Daily Profit %"]
        )
        st.altair_chart(bars, use_container_width=True)

# Row 2: Radar only
st.subheader("Model Evaluation Metrics")
col3, _ = st.columns([1,1])  # leave second column empty

with col3:
    st.markdown("**Radar Chart (Precision, Recall, F1)**")
    radar_data = pd.DataFrame({
        "Metric":["Precision","Recall","F1"],
        "SELL":[report["SELL"]["precision"],report["SELL"]["recall"],report["SELL"]["f1-score"]],
        "HOLD":[report["HOLD"]["precision"],report["HOLD"]["recall"],report["HOLD"]["f1-score"]],
        "BUY":[report["BUY"]["precision"],report["BUY"]["recall"],report["BUY"]["f1-score"]],
    }).set_index("Metric")

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))  # small chart
    angles = np.linspace(0, 2*np.pi, len(radar_data.columns), endpoint=False).tolist()
    angles += angles[:1]
    for idx,row in radar_data.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=idx)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_data.columns)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2,1.1), fontsize=8)
    st.pyplot(fig)

# Row 3: Confusion Matrices
st.subheader("Confusion Matrices")
col5, col6 = st.columns(2)

with col5:
    st.markdown("**Counts**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["SELL","HOLD","BUY"], yticklabels=["SELL","HOLD","BUY"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col6:
    st.markdown("**Normalized (%)**")
    fig, ax = plt.subplots()
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=["SELL","HOLD","BUY"], yticklabels=["SELL","HOLD","BUY"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# ---------------- Download trades ----------------
st.download_button("Download trades.csv", trades.to_csv(index=False).encode(), "trades.csv", "text/csv")
