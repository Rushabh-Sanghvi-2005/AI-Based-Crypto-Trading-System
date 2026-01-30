import pandas as pd
import numpy as np
import pickle
import ta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Load data
df = pd.read_csv(r"1. Dataset Extraction\btc.csv")
c = df["close"]
v = df["volume"]
h = df["high"]
l = df["low"]

# Features
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

# Labels: SELL=0, HOLD=1, BUY=2
future_ret = (c.shift(-60) / c - 1).reindex(X.index)
y = pd.Series(1, index=X.index)   # default HOLD
y[future_ret > 0.005] = 2         # BUY
y[future_ret < -0.005] = 0        # SELL

# Split
Xtr, Xte, ytr, yte = train_test_split(X, y, shuffle=False, test_size=0.2)

# Class balancing
w = compute_sample_weight("balanced", ytr)

# Model
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9
)
model.fit(Xtr, ytr, sample_weight=w)

# Eval
pred = model.predict(Xte)
print("acc =", accuracy_score(yte, pred))
# print(confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=["SELL","HOLD","BUY"]))

# Save
'''
pickle.dump(model, open("xgb_model.pkl", "wb"))
'''