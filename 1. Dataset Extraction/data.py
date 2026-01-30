'''
⚡ What this does

Calls Binance 45 times to backfill ~31 days (45,000 mins).

Concatenates results in chronological order.

Exports btc.csv with time, open, high, low, close, volume.

👉 This file is now ready for the training script.
'''


import requests, pandas as pd, time

url = "https://api.binance.com/api/v3/klines"
params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 1000}
all_data, end_time = [], int(time.time() * 1000)  # current time in ms

for _ in range(45):  # ~31 days worth (45*1000 minutes)
    params["endTime"] = end_time
    data = requests.get(url, params=params).json()
    all_data = data + all_data   # prepend older candles
    end_time = data[0][0] - 1    # move window back

df = pd.DataFrame(all_data, columns=[
    "time","open","high","low","close","volume",
    "_","quote","trades","taker_base","taker_quote","ignore"
])

# keep only required columns
df = df[["time","open","high","low","close","volume"]]
df["time"] = pd.to_datetime(df["time"], unit="ms")
df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})

print(df.head)

'''
df.to_csv("btc.csv", index=False)
'''

print("Saved", len(df), "rows to btc.csv")
