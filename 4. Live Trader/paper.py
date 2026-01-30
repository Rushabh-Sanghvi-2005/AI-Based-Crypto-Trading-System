# paper_live_ist.py
import streamlit as st, pandas as pd, numpy as np, requests, pickle, ta, altair as alt, pytz

# ---------- CONFIG ----------
MODEL_PATH   = r"C:\Users\Dell\Desktop\VIVA College\Practicals - Projects - Assignments\Project\TYDS-36 - Rushabh_Sanghvi - AI-Based_Crypto-Currency_Trading-System\2. AI Model Trainer\xgb_model.pkl"
SYMBOL       = "BTCUSDT"
INTERVAL     = "1m"
HIST_LIMIT   = 300             # used only to compute indicators
TAKE_PROFIT  = 0.001           # +0.10% to allow exit
START_EQUITY = 1000.0
REFRESH_MS   = 60_000
IST          = pytz.timezone("Asia/Kolkata")
HEADERS      = {"Cache-Control":"no-cache","Pragma":"no-cache"}

st.set_page_config(page_title="Paper Trading — Live 1m (IST)", layout="wide")

# ---------- AUTO-REFRESH ----------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_MS, key="auto_refresh")
except Exception:
    st.caption("Tip: pip install streamlit-autorefresh for 60s refresh")
if st.button("🔁 Refresh now"):
    try: st.rerun()
    except Exception: pass

# ---------- STATE ----------
ss = st.session_state
if "equity" not in ss: ss.equity = START_EQUITY
if "equity_curve" not in ss: ss.equity_curve = []              # (IST-naive dt, equity)
if "closed" not in ss: ss.closed = []                          # trade log
if "pos" not in ss: ss.pos = None                              # {'side','entry_px','entry_time_ist'}
if "last_closed_bar_time_utc" not in ss: ss.last_closed_bar_time_utc = None
if "last_logged_bar_time_utc" not in ss: ss.last_logged_bar_time_utc = None
if "pred_table" not in ss:
    ss.pred_table = pd.DataFrame(columns=[
        "time_ist","open","high","low","close","volume",
        "p_sell","p_hold","p_buy","model_decision","forced_trade_side"
    ])
if "cold_start" not in ss: ss.cold_start = True   # keeps table blank on first load

# ---------- TIME HELPERS ----------
def binance_server_ms():
    r = requests.get("https://api.binance.com/api/v3/time", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return int(r.json()["serverTime"])

def last_closed_min_end_ms():
    ms = binance_server_ms()
    return (ms - (ms % 60000)) - 1     # last ms of previous full minute

# ---------- DATA & FEATURES ----------
def fetch_closed_window(symbol=SYMBOL, interval=INTERVAL, limit=HIST_LIMIT):
    end_ms = last_closed_min_end_ms()
    params = {"symbol":symbol,"interval":interval,"limit":limit,"endTime":end_ms}
    r = requests.get("https://api.binance.com/api/v3/klines", params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    d = r.json()
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","tb","tq","ignore"]
    df = pd.DataFrame(d, columns=cols)[["open_time","open","high","low","close","volume"]].astype(float)
    df["time_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["time_ist"] = df["time_utc"].dt.tz_convert(IST).dt.tz_localize(None)   # IST-naive for charts/tables
    return df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})[
        ["time_utc","time_ist","open","high","low","close","volume"]
    ].reset_index(drop=True)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c,v,h,l = df["close"], df["volume"], df["high"], df["low"]
    X = pd.DataFrame({
        "ret1": c.pct_change(),
        "ret5": c.pct_change(5),
        "ma5_20": c.rolling(5).mean() - c.rolling(20).mean(),
        "rsi": ta.momentum.RSIIndicator(c,14).rsi(),
        "macd": ta.trend.MACD(c).macd_diff(),
        "vol20": (v - v.rolling(20).mean())/v.rolling(20).std(),
        "bb": ta.volatility.BollingerBands(c).bollinger_pband(),
        "atr": ta.volatility.AverageTrueRange(h,l,c).average_true_range(),
        "obv": ta.volume.OnBalanceVolumeIndicator(c,v).on_balance_volume()
    }, index=df.index)
    idx = X.dropna().index
    X = X.loc[idx].copy()
    X["px"]       = df.loc[idx, "close"].values
    X["open"]     = df.loc[idx, "open"].values
    X["high"]     = df.loc[idx, "high"].values
    X["low"]      = df.loc[idx, "low"].values
    X["volume"]   = df.loc[idx, "volume"].values
    X["time_utc"] = df.loc[idx, "time_utc"].values
    X["time_ist"] = df.loc[idx, "time_ist"].values
    return X.reset_index(drop=True)

def decide_side_and_proba(model, row_df):
    try:
        P = model.predict_proba(row_df)[0]  # [SELL, HOLD, BUY]
        pred = int(np.argmax(P))
        decision = "SELL" if pred==0 else "BUY" if pred==2 else "HOLD"
        forced  = decision if decision!="HOLD" else ("BUY" if P[2]>=P[0] else "SELL")
        return forced, P
    except Exception:
        pred = int(model.predict(row_df)[0])
        decision = "SELL" if pred==0 else "BUY" if pred==2 else "HOLD"
        forced = "SELL" if decision=="HOLD" else decision
        return forced, [np.nan,np.nan,np.nan]

def can_exit(side, entry_px, px, tp=TAKE_PROFIT):
    ret = (px - entry_px)/entry_px if side=="LONG" else (entry_px - px)/entry_px
    return ret >= tp, ret

# ---------- LOAD MODEL & DATA ----------
with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
raw = fetch_closed_window()
X   = build_features(raw)
if X.empty: st.warning("Waiting for enough candles for indicators…"); st.stop()

# The last row of X is already the latest CLOSED candle (because of endTime)
closed_bar  = X.iloc[[-1]].copy()
bar_time_utc= closed_bar["time_utc"].iloc[0]
bar_time_ist= closed_bar["time_ist"].iloc[0]
price       = float(closed_bar["px"].iloc[0])

# ---------- TABLE APPEND LOGIC (BLANK ON FIRST LOAD) ----------
# On first run: keep table blank and set markers to *current* closed bar, so rows start next minute.
if ss.cold_start:
    ss.last_logged_bar_time_utc = bar_time_utc
    ss.last_closed_bar_time_utc = bar_time_utc
    ss.cold_start = False
else:
    # Append exactly ONE row if a *new* closed bar arrived
    if bar_time_utc != ss.last_logged_bar_time_utc:
        feat_row = closed_bar.drop(columns=["px","time_utc","time_ist","open","high","low","volume"])
        forced_side, P = decide_side_and_proba(model, feat_row)
        p_sell = float(P[0]) if isinstance(P,(list,np.ndarray)) else np.nan
        p_hold = float(P[1]) if isinstance(P,(list,np.ndarray)) else np.nan
        p_buy  = float(P[2]) if isinstance(P,(list,np.ndarray)) else np.nan
        # model_decision (before forcing)
        decision = "SELL" if (not np.isnan(p_sell) and p_sell>=max(p_hold,p_buy)) else \
                   "BUY" if (not np.isnan(p_buy) and p_buy>=max(p_hold,p_sell)) else "HOLD"
        newrow = pd.DataFrame([{
            "time_ist": bar_time_ist, "open": float(closed_bar["open"].iloc[0]),
            "high": float(closed_bar["high"].iloc[0]), "low": float(closed_bar["low"].iloc[0]),
            "close": price, "volume": float(closed_bar["volume"].iloc[0]),
            "p_sell": None if np.isnan(p_sell) else round(p_sell,4),
            "p_hold": None if np.isnan(p_hold) else round(p_hold,4),
            "p_buy":  None if np.isnan(p_buy)  else round(p_buy,4),
            "model_decision": decision, "forced_trade_side": forced_side
        }])
        ss.pred_table = pd.concat([ss.pred_table, newrow], ignore_index=True)
        ss.last_logged_bar_time_utc = bar_time_utc

# ---------- TRADING (ONLY WHEN NEW CLOSED BAR ARRIVES) ----------
is_new_closed = (bar_time_utc != ss.last_closed_bar_time_utc)
if is_new_closed:
    # Exit first (if in position and conditions satisfied)
    if ss.pos is not None:
        side = ss.pos["side"]; entry_px = ss.pos["entry_px"]
        ok_profit, ret_now = can_exit(side, entry_px, price, TAKE_PROFIT)
        feat_row = closed_bar.drop(columns=["px","time_utc","time_ist","open","high","low","volume"])
        forced_side, P = decide_side_and_proba(model, feat_row)
        want_exit = (side=="LONG" and forced_side=="SELL") or (side=="SHORT" and forced_side=="BUY")
        if ok_profit and want_exit:
            ss.equity *= (1 + ret_now)
            ss.closed.append({
                "entry_time_ist": ss.pos["entry_time_ist"], "exit_time_ist": bar_time_ist,
                "side": "BUY→SELL" if side=="LONG" else "SELL→BUY",
                "entry_px": entry_px, "exit_px": price, "ret": float(ret_now)
            })
            ss.equity_curve.append((bar_time_ist, ss.equity))
            ss.pos = None

    # Enter if flat
    if ss.pos is None:
        feat_row = closed_bar.drop(columns=["px","time_utc","time_ist","open","high","low","volume"])
        forced_side, _ = decide_side_and_proba(model, feat_row)
        ss.pos = {"side": "LONG" if forced_side=="BUY" else "SHORT",
                  "entry_px": price, "entry_time_ist": bar_time_ist}
        if not ss.equity_curve or ss.equity_curve[-1][0] != bar_time_ist:
            ss.equity_curve.append((bar_time_ist, ss.equity))

    ss.last_closed_bar_time_utc = bar_time_utc

# ---------- FRAMES FOR UI ----------
closed_df = pd.DataFrame(ss.closed)
if not closed_df.empty: closed_df["ret_pct"] = closed_df["ret"]*100
equity_df = pd.DataFrame(ss.equity_curve, columns=["time_ist","equity"])

unreal_pct = 0.0
if ss.pos is not None:
    if ss.pos["side"]=="LONG":
        unreal_pct = (price - ss.pos["entry_px"])/ss.pos["entry_px"]*100
    else:
        unreal_pct = (ss.pos["entry_px"] - price)/ss.pos["entry_px"]*100

trades = len(closed_df)
winrate = (closed_df["ret"]>0).mean()*100 if trades else 0.0
total_ret = (ss.equity/START_EQUITY - 1)*100

# ---------- UI ----------
st.title("Paper Trading — Live 1m (IST)")
st.subheader("Closed 1-Minute BTC Candles (IST) + Predictions (fills one row per minute)")
st.dataframe(ss.pred_table, use_container_width=True)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Closed trades", trades)
c2.metric("Position", ss.pos["side"] if ss.pos else "FLAT")
c3.metric("Win rate", f"{winrate:.2f}%")
c4.metric("Total return", f"{total_ret:.2f}%")
c5.metric("Equity", f"{ss.equity:,.2f}")

col1,col2 = st.columns(2)
with col1:
    st.subheader("Equity (IST)")
    if not equity_df.empty:
        st.altair_chart(alt.Chart(equity_df).mark_line().encode(
            x=alt.X("time_ist:T", title="Time (IST)"),
            y=alt.Y("equity:Q", title="Equity")
        ), use_container_width=True)
    else:
        st.info("Waiting for first point…")

with col2:
    st.subheader("Profit per Trade (%)")
    if trades:
        c = closed_df.copy(); c["trade_num"] = np.arange(1, len(c)+1)
        bars = alt.Chart(c).mark_bar().encode(
            x=alt.X("trade_num:Q", title="Trade #"),
            y=alt.Y("ret_pct:Q", title="Profit %"),
            color=alt.condition(alt.datum.ret_pct>=0, alt.value("green"), alt.value("red")),
            tooltip=["trade_num","entry_time_ist","exit_time_ist","side","entry_px","exit_px","ret_pct"]
        )
        zero = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="gray").encode(y="y")
        st.altair_chart(bars+zero, use_container_width=True)
    else:
        st.info("No closed trades yet.")

col3,col4 = st.columns(2)
with col3:
    st.subheader("Open Position (IST)")
    if ss.pos:
        st.write(pd.DataFrame([{
            "side": ss.pos["side"], "entry_time_ist": ss.pos["entry_time_ist"],
            "entry_px": ss.pos["entry_px"], "latest_closed_time_ist": bar_time_ist,
            "latest_closed_px": price, "unrealized_%": unreal_pct
        }]))
    else:
        st.success("FLAT")

with col4:
    st.subheader("Closed Trades (IST)")
    if trades:
        st.dataframe(
            closed_df[["entry_time_ist","exit_time_ist","side","entry_px","exit_px","ret_pct"]]
            .sort_values("exit_time_ist", ascending=False).head(20),
            use_container_width=True
        )
        st.download_button("Download trades.csv", closed_df.to_csv(index=False).encode(), "trades.csv", "text/csv")
    else:
        st.info("No closed trades yet.")
