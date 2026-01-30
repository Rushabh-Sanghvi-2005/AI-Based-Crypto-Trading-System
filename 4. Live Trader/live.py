# live_sim.py  — Simulated/illusion live trading for demo/viva
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Live Trading Dashboard (Demo)", layout="wide")

# --- Auto-refresh every 5 seconds (fallback to manual button if component missing) ---
REFRESH_MS = 5_000
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_MS, key="auto_refresh_key")
except Exception:
    st.caption("Tip: pip install streamlit-autorefresh for auto refresh (5s).")

if st.button("🔁 Refresh now"):
    st.rerun()

# --- Demo config (tweak to control the illusion) ---
START_EQUITY = 1000.0
START_PRICE  = 50_000.0
DRIFT        = 0.0005   # +0.05% avg move per tick (equity tends to rise slowly)
VOL          = 0.0020   # per-tick volatility (~0.2%)
HIT_RATE     = 0.72     # probability the chosen side matches the price move (win rate driver)

# --- Session state init ---
if "equity" not in st.session_state:
    st.session_state.equity = START_EQUITY
if "price" not in st.session_state:
    st.session_state.price = START_PRICE
if "trades" not in st.session_state:
    st.session_state.trades = []  # list of dicts
if "equity_curve" not in st.session_state:
    st.session_state.equity_curve = []  # (timestamp, equity)
if "trade_num" not in st.session_state:
    st.session_state.trade_num = 0

# --- One simulated tick = one trade (closes instantly) ---
now = datetime.now()

# simulate market move
prev_price = st.session_state.price
dret = np.random.normal(loc=DRIFT, scale=VOL)  # underlying price return
new_price = prev_price * (1 + dret)

# choose side with bias to be "right" (HIT_RATE)
if (dret >= 0 and np.random.rand() < HIT_RATE) or (dret < 0 and np.random.rand() >= HIT_RATE):
    side = "BUY"
else:
    side = "SELL"

# compute trade P&L based on chosen side and price move
if side == "BUY":
    ret = (new_price - prev_price) / prev_price
else:
    ret = (prev_price - new_price) / prev_price

# optional: tiny noise to avoid all bars being the same shape
ret += np.random.normal(0, VOL*0.05)

# update state
st.session_state.price = new_price
st.session_state.equity *= (1 + ret)
st.session_state.trade_num += 1
st.session_state.trades.append({
    "trade_num": st.session_state.trade_num,
    "time": now,
    "side": side,
    "entry_px": float(prev_price),
    "exit_px":  float(new_price),
    "ret":      float(ret),
    "ret_pct":  float(ret * 100),
})
st.session_state.equity_curve.append((now, st.session_state.equity))

# --- Build frames for UI ---
trades_df = pd.DataFrame(st.session_state.trades)
equity_df = pd.DataFrame(st.session_state.equity_curve, columns=["time","equity"])

total_trades = len(trades_df)
winrate = (trades_df["ret"] > 0).mean()*100 if total_trades else 0.0
total_return = (st.session_state.equity/START_EQUITY - 1)*100

# --- Dashboard ---
st.title("Live Trading Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Closed trades", total_trades)
c2.metric("Win rate", f"{winrate:.2f}%")
c3.metric("Total return", f"{total_return:.2f}%")
c4.metric("Equity", f"${st.session_state.equity:,.2f}")

# Row 1: Equity growth + Profit per trade
col1, col2 = st.columns(2)

with col1:
    st.subheader("Equity Growth (Live)")
    if not equity_df.empty:
        line = alt.Chart(equity_df).mark_line().encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("equity:Q", title="Equity ($)")
        )
        st.altair_chart(line, use_container_width=True)
    else:
        st.info("Waiting for first point…")

with col2:
    st.subheader("Profit per Trade (Live)")
    if not trades_df.empty:
        color = alt.condition(alt.datum.ret_pct >= 0, alt.value("green"), alt.value("red"))
        bars = alt.Chart(trades_df).mark_bar().encode(
            x=alt.X("trade_num:Q", title="Trade #"),
            y=alt.Y("ret_pct:Q", title="Profit per Trade (%)"),
            color=color,
            tooltip=["trade_num","time","side","entry_px","exit_px","ret_pct"]
        )
        zero = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="gray").encode(y="y")
        st.altair_chart(bars + zero, use_container_width=True)
    else:
        st.info("No trades yet.")

# Row 2: Live table + download
st.subheader("Live Trade Records")
if not trades_df.empty:
    st.dataframe(
        trades_df[["trade_num","time","side","entry_px","exit_px","ret_pct"]].sort_values("trade_num", ascending=False).head(20),
        use_container_width=True
    )
    st.download_button("Download trades.csv", trades_df.to_csv(index=False).encode(), "trades.csv", "text/csv")
else:
    st.info("No trades yet.")
