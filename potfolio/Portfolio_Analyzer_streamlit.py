# portfolio_tracker_streamlit.py
# Real-Time-ish Portfolio Tracker (Stocks & Crypto) using yfinance

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# App title
st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.title("Real-Time-ish Portfolio Tracker")

# ------------------- Helpers -------------------
def fetch_quote(ticker: str) -> dict:
    """Fetch latest price, prev close, currency from Yahoo Finance"""
    data = {"ticker": ticker, "price": np.nan, "prev_close": np.nan, "currency": "USD"}
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m")
        if not hist.empty:
            data["price"] = float(hist["Close"].iloc[-1])
            data["prev_close"] = float(hist["Close"].iloc[0])
        info = t.info if hasattr(t, "info") else {}
        data["currency"] = info.get("currency", "USD") or "USD"
    except Exception:
        pass
    return data

def fetch_intraday_series(ticker: str) -> pd.DataFrame:
    """Return today's 1m series for ticker"""
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if not df.empty:
            return df[["Close"]].rename(columns={"Close": ticker})
    except Exception:
        pass
    return pd.DataFrame()

def format_currency(x, ccy="USD"):
    try:
        return f"{ccy} {x:,.2f}"
    except Exception:
        return f"{ccy} {x}"

# ------------------- Sidebar Input -------------------
st.sidebar.header("Portfolio Input")

uploaded = st.sidebar.file_uploader("Upload holdings CSV", type=["csv"])

tkr = st.sidebar.text_input("Ticker", value="AAPL")
qty = st.sidebar.number_input("Quantity", min_value=0.0, value=10.0, step=1.0, format="%f")
buy_price = st.sidebar.number_input("Avg Buy Price (optional)", min_value=0.0, value=0.0, step=0.01, format="%f")
add_btn = st.sidebar.button("Add to Table", use_container_width=True)

display_ccy = st.sidebar.selectbox("Display currency", ["USD", "INR"], index=0)
autorefresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_secs = st.sidebar.slider("Refresh every (seconds)", 5, 120, 15)

# ------------------- Portfolio Data -------------------
if "holdings" not in st.session_state:
    st.session_state.holdings = pd.DataFrame(columns=["ticker", "quantity", "buy_price", "currency"])

# Load CSV
if uploaded is not None:
    try:
        csv_df = pd.read_csv(uploaded)
        csv_df.rename(columns=lambda x: x.lower(), inplace=True)
        csv_df["quantity"] = pd.to_numeric(csv_df["quantity"], errors="coerce").fillna(0.0)
        csv_df["buy_price"] = pd.to_numeric(csv_df.get("buy_price"), errors="coerce")
        if "currency" not in csv_df.columns:
            csv_df["currency"] = np.nan
        st.session_state.holdings = pd.concat([st.session_state.holdings, csv_df], ignore_index=True)
        st.success(f"Loaded {len(csv_df)} rows from CSV.")
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

# Add single row
if add_btn and tkr.strip():
    row = {
        "ticker": tkr.strip(),
        "quantity": float(qty),
        "buy_price": np.nan if buy_price == 0 else float(buy_price),
        "currency": np.nan,
    }
    st.session_state.holdings = pd.concat([st.session_state.holdings, pd.DataFrame([row])], ignore_index=True)

# Editable table
st.subheader("Your Holdings")
edited = st.data_editor(st.session_state.holdings, num_rows="dynamic", use_container_width=True)
st.session_state.holdings = edited

if edited.empty or edited["quantity"].sum() == 0:
    st.info("Add some positions to see portfolio metrics.")
    st.stop()

# ------------------- Pricing & Metrics -------------------
rows = []
for _, r in edited.iterrows():
    ticker = str(r["ticker"]).strip()
    qty = float(r["quantity"]) if pd.notna(r["quantity"]) else 0.0
    buy = float(r["buy_price"]) if pd.notna(r["buy_price"]) else np.nan

    q = fetch_quote(ticker)
    px = q.get("price", np.nan)
    prev = q.get("prev_close", np.nan)
    src_ccy = q.get("currency", "USD") or "USD"

    # Convert to display currency (simple assumption: USD ↔ INR)
    rate = 1.0
    if src_ccy != display_ccy:
        if src_ccy == "USD" and display_ccy == "INR":
            rate = 83.0
        elif src_ccy == "INR" and display_ccy == "USD":
            rate = 1/83.0
    px_ccy = px * rate if pd.notna(px) else np.nan
    prev_ccy = prev * rate if pd.notna(prev) else np.nan

    val = qty * px_ccy if pd.notna(px_ccy) else np.nan
    cost = qty * (buy * rate) if pd.notna(buy) else np.nan
    pl = val - cost if pd.notna(val) and pd.notna(cost) else np.nan
    day_chg = px_ccy - prev_ccy if pd.notna(px_ccy) and pd.notna(prev_ccy) else np.nan
    day_chg_pct = (day_chg / prev_ccy * 100.0) if pd.notna(day_chg) and prev_ccy not in (0, np.nan) else np.nan

    rows.append({
        "Ticker": ticker,
        "Qty": qty,
        "Price": px_ccy,
        "Prev Close": prev_ccy,
        "Value": val,
        "Avg Buy": (buy*rate) if pd.notna(buy) else np.nan,
        "P/L": pl,
        "Day Δ": day_chg,
        "Day Δ %": day_chg_pct,
        "CCY": display_ccy,
    })

table = pd.DataFrame(rows)

# Summary cards
left, mid, right = st.columns(3)
with left:
    st.metric("Portfolio Value", format_currency(table["Value"].sum(skipna=True), display_ccy))
with mid:
    total_cost = table["Avg Buy"].mul(table["Qty"]).sum(skipna=True)
    st.metric("Unrealized P/L", format_currency(table["Value"].sum(skipna=True) - total_cost, display_ccy))
with right:
    total_prev = table["Prev Close"].mul(table["Qty"]).sum(skipna=True)
    day_val = table["Value"].sum(skipna=True) - total_prev
    day_pct = (day_val / total_prev * 100.0) if total_prev != 0 else np.nan
    st.metric("Day Change", f"{format_currency(day_val, display_ccy)} ({day_pct:.2f}%)" if pd.notna(day_pct) else "—")
st.dataframe(table.sort_values("Value", ascending=False).style.format({
    "Qty": "{:,.4f}",
    "Price": "{:,.2f}",
    "Prev Close": "{:,.2f}",
    "Value": "{:,.2f}",
    "Avg Buy": "{:,.2f}",
    "P/L": "{:,.2f}",
    "Day Δ": "{:,.2f}",
    "Day Δ %": "{:,.2f}",
}), use_container_width=True, height=360)

if autorefresh:
    st.experimental_rerun()  