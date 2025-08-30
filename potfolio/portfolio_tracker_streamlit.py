# portfolio_tracker_streamlit.py
# Streamlit Portfolio Tracker (equities + crypto) with near real-time quotes via yfinance
#
# How to run:
#   1) pip install -r requirements.txt
#   2) streamlit run portfolio_tracker_streamlit.py
#
# Features:
#   - Add"edit holdings via sidebar form or upload CSV
#   - Auto-refresh every N seconds
#   - Supports equities, ETFs, and popular crypto pairs via Yahoo (e.g., BTC-USD, ETH-USD)
#   - Optional average buy price for P"L
#   - Allocation pie, portfolio value line (intraday), per-position stats
#   - Multi-currency display: convert USD to INR (or vice-versa) using Yahoo FX tickers
#
# CSV format (headers are case-insensitive):
#   ticker, quantity, buy_price (optional), exchange (optional), currency (optional)
#   Examples:
#     AAPL,10,170
#     MSFT,5
#     BTC-USD,0.01
#     RELIANCE.NS,4,2600,,INR
#
# Notes:
#   - For Indian stocks, use the .NS suffix (e.g., RELIANCE.NS, TCS.NS). For London, .L, etc.
#   - Quotes from Yahoo are delayed for some markets; crypto is typically live.
#   - "Real-time" here means the app auto-pulls fresh quotes at intervals you pick.
#
import io
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
APP_TITLE = "Real‑Time-ish Portfolio Tracker"
DEFAULT_BASE_CCY = "USD" 
st.set_page_config(page_title=APP_TITLE, layout="wide")
@st.cache_data(ttl=60, show_spinner=False)
def fx_rate(base_ccy: str, display_ccy: str) -> float:
    """Fetch FX from Yahoo Finance. Example: USDINR=X returns INR per 1 USD."""
    if base_ccy == display_ccy:
        return 1.0
    pair = None
    if base_ccy == USD and display_ccy == INR:
        pair = "USDINR=X"
    elif base_ccy == INR and display_ccy == USD:
        pair = INRUSD=X
    else:
        pair = f"{base_ccy}{display_ccy}=X"
    try:
        t = yf.Ticker(pair)
        hist = t.history(period="1d", interval="1m")
        if hist is not None and len(hist) > 0:
            return float(hist["Close"].dropna().iloc[-1])
        return float(t.fast_info.last_price)
    except Exception:
        return 1.0
@st.cache_data(ttl=30, show_spinner=False)
def fetch_quote(ticker: str) -> dict:
    """Get the latest price for a ticker using multiple fallbacks."""
    data = {"ticker": ticker, "price": np.nan, "prev_close": np.nan, "currency": "USD"}
    try:
        t = yf.Ticker(ticker)
        try:
            data["price"] = float(t.fast_info.last_price)
            data["prev_close"] = float(t.fast_info.previous_close)
            data["currency"] = getattr(t.fast_info, "currency", "USD") or "USD"
            return data
        except Exception:
            pass
        hist = t.history(period="1d", interval="1m")
        if hist is not None and len(hist) > 0:
            data["price"] = float(hist["Close"].dropna().iloc[-1])
            data["prev_close"] = float(hist["Close"].dropna().iloc[0])
        info = t.info or {}
        data["currency"] = info.get("currency", data["currency"]) or data["currency"]
    except Exception:
        pass
    return data
@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_series(ticker: str) -> pd.DataFrame:
    """Return today's 1m series for the ticker."""
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df[["Close"]].rename(columns={"Close": ticker}).dropna()
    except Exception:
        pass
    return pd.DataFrame()
def sanitize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for want in [ticker, quantity, buy_price, exchange, currency]:
        for c in df.columns:
            if c.lower() == want:
                rename_map[c] = want
                break
    df = df.rename(columns=rename_map)
    if ticker not in df.columns or quantity not in df.columns:
        raise ValueError("CSV must have at least 'ticker' and 'quantity' columns.")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    if buy_price in df.columns:
        df["buy_price"] = pd.to_numeric(df.get("buy_price"), errors="coerce")
    else:
        df["buy_price"] = np.nan
    if currency not in df.columns:
        df["currency"] = np.nan
    return df
def format_currency(x, ccy):
    try:
        return f"{ccy} {x:,.2f}"
    except Exception:
        return f"{ccy} {x}"
st.title(APP_TITLE)
st.caption("Add tickers, upload a CSV, and toggle auto-refresh for live updates.")
with st.sidebar:
    st.header("Portfolio Input")
    uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
    st.markdown("Or add a single position below:")
    col1, col2 = st.columns(2)
    with col1:
        tkr = st.text_input("Ticker", value="AAPL")
    with col2:
        qty = st.number_input("Quantity", min_value=0.0, value=10.0, step=1.0, format="%f")
    buy_price = st.number_input("Avg Buy Price (optional)", min_value=0.0, value=0.0, step=0.01, format="%f")
    add_btn = st.button("Add to Table", use_container_width=True)
    st.divider()
    st.header("Display & Refresh")
    display_ccy = st.selectbox("Display currency", ["USD", "INR"], index=0)
    autorefresh = st.toggle("Auto-refresh", value=True, help="Pull fresh prices periodically.")
    refresh_secs = st.slider("Refresh every (seconds)", 5, 120, 15, help="Frequency of updates.")
    from streamlit.runtime.scriptrunner import st_autorefresh
    if autorefresh:
     st_autorefresh(interval=refresh_secs * 1000, key="auto_refresh_key")
if "holdings" not in st.session_state:
    st.session_state.holdings = pd.DataFrame(columns=["ticker", "quantity", "buy_price", "exchange", "currency"])
if uploaded is not None:
    try:
        csv_df = pd.read_csv(uploaded)
        csv_df = sanitize_holdings(csv_df)
        st.session_state.holdings = pd.concat([st.session_state.holdings, csv_df], ignore_index=True)
        st.success(f"Loaded {len(csv_df)} rows from CSV.")
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
if add_btn and tkr.strip():
    row = {
        "ticker": tkr.strip(),
        "quantity": float(qty),
        "buy_price": np.nan if buy_price == 0 else float(buy_price),
        "exchange": np.nan,
        "currency": np.nan,
    }
    st.session_state.holdings = pd.concat([st.session_state.holdings, pd.DataFrame([row])], ignore_index=True)
st.subheader("Your Holdings")
edited = st.data_editor(
    st.session_state.holdings,
    num_rows="dynamic",
    use_container_width=True,
    key="editor",
)
st.session_state.holdings = edited
if edited.empty or edited["quantity"].sum() == 0:
    st.info("Add some positions to see portfolio metrics.")
    st.stop()
rows = []
fx_cache = {}
for _, r in edited.iterrows():
    ticker = str(r["ticker"]).strip()
    qty = float(r["quantity"]) if pd.notna(r["quantity"]) else 0.0
    buy = float(r["buy_price"]) if pd.notna(r["buy_price"]) else np.nan
    q = fetch_quote(ticker)
    px = q.get("price", np.nan)
    prev = q.get("prev_close", np.nan)
    src_ccy = q.get("currency", "USD") or "USD"
    key = (src_ccy, display_ccy)
    if key not in fx_cache:
        fx_cache[key] = fx_rate(src_ccy, display_ccy)
    rate = fx_cache[key] if fx_cache[key] else 1.0
    px_ccy = px * rate if pd.notna(px) else np.nan
    prev_ccy = prev * rate if pd.notna(prev) else np.nan
    val = qty * px_ccy if pd.notna(px_ccy) else np.nan
    cost = qty * (buy * rate) if pd.notna(buy) else np.nan
    pl = (val - cost) if (pd.notna(val) and pd.notna(cost)) else np.nan
    day_chg = (px_ccy - prev_ccy) if (pd.notna(px_ccy) and pd.notna(prev_ccy)) else np.nan
    day_chg_pct = (day_chg/prev_ccy * 100.0) if (pd.notna(day_chg) and prev_ccy not in (0, np.nan)) else np.nan
    rows.append({
        "Ticker": ticker,
        "Qty": qty,
        "Price": px_ccy,
        "Prev Close": prev_ccy,
        "Value": val,
        "Avg Buy": (buy * rate) if pd.notna(buy) else np.nan,
        "P/L": pl,
        "Day Δ": day_chg,
        "Day Δ %": day_chg_pct,
        "CCY": display_ccy,
    })
table = pd.DataFrame(rows)
left, mid, right = st.columns(3)
with left:
    ttl_val = table["Value"].sum(skipna=True)
    st.metric("Portfolio Value", format_currency(ttl_val, display_ccy))
with mid:
    ttl_cost = table["Avg Buy"].mul(table["Qty"]).sum(skipna=True)
    ttl_pl = ttl_val - ttl_cost if (pd.notna(ttl_val) and pd.notna(ttl_cost)) else np.nan
    st.metric("Unrealized P/L", format_currency(ttl_pl, display_ccy))
with right:
    ttl_prev = table["Prev Close"].mul(table["Qty"]).sum(skipna=True)
    ttl_day = ttl_val - ttl_prev if (pd.notna(ttl_val) and pd.notna(ttl_prev)) else np.nan
    ttl_day_pct = (ttl_day /ttl_prev * 100.0) if (pd.notna(ttl_day) and ttl_prev not in (0, np.nan)) else np.nan
    st.metric("Day Change", f"{format_currency(ttl_day, display_ccy)} ({ttl_day_pct:.2f}% )" if pd.notna(ttl_day_pct) else "—")
st.dataframe(
    table.sort_values("Value", ascending=False).style.format({
        "Qty": "{:,.4f}",
        "Price": "{:,.2f}",
        "Prev Close": "{:,.2f}",
        "Value": "{:,.2f}",
        "Avg Buy": "{:,.2f}",
        "P/L": "{:,.2f}",
        "Day Δ": "{:,.2f}",
        "Day Δ %": "{:,.2f}",
    }),
    use_container_width=True,
    height=360,
)
st.subheader("Allocation")
alloc = table[["Ticker", "Value"]].dropna()
if not alloc.empty and alloc["Value"].sum() > 0:
    st.plotly_chart(
        {
            "data": [{
                "type": "pie",
                "labels": alloc["Ticker"].tolist(),
                "values": alloc["Value"].tolist(),
                "hole": 0.35,
            }],
            "layout": {"height": 400}
        },
        use_container_width=True
    )
else:
    st.caption("No data for allocation chart.")
st.subheader("Today: Portfolio Value (intraday)")
series_list = []
for t in table["Ticker"].unique():
    s = fetch_intraday_series(t)
    if not s.empty:
        src_quote = fetch_quote(t)
        src_ccy = src_quote.get("currency", "USD") or "USD"
        rate = fx_cache.get((src_ccy, display_ccy)) or fx_rate(src_ccy, display_ccy)
        s = s * rate
        qty = table.loc[table["Ticker"] == t, "Qty"].iloc[0]
        s = s * float(qty)
        series_list.append(s)
if series_list:
    combined = pd.concat(series_list, axis=1).dropna(how="all")
    combined["Portfolio"] = combined.sum(axis=1, skipna=True)
    st.line_chart(combined[["Portfolio"]])
else:
    st.caption("Intraday series unavailable for current tickers.")
st.divider()
st.caption("Quotes via Yahoo Finance. Some markets may be delayed. This is not investment advice.")