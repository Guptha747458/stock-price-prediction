import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# Import custom modules
from src.data_ingestion import fetch_stock_data, get_ticker_info, search_tickers
from src.features import add_technical_indicators, prepare_regression_data, prepare_lstm_data, create_lagged_features
from src.models import (
    train_random_forest, 
    train_xgboost, 
    PyTorchLSTMRegressor, 
    calculate_metrics
)
from src.trading import generate_trading_signals

# --- Page Configuration ---
st.set_page_config(
    page_title="AlphaPulse - Stock Price Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom Styling (Dark Glassmorphism UI) ---
_CSS = """
<style>
/* --- AlphaPulse 2.0 — Aurora Glass Design --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 800px at 80% -10%, rgba(120,119,198,0.25), transparent 60%),
                radial-gradient(900px 600px at -10% 20%, rgba(56,189,248,0.18), transparent 55%),
                #070B14;
    font-family: 'Inter', sans-serif;
    color: #E6EDF7;
}

/* subtle animated grid */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 32px 32px;
    mask-image: radial-gradient(ellipse at center, black 40%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.main-title {
    font-family: 'Sora', sans-serif;
    background: linear-gradient(135deg, #22D3EE 0%, #818CF8 45%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
    text-shadow: 0 0 30px rgba(129,140,248,0.25);
}

.subtitle {
    color: #9AA8BD;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #0A0F1A 100%);
    border-right: 1px solid rgba(148,163,184,0.12);
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

/* Cards */
.metric-card {
    position: relative;
    background: rgba(17, 25, 40, 0.65);
    backdrop-filter: blur(16px) saturate(140%);
    -webkit-backdrop-filter: blur(16px) saturate(140%);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 10px 30px rgba(2,6,23,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    overflow: hidden;
}
.metric-card::after {
    content: "";
    position: absolute;
    inset: -1px;
    background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(192,132,252,0.22));
    opacity: 0;
    transition: opacity 0.22s ease;
    z-index: 0;
    border-radius: 16px;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56,189,248,0.35);
    box-shadow: 0 14px 40px rgba(56,189,248,0.12);
}
.metric-card:hover::after { opacity: 0.15; }

.metric-label {
    font-size: 0.78rem;
    color: #8FA0B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 2.15rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.1;
}

/* Recommendation badges */
.rec-badge {
    display: inline-block;
    padding: 0.55rem 1.4rem;
    font-size: 1.35rem;
    font-weight: 800;
    border-radius: 9999px;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 6px 20px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
}
.rec-strong-buy  { background: linear-gradient(135deg, #059669, #10B981); color: white; }
.rec-buy         { background: linear-gradient(135deg, #047857, #059669); color: white; }
.rec-hold        { background: linear-gradient(135deg, #334155, #475569); color: #E2E8F0; }
.rec-sell        { background: linear-gradient(135deg, #B91C1C, #EF4444); color: white; }
.rec-strong-sell { background: linear-gradient(135deg, #7F1D1D, #B91C1C); color: white; }

/* Status */
.status-ok { color: #34D399; font-weight: 600; }

/* Buttons and inputs */
.stButton > button {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    color: #E2E8F0;
    border: 1px solid rgba(148,163,184,0.2);
    border-radius: 10px;
    padding: 0.55rem 1.1rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    border-color: rgba(56,189,248,0.5);
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15);
    transform: translateY(-1px);
}

/* Dataframes and tables */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,0.14);
}

/* Plotly chart containers */
.js-plotly-plot {
    border-radius: 14px;
    overflow: hidden;
}

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #1F2A44; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #2A3A5E; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# --- App Header ---
st.markdown('<div class="main-title">AlphaPulse</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Stock Prediction & Quantitative Analysis Dashboard</div>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.markdown("### ⚙️ Control Dashboard")

# ── Global Stock Catalogue ────────────────────────────────────────────────────
GLOBAL_STOCKS = {
    "🇺🇸 US Tech & Growth": [
        ("AAPL",  "Apple Inc."),
        ("MSFT",  "Microsoft Corp."),
        ("NVDA",  "NVIDIA Corp."),
        ("TSLA",  "Tesla Inc."),
        ("GOOGL", "Alphabet (Google)"),
        ("AMZN",  "Amazon.com"),
        ("META",  "Meta Platforms"),
        ("NFLX",  "Netflix"),
        ("AMD",   "Advanced Micro Devices"),
        ("INTC",  "Intel Corp."),
    ],
    "🇺🇸 US Finance & Industrials": [
        ("JPM",   "JPMorgan Chase"),
        ("GS",    "Goldman Sachs"),
        ("BAC",   "Bank of America"),
        ("BRK-B", "Berkshire Hathaway B"),
        ("BA",    "Boeing"),
        ("GE",    "GE Aerospace"),
        ("XOM",   "ExxonMobil"),
        ("CVX",   "Chevron"),
        ("PFE",   "Pfizer"),
        ("JNJ",   "Johnson & Johnson"),
    ],
    "🇮🇳 Indian Markets (NSE)": [
        ("RELIANCE.NS",  "Reliance Industries"),
        ("TCS.NS",       "Tata Consultancy"),
        ("INFY.NS",      "Infosys"),
        ("HDFCBANK.NS",  "HDFC Bank"),
        ("WIPRO.NS",     "Wipro Ltd."),
        ("ICICIBANK.NS", "ICICI Bank"),
        ("TATAMOTORS.NS","Tata Motors"),
        ("BAJFINANCE.NS","Bajaj Finance"),
        ("ADANIENT.NS",  "Adani Enterprises"),
        ("SBIN.NS",      "State Bank of India"),
    ],
    "🇬🇧🇩🇪🇫🇷 European Markets": [
        ("SAP",    "SAP SE (NYSE)"),
        ("ASML",   "ASML Holding"),
        ("LVMH.PA","LVMH (Euronext)"),
        ("SIE.DE", "Siemens AG (XETRA)"),
        ("BP",     "BP plc (NYSE)"),
        ("SHEL",   "Shell plc (NYSE)"),
        ("NESN.SW","Nestlé SA (SIX)"),
        ("AZN",    "AstraZeneca (NYSE)"),
        ("VOW3.DE","Volkswagen (XETRA)"),
        ("ULVR.L", "Unilever (LSE)"),
    ],
    "🌏 Asian & Emerging Markets": [
        ("9988.HK",  "Alibaba (HKEX)"),
        ("700.HK",   "Tencent (HKEX)"),
        ("005930.KS","Samsung Electronics"),
        ("7203.T",   "Toyota Motor (TSE)"),
        ("9984.T",   "SoftBank Group (TSE)"),
        ("BABA",     "Alibaba (NYSE ADR)"),
        ("BIDU",     "Baidu (NYSE ADR)"),
        ("PDD",      "PDD Holdings"),
        ("SE",       "Sea Limited"),
        ("GRAB",     "Grab Holdings"),
    ],
    "₿ Crypto / Digital Assets": [
        ("BTC-USD",  "Bitcoin"),
        ("ETH-USD",  "Ethereum"),
        ("BNB-USD",  "BNB (Binance Coin)"),
        ("SOL-USD",  "Solana"),
        ("XRP-USD",  "XRP"),
        ("ADA-USD",  "Cardano"),
        ("AVAX-USD", "Avalanche"),
        ("DOGE-USD", "Dogecoin"),
        ("LINK-USD", "Chainlink"),
        ("DOT-USD",  "Polkadot"),
    ],
    "📊 ETFs & Indices": [
        ("SPY",  "S&P 500 ETF (SPDR)"),
        ("QQQ",  "NASDAQ-100 ETF"),
        ("DIA",  "Dow Jones ETF"),
        ("IWM",  "Russell 2000 ETF"),
        ("GLD",  "Gold ETF (SPDR)"),
        ("SLV",  "Silver ETF"),
        ("VTI",  "Vanguard Total Market ETF"),
        ("EEM",  "Emerging Markets ETF"),
        ("ARKK", "ARK Innovation ETF"),
        ("TLT",  "20+ Year Treasury Bond ETF"),
    ],
    "🏗️ Custom (Any Exchange)": [],   # sentinel for custom mode
}

# ── Stock Selection Mode ──────────────────────────────────────────────────────
st.sidebar.markdown("### 🌍 Stock Selection")

# ---- Live Search by company name ----------------------------------------
with st.sidebar.expander("🔍 Search by Company Name", expanded=False):
    search_query = st.text_input(
        "Type company name or partial ticker:",
        value="",
        placeholder="e.g. Apple, Samsung, Tata, Bitcoin…",
        key="search_query_input"
    )
    do_search = st.button("Search", key="search_btn", use_container_width=True)

    # If the user typed a query and pressed search, perform the search and store in session state
    if do_search and search_query.strip():
        with st.spinner("Searching Yahoo Finance…"):
            st.session_state["search_results"] = search_tickers(search_query.strip(), max_results=12)
            st.session_state["search_query_last"] = search_query.strip()

    # Display results if they exist in session state
    if "search_results" in st.session_state and st.session_state["search_results"]:
        results = st.session_state["search_results"]
        st.success(f"Results for '{st.session_state.get('search_query_last', '')}':")
        # Build selectable labels
        result_labels = [
            f"{r['symbol']}  —  {r['name']}  [{r['exchange']} · {r['type_display']}]"
            for r in results
        ]
        chosen_label = st.radio(
            "Pick a result to use it:",
            result_labels,
            key="search_result_radio"
        )
        if st.button("✅ Use this ticker", key="use_search_ticker", use_container_width=True):
            chosen_symbol = chosen_label.split("  —  ")[0].strip()
            st.session_state["custom_ticker"]   = chosen_symbol
            st.session_state["force_custom"]    = True   # flag to switch to Custom mode
            st.session_state["has_run"]         = True   # Run automatically on search select!
            st.session_state["last_seen_ticker"] = chosen_symbol.upper()
            # Clear search state to close the panel cleanly
            del st.session_state["search_results"]
            st.rerun()
    elif do_search:
        st.warning("No results found. Try a different name or check spelling.")

# If a search result was just applied, force Custom mode
if st.session_state.get("force_custom"):
    default_category_index = list(GLOBAL_STOCKS.keys()).index("🏗️ Custom (Any Exchange)")
else:
    default_category_index = 0

market_category = st.sidebar.selectbox(
    "Market / Category:",
    list(GLOBAL_STOCKS.keys()),
    index=default_category_index,
    key="market_category"
)
# Reset the force flag once the widget has rendered
st.session_state["force_custom"] = False

if market_category == "🏗️ Custom (Any Exchange)":
    # Full custom entry mode
    st.sidebar.markdown(
        "<small style='color:#94A3B8'>Enter any valid Yahoo Finance ticker symbol.</small>",
        unsafe_allow_html=True
    )
    ticker = st.sidebar.text_input(
        "Ticker Symbol:",
        value=st.session_state.get("custom_ticker", "AAPL"),
        placeholder="e.g. AAPL, RELIANCE.NS, BTC-USD",
        key="custom_ticker_input"
    ).strip().upper()
    st.session_state["custom_ticker"] = ticker

    # Exchange format quick-reference
    with st.sidebar.expander("📌 Ticker Format Guide (click to expand)"):
        st.markdown("""
| Exchange | Suffix | Example |
|---|---|---|
| NYSE / NASDAQ | *(none)* | `AAPL`, `TSLA` |
| NSE India | `.NS` | `RELIANCE.NS` |
| BSE India | `.BO` | `RELIANCE.BO` |
| London (LSE) | `.L` | `ULVR.L` |
| XETRA (Germany)| `.DE` | `SIE.DE` |
| Euronext Paris | `.PA` | `LVMH.PA` |
| Tokyo (TSE) | `.T` | `7203.T` |
| HK Exchange | `.HK` | `700.HK` |
| Korea (KSE) | `.KS` | `005930.KS` |
| SIX Swiss | `.SW` | `NESN.SW` |
| Crypto (Yahoo) | `-USD` | `BTC-USD` |
        """)
else:
    # Pick from curated list for selected category
    # Key includes the category name so switching markets resets the index to 0
    stock_options = GLOBAL_STOCKS[market_category]
    display_labels = [f"{sym}  —  {name}" for sym, name in stock_options]
    selected_label = st.sidebar.selectbox(
        "Select Stock:",
        display_labels,
        index=0,
        key=f"stock_picker_{market_category}"   # <-- category-scoped key fixes reset bug
    )
    ticker = selected_label.split("  —  ")[0].strip().upper()

# Final safety guard – never pass an empty ticker to the data loader
if not ticker:
    ticker = "AAPL"

st.sidebar.markdown(
    f"<div style='background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.25);border-radius:8px;padding:0.5rem 0.8rem;margin-top:0.3rem;'>"
    f"<span style='color:#94A3B8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;'>Active Ticker</span><br>"
    f"<span style='color:#38BDF8;font-size:1.3rem;font-weight:700;'>{ticker}</span>"
    f"</div>",
    unsafe_allow_html=True
)

# ── Date range selection ──────────────────────────────────────────────────────
st.sidebar.markdown("### 📅 Date Range")
today = datetime.date.today()
three_years_ago = today - datetime.timedelta(days=3 * 365)
start_date = st.sidebar.date_input("Start Date:", value=three_years_ago)
end_date = st.sidebar.date_input("End Date:", value=today)

# Model configuration options
st.sidebar.markdown("### 🤖 Model Settings")
sequence_length = st.sidebar.slider("LSTM Sequence Length (Days):", min_value=5, max_value=30, value=10)
lstm_epochs = st.sidebar.slider("LSTM Training Epochs:", min_value=5, max_value=30, value=15)
train_split = st.sidebar.slider("Train/Test Split Ratio:", min_value=0.6, max_value=0.9, value=0.8, step=0.05)

st.sidebar.markdown("---")
run_button = st.sidebar.button("⚡ Fetch & Train Models", use_container_width=True)

# --- Main App Execution Logic ---

# --- Main App Execution Logic ---

@st.cache_data(ttl=3600)
def load_data(ticker_symbol, start, end):
    """
    Helper function to load data and cache it.
    """
    df_raw = fetch_stock_data(ticker_symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    df_with_indicators = add_technical_indicators(df_raw)
    info = get_ticker_info(ticker_symbol)
    return df_raw, df_with_indicators, info

# Initialize execution state
if "has_run" not in st.session_state:
    st.session_state["has_run"] = False

if "last_seen_ticker" not in st.session_state:
    st.session_state["last_seen_ticker"] = ticker
else:
    # If the user changed the active stock selection in the sidebar, auto-trigger execution!
    if st.session_state["last_seen_ticker"] != ticker:
        st.session_state["has_run"] = True
        st.session_state["last_seen_ticker"] = ticker

# If user clicks the run button, trigger execution
if run_button:
    st.session_state["has_run"] = True
    st.session_state["last_seen_ticker"] = ticker

# Render Welcome Landing Page if not run yet
if not st.session_state["has_run"]:
    st.markdown(
'<div style="background: rgba(30, 41, 59, 0.45); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 2.5rem; margin-top: 1rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">'
'<h2 style="color: #38BDF8; font-weight: 800; font-size: 2.2rem; margin-bottom: 0.5rem; font-family: \'Outfit\', sans-serif;">Welcome to AlphaPulse! 📈</h2>'
'<p style="color: #94A3B8; font-size: 1.15rem; margin-bottom: 2rem;">An advanced, AI-powered stock prediction &amp; quantitative analysis platform utilizing machine learning models to forecast price trends.</p>'
'<h3 style="color: #E2E8F0; font-size: 1.3rem; margin-bottom: 1rem; font-family: \'Outfit\', sans-serif;">🎯 Core Features &amp; Capabilities</h3>'
'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2.5rem;">'
'<div style="background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 1.2rem;">'
'<h4 style="color: #818CF8; margin-bottom: 0.5rem; font-size: 1.05rem;">🧠 Machine Learning Predictions</h4>'
'<p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.4;">Trains three separate models in real-time on historical prices: <b>Random Forest</b>, <b>XGBoost</b>, and a deep learning <b>PyTorch LSTM Network</b>.</p>'
'</div>'
'<div style="background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 1.2rem;">'
'<h4 style="color: #C084FC; margin-bottom: 0.5rem; font-size: 1.05rem;">📊 Technical Charts &amp; Overlays</h4>'
'<p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.4;">Generates interactive candlestick charts with overlays for SMAs, EMAs, Bollinger Bands, and RSI indicators.</p>'
'</div>'
'<div style="background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 1.2rem;">'
'<h4 style="color: #38BDF8; margin-bottom: 0.5rem; font-size: 1.05rem;">🔮 Autoregressive Forecasts</h4>'
'<p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.4;">Rolls predictions recursively to project a <b>7-day price forecast</b> showing trends and model consensus.</p>'
'</div>'
'<div style="background: rgba(15, 23, 42, 0.3); border: 1px solid rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 1.2rem;">'
'<h4 style="color: #34D399; margin-bottom: 0.5rem; font-size: 1.05rem;">🚦 Consensus Trading Signals</h4>'
'<p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.4;">Combines models and technical indicator states to compute quantitative buy/hold/sell trading signals.</p>'
'</div>'
'</div>'
'<div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(129, 140, 248, 0.1) 100%); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 12px; padding: 1.5rem; text-align: center;">'
'<p style="color: #F8FAFC; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.3rem;">👈 How to start:</p>'
'<p style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 0;">Select a stock from the left sidebar or search by company name, then click <b>⚡ Fetch &amp; Train Models</b> to load the interactive dashboard.</p>'
'</div>'
'</div>',
        unsafe_allow_html=True
    )
    st.stop()  # Stop execution early so models don't train and tabs don't show yet

# Execute pipeline
data_loaded = False
try:
    with st.spinner(f"Ingesting market data for **{ticker}**..."):
        df_raw, df, info = load_data(ticker, start_date, end_date)
        data_loaded = True
except Exception as e:
    st.error(f"Error loading stock data: {str(e)}")
    st.info("Please verify the ticker symbol and date range in the sidebar. You can search for the correct ticker using the 'Search by Company Name' tool above.")

if data_loaded:
    # Display Ticker Information Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## {info['longName']} ({info['symbol']})")
        st.markdown(f"**Sector**: {info['sector']} | **Industry**: {info['industry']} | **Currency**: {info['currency']}")
    with col2:
        if info.get('logo_url'):
            st.markdown(f'<img src="{info["logo_url"]}" width="80" style="float:right; border-radius:8px;">', unsafe_allow_html=True)
            
    # Business Summary expander
    with st.expander("📖 View Company Description"):
        st.write(info['longBusinessSummary'])
        if info['website'] != 'N/A':
            st.markdown(f"[Visit Company Website]({info['website']})")

    # Display Current Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    current_price = info['currentPrice'] if info['currentPrice'] is not None else df['Close'].iloc[-1]
    
    # Calculate price change compared to previous day
    prev_close = df['Close'].iloc[-2]
    day_change = current_price - prev_close
    day_change_pct = (day_change / prev_close) * 100.0
    
    change_color = "#10B981" if day_change >= 0 else "#EF4444"
    change_symbol = "▲" if day_change >= 0 else "▼"
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">{info['currency']} {current_price:,.2f}</div>
            <div style="color: {change_color}; font-size: 0.9rem; font-weight:600; margin-top:0.2rem;">
                {change_symbol} {abs(day_change):.2f} ({day_change_pct:+.2f}%) Today
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col2:
        day_low = info['dayLow'] if info['dayLow'] is not None else df['Low'].iloc[-1]
        day_high = info['dayHigh'] if info['dayHigh'] is not None else df['High'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Daily Range</div>
            <div class="metric-value" style="font-size: 1.6rem;">{day_low:,.2f} - {day_high:,.2f}</div>
            <div style="color: #94A3B8; font-size: 0.9rem; margin-top:0.6rem;">
                Low - High
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col3:
        volume = info['volume'] if info['volume'] is not None else df['Volume'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volume</div>
            <div class="metric-value">{volume:,.0f}</div>
            <div style="color: #94A3B8; font-size: 0.9rem; margin-top:0.6rem;">
                Shares Traded Today
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m_col4:
        market_cap = info['marketCap']
        market_cap_str = f"{market_cap:,.0f}" if market_cap else "N/A"
        if market_cap and market_cap >= 1e12:
            market_cap_str = f"{market_cap/1e12:.2f}T"
        elif market_cap and market_cap >= 1e9:
            market_cap_str = f"{market_cap/1e9:.2f}B"
        elif market_cap and market_cap >= 1e6:
            market_cap_str = f"{market_cap/1e6:.2f}M"
            
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Market Capitalization</div>
            <div class="metric-value">{market_cap_str}</div>
            <div style="color: #94A3B8; font-size: 0.9rem; margin-top:0.6rem;">
                Valuation ({info['currency']})
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Training the Models ---
    # We will compute training inside a cached function to avoid retraining on tab switches
    @st.cache_data(show_spinner=False)
    def prepare_all_data(df_processed, seq_len, split_ratio):
        """Cache-friendly: returns serialisable data (numpy arrays + scalers)."""
        lag_days = 5
        reg_data = prepare_regression_data(df_processed, lag_days=lag_days, train_split=split_ratio)
        lstm_data = prepare_lstm_data(df_processed, sequence_length=seq_len, train_split=split_ratio)
        return reg_data, lstm_data, lag_days

    @st.cache_resource(show_spinner=False)
    def train_all_models(df_hash_key, seq_len, epochs_num, split_ratio):
        """Cache-friendly: returns non-serialisable model objects.
        df_hash_key is a tuple (ticker, start, end) so Streamlit can invalidate on param change.
        """
        reg_data, lstm_data, _lag_days = prepare_all_data(
            st.session_state['_df_cache'], seq_len, split_ratio
        )
        X_tr_reg, y_tr_reg, X_te_reg, y_te_reg, feat_scaler_reg, scaler_reg, feat_cols_reg = reg_data
        X_tr_lstm, y_tr_lstm, X_te_lstm, y_te_lstm, feat_scaler_lstm, scaler_lstm, feat_cols_lstm = lstm_data
        
        rf_model = train_random_forest(X_tr_reg, y_tr_reg)
        xgb_model = train_xgboost(X_tr_reg, y_tr_reg)
        
        lstm_model = PyTorchLSTMRegressor(
            input_size=len(feat_cols_lstm),
            hidden_size=32,
            num_layers=1,
            epochs=epochs_num,
            batch_size=32
        )
        lstm_model.fit(X_tr_lstm, y_tr_lstm, val_data=(X_te_lstm, y_te_lstm))
        
        return {'rf': rf_model, 'xgb': xgb_model, 'lstm': lstm_model}

    # Cache the df so the resource-cached train function can access it
    st.session_state['_df_cache'] = df

    # Initialize session state for button trigger
    if 'train_trigger' not in st.session_state:
        st.session_state.train_trigger = False

    if run_button:
        st.session_state.train_trigger = True
        # Clear both caches to force full retrain
        st.cache_resource.clear()
        st.cache_data.clear()

    # Derive hash key from current parameters to bust model cache on param change
    _cache_key = (ticker, str(start_date), str(end_date), sequence_length, lstm_epochs, train_split)

    with st.spinner("🧠 Training Random Forest, XGBoost, and PyTorch LSTM models on stock history..."):
        reg_data, lstm_data, lag_days = prepare_all_data(df, sequence_length, train_split)
        models_dict = train_all_models(_cache_key, sequence_length, lstm_epochs, train_split)
        st.toast("Models trained successfully!", icon="✅")

    # Unpack model data
    rf_model = models_dict['rf']
    xgb_model = models_dict['xgb']
    lstm_model = models_dict['lstm']

    # Unpack data — use .copy() to prevent inadvertent mutation of cached arrays
    X_tr_reg, y_tr_reg, X_te_reg, y_te_reg, feat_scaler_reg, scaler_reg, feat_cols_reg = reg_data
    X_tr_reg, y_tr_reg, X_te_reg, y_te_reg = X_tr_reg.copy(), y_tr_reg.copy(), X_te_reg.copy(), y_te_reg.copy()

    X_tr_lstm, y_tr_lstm, X_te_lstm, y_te_lstm, feat_scaler_lstm, scaler_lstm, feat_cols_lstm = lstm_data
    X_tr_lstm, y_tr_lstm, X_te_lstm, y_te_lstm = X_tr_lstm.copy(), y_tr_lstm.copy(), X_te_lstm.copy(), y_te_lstm.copy()

    # Align clean datasets and split indices for metric calculations and plotting
    df_lags_clean = create_lagged_features(df, lag_days=lag_days)
    df_clean_reg = df_lags_clean.dropna().iloc[:-1]
    split_idx_reg = int(len(df_clean_reg) * train_split)

    df_clean_lstm = df.dropna().iloc[:-1]
    split_idx_lstm = int((len(df_clean_lstm) - sequence_length) * train_split)

    # --- Generate Predictions for Test Set & Tomorrow ---
    
    # 1. XGBoost & Random Forest test set predictions
    y_pred_rf_scaled = rf_model.predict(X_te_reg)
    y_pred_xgb_scaled = xgb_model.predict(X_te_reg)
    
    y_pred_rf = scaler_reg.inverse_transform(y_pred_rf_scaled.reshape(-1, 1)).ravel()
    y_pred_xgb = scaler_reg.inverse_transform(y_pred_xgb_scaled.reshape(-1, 1)).ravel()
    y_true_reg = scaler_reg.inverse_transform(y_te_reg.reshape(-1, 1)).ravel()
    
    # 2. LSTM test set predictions
    y_pred_lstm_scaled = lstm_model.predict(X_te_lstm)
    y_pred_lstm = scaler_lstm.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).ravel()
    y_true_lstm = scaler_lstm.inverse_transform(y_te_lstm.reshape(-1, 1)).ravel()
    
    # Predict tomorrow's price (1 day ahead)
    # A. Regression tomorrow input
    # Get the very latest row containing features
    df_lags = create_lagged_features(df, lag_days=lag_days)
    # Use the last fully-populated row (after dropna) to avoid NaN features crashing the scaler
    latest_feat_reg_df = df_lags[feat_cols_reg].dropna().iloc[-1:]
    latest_feat_reg_scaled = feat_scaler_reg.transform(latest_feat_reg_df.values)
    
    pred_tomorrow_rf_scaled = rf_model.predict(latest_feat_reg_scaled)
    pred_tomorrow_rf = scaler_reg.inverse_transform(pred_tomorrow_rf_scaled.reshape(-1, 1))[0][0]
    
    pred_tomorrow_xgb_scaled = xgb_model.predict(latest_feat_reg_scaled)
    pred_tomorrow_xgb = scaler_reg.inverse_transform(pred_tomorrow_xgb_scaled.reshape(-1, 1))[0][0]
    
    # B. LSTM tomorrow input (requires last sequence_length days of scaled features)
    # Use dropna() to guard against NaN-containing tail rows from rolling-window indicators
    df_no_nan = df.dropna()
    latest_sequence_lstm = df_no_nan[feat_cols_lstm].iloc[-sequence_length:].values
    latest_sequence_lstm_scaled = feat_scaler_lstm.transform(latest_sequence_lstm)
    
    pred_tomorrow_lstm_scaled = lstm_model.predict(np.expand_dims(latest_sequence_lstm_scaled, axis=0))
    pred_tomorrow_lstm = scaler_lstm.inverse_transform(pred_tomorrow_lstm_scaled.reshape(-1, 1))[0][0]

    # --- App Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "📈 Technical Charts & Indicators", 
        "🔮 Forecast & Consensus Recommendations", 
        "📊 Model Comparison & Analytics"
    ])

    # ==================== TAB 1: TECHNICAL ANALYSIS ====================
    with tab1:
        st.markdown("### Technical Analysis Dashboard")
        
        # User toggles for indicators
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            show_sma = st.checkbox("Overlay SMAs (10, 50, 200)", value=True)
        with col_t2:
            show_ema = st.checkbox("Overlay EMAs (10, 50, 200)", value=False)
        with col_t3:
            show_bb = st.checkbox("Overlay Bollinger Bands (20)", value=True)
            
        # Create Plotly Chart with Subplots (Candlestick, RSI, Volume)
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 1. Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Stock Price"
            ),
            row=1, col=1
        )
        
        # Overlays
        colors_sma = {10: "#38BDF8", 50: "#3B82F6", 200: "#1D4ED8"}
        colors_ema = {10: "#F472B6", 50: "#EC4899", 200: "#BE185D"}
        
        if show_sma:
            for w in [10, 50, 200]:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[f'SMA_{w}'], name=f'SMA {w}', line=dict(color=colors_sma[w], width=1.5)),
                    row=1, col=1
                )
                
        if show_ema:
            for w in [10, 50, 200]:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[f'EMA_{w}'], name=f'EMA {w}', line=dict(color=colors_ema[w], width=1.5, dash='dash')),
                    row=1, col=1
                )
                
        if show_bb:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color="#10B981", width=1, dash='dot')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle (SMA 20)', line=dict(color="#F59E0B", width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color="#EF4444", width=1, dash='dot')),
                row=1, col=1
            )
            
        # 2. RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color="#A78BFA", width=1.5)),
            row=2, col=1
        )
        # RSI threshold lines
        fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10B981", line_width=1, row=2, col=1)
        
        # 3. Volume
        colors_vol = ['#EF4444' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#10B981' for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol, opacity=0.7),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Volume_MA_10'], name='Volume MA 10', line=dict(color="#F59E0B", width=1.5)),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.08,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            ),
            margin=dict(t=30, b=80, l=50, r=50)
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[10, 90])
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        
        st.plotly_chart(fig, config={'displayModeBar': True, 'scrollZoom': True})

    # ==================== TAB 2: FORECAST & RECOMMENDATIONS ====================
    with tab2:
        st.markdown("### Consensus Recommendation Dashboard")
        
        # We generate consensus based on the average prediction of our models
        avg_pred_tomorrow = (pred_tomorrow_rf + pred_tomorrow_xgb + pred_tomorrow_lstm) / 3.0
        
        # Run recommendation generator on the LSTM model (or best consensus)
        recommendation_data = generate_trading_signals(df, avg_pred_tomorrow, "Models Consensus (Avg)")
        
        rec = recommendation_data['recommendation']
        score = recommendation_data['score']
        reasons = recommendation_data['reasons']
        
        # Rec badge color class mapping
        rec_class = "rec-hold"
        if rec == "Strong Buy":
            rec_class = "rec-strong-buy"
        elif rec == "Buy":
            rec_class = "rec-buy"
        elif rec == "Sell":
            rec_class = "rec-sell"
        elif rec == "Strong Sell":
            rec_class = "rec-strong-sell"
            
        rec_col1, rec_col2 = st.columns([1, 2])
        
        with rec_col1:
            st.markdown("#### Trading Consensus Rating")
            st.markdown(f'<div class="rec-badge {rec_class}">{rec}</div>', unsafe_allow_html=True)
            st.markdown(f"<br>**Quantitative Sentiment Score**: `{score:+.1f} / +5.0`", unsafe_allow_html=True)
            
            # Sub-metrics progress bars
            st.markdown("#### Technical Breakdown Score")
            st.write(f"RSI Status: **{recommendation_data['metrics']['rsi']:.1f}**")
            st.progress(float(np.clip(recommendation_data['metrics']['rsi'] / 100.0, 0.0, 1.0)))
            
            bb_upper = recommendation_data['metrics']['bb_upper']
            bb_lower = recommendation_data['metrics']['bb_lower']
            curr_pr = recommendation_data['metrics']['current_price']
            bb_percent = (curr_pr - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            st.write(f"Bollinger Band Position: **{bb_percent * 100.0:.1f}%** (0%=Lower, 100%=Upper)")
            st.progress(float(np.clip(bb_percent, 0.0, 1.0)))
            
        with rec_col2:
            st.markdown("#### Quant Analysis & Prediction Signal Details")
            for r in reasons:
                st.markdown(r)
                
        st.markdown("---")
        st.markdown("### Model Predictions vs Actual Stock Prices")
        
        # Alignment of test sets
        # Regression test set dates correspond to the last len(y_true_reg) index in df_clean
        # Let's extract the actual datetime indices for both test sets
        test_dates_reg = df_clean_reg.index[split_idx_reg:]
        
        # LSTM sequence creates elements of size: len(df_clean_lstm) - sequence_length
        test_dates_lstm = df_clean_lstm.index[sequence_length + split_idx_lstm:]
        
        # Plotly chart comparison
        pred_fig = go.Figure()
        
        # Actual prices (Historical Test Set values)
        pred_fig.add_trace(go.Scatter(x=test_dates_reg, y=y_true_reg, name="Actual Close", line=dict(color="#F8FAFC", width=2.5)))
        
        # Model predictions
        pred_fig.add_trace(go.Scatter(x=test_dates_reg, y=y_pred_rf, name="Random Forest Predict", line=dict(color="#F59E0B", width=1.5, dash='dash')))
        pred_fig.add_trace(go.Scatter(x=test_dates_reg, y=y_pred_xgb, name="XGBoost Predict", line=dict(color="#10B981", width=1.5, dash='dash')))
        pred_fig.add_trace(go.Scatter(x=test_dates_lstm, y=y_pred_lstm, name="PyTorch LSTM Predict", line=dict(color="#38BDF8", width=1.5)))
        
        pred_fig.update_layout(
            height=500,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title=f"Close Price ({info['currency']})",
            margin=dict(t=30, b=80, l=50, r=50),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(pred_fig, config={'displayModeBar': True, 'scrollZoom': True})
        
        st.markdown("---")
        st.markdown("### Next 7 Days Rolling Price Forecast")
        
        # Implement rolling 7 day prediction forecast for all 3 models
        # For simplicity and robust display, we roll the Close price predictions.
        # LSTM rolling forecast:
        # We start with the latest sequence of length `sequence_length` of features.
        # When we predict tomorrow, we append the predicted Close to our sequence, shift, 
        # and recompute technical indicators (or keep them fixed/updated).
        # To make it robust and prevent errors, we can generate a multi-step forecast using a simpler autoregressive lag method.
        # Let's perform rolling predictions:
        
        future_days = 7
        future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
        
        # LSTM multi-day rolling forecast:
        lstm_rolling_seq = latest_sequence_lstm_scaled.copy() # shape (sequence_length, num_features)
        lstm_forecast = []
        
        # For indicators recalculation, let's keep them constant and just roll the Close price column (index 3 in features: Open, High, Low, Close, Volume... etc. Let's find Close index)
        close_idx = feat_cols_lstm.index('Close')
        
        for _ in range(future_days):
            pred_scaled = lstm_model.predict(np.expand_dims(lstm_rolling_seq, axis=0))
            lstm_forecast.append(pred_scaled[0])
            
            # Roll sequence: remove first, add new row where 'Close' is the predicted scaled value
            new_row = lstm_rolling_seq[-1].copy()
            new_row[close_idx] = pred_scaled[0] # Put prediction in the close position
            lstm_rolling_seq = np.vstack([lstm_rolling_seq[1:], new_row])
            
        lstm_forecast_prices = scaler_lstm.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).ravel()
        
        # XGBoost rolling forecast:
        xgb_rolling_features = latest_feat_reg_df.values.copy() # shape (1, num_features)
        xgb_forecast = []
        # Find features index for lags: 'Close_Lag_1', 'Close_Lag_2' etc.
        lag_indices = [feat_cols_reg.index(f'Close_Lag_{i}') for i in range(1, lag_days + 1)]
        close_idx_reg = feat_cols_reg.index('Close')
        
        current_xgb_feat = latest_feat_reg_df.values[0].copy()
        
        for _ in range(future_days):
            scaled_feat = feat_scaler_reg.transform([current_xgb_feat])
            pred_scaled = xgb_model.predict(scaled_feat)
            pred_price = scaler_reg.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            xgb_forecast.append(pred_price)
            
            # Roll lags: shift Lag_i to Lag_{i+1}
            # current_xgb_feat has lags in the order: Lag_1, Lag_2, Lag_3...
            # Lag_1 becomes the new predicted price, Lag_2 becomes the old Lag_1, and so on.
            for i in range(lag_days - 1, 0, -1):
                current_xgb_feat[lag_indices[i]] = current_xgb_feat[lag_indices[i - 1]]
            current_xgb_feat[lag_indices[0]] = current_xgb_feat[close_idx_reg] # Lag_1 becomes previous Close
            current_xgb_feat[close_idx_reg] = pred_price # Close becomes predicted price
            
        # Random Forest rolling forecast:
        current_rf_feat = latest_feat_reg_df.values[0].copy()
        rf_forecast = []
        for _ in range(future_days):
            scaled_feat = feat_scaler_reg.transform([current_rf_feat])
            pred_scaled = rf_model.predict(scaled_feat)
            pred_price = scaler_reg.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            rf_forecast.append(pred_price)
            
            # Roll lags
            for i in range(lag_days - 1, 0, -1):
                current_rf_feat[lag_indices[i]] = current_rf_feat[lag_indices[i - 1]]
            current_rf_feat[lag_indices[0]] = current_rf_feat[close_idx_reg]
            current_rf_feat[close_idx_reg] = pred_price
            
        # Draw Forecast Line Chart
        forecast_fig = go.Figure()
        
        # Past 15 days of actual prices for context
        past_context_df = df.iloc[-15:]
        forecast_fig.add_trace(go.Scatter(x=past_context_df.index, y=past_context_df['Close'], name="Historical Close", line=dict(color="#F8FAFC", width=2.5)))
        
        # Connect past close to tomorrow's prediction
        last_date = df.index[-1]
        last_price = df['Close'].iloc[-1]
        
        fc_dates = [last_date] + future_dates
        
        rf_fc_prices = np.concatenate([[last_price], rf_forecast])
        xgb_fc_prices = np.concatenate([[last_price], xgb_forecast])
        lstm_fc_prices = np.concatenate([[last_price], lstm_forecast_prices])
        
        forecast_fig.add_trace(go.Scatter(x=fc_dates, y=rf_fc_prices, name="Random Forest Forecast", line=dict(color="#F59E0B", width=2, dash='dash')))
        forecast_fig.add_trace(go.Scatter(x=fc_dates, y=xgb_fc_prices, name="XGBoost Forecast", line=dict(color="#10B981", width=2, dash='dash')))
        forecast_fig.add_trace(go.Scatter(x=fc_dates, y=lstm_fc_prices, name="PyTorch LSTM Forecast", line=dict(color="#38BDF8", width=2)))
        
        forecast_fig.update_layout(
            height=450,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title=f"Price ({info['currency']})",
            margin=dict(t=30, b=80, l=50, r=50),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(forecast_fig, config={'displayModeBar': True, 'scrollZoom': True})

    # ==================== TAB 3: MODEL COMPARISON ====================
    with tab3:
        st.markdown("### Model Performance Analysis")
        
        # Calculate performance metrics using actual values in the test set
        # RF and XGBoost test sets align with y_true_reg
        # We need to extract the previous day's close for the directional accuracy calculation
        # Let's align y_today for both regression and LSTM
        
        # Regression y_today is the actual Close on the day of prediction
        y_today_reg = df_clean_reg['Close'].iloc[split_idx_reg:].values
        metrics_rf = calculate_metrics(y_true_reg, y_pred_rf, y_today_reg)
        metrics_xgb = calculate_metrics(y_true_reg, y_pred_xgb, y_today_reg)
        
        # LSTM y_today
        y_today_lstm = df_clean_lstm['Close'].iloc[sequence_length + split_idx_lstm:].values
        metrics_lstm = calculate_metrics(y_true_lstm, y_pred_lstm, y_today_lstm)
        
        # Create metrics comparison table
        metrics_df = pd.DataFrame({
            'Random Forest': [metrics_rf['RMSE'], metrics_rf['MAE'], metrics_rf['R2'], metrics_rf['Directional_Accuracy']],
            'XGBoost': [metrics_xgb['RMSE'], metrics_xgb['MAE'], metrics_xgb['R2'], metrics_xgb['Directional_Accuracy']],
            'PyTorch LSTM': [metrics_lstm['RMSE'], metrics_lstm['MAE'], metrics_lstm['R2'], metrics_lstm['Directional_Accuracy']]
        }, index=['RMSE (Lower is Better)', 'MAE (Lower is Better)', 'R2 Score (Higher is Better)', 'Directional Accuracy (% Higher is Better)'])
        
        st.dataframe(metrics_df.T.style.highlight_max(axis=0, subset=['R2 Score (Higher is Better)', 'Directional Accuracy (% Higher is Better)'], color='#065F46').highlight_min(axis=0, subset=['RMSE (Lower is Better)', 'MAE (Lower is Better)'], color='#065F46'), width='stretch')
        
        # Metric comparison chart
        st.markdown("#### Model Directional Accuracy Comparison")
        acc_fig = go.Figure(data=[
            go.Bar(
                x=['Random Forest', 'XGBoost', 'PyTorch LSTM'],
                y=[metrics_rf['Directional_Accuracy'], metrics_xgb['Directional_Accuracy'], metrics_lstm['Directional_Accuracy']],
                marker_color=['#F59E0B', '#10B981', '#38BDF8'],
                text=[f"{metrics_rf['Directional_Accuracy']:.1f}%", f"{metrics_xgb['Directional_Accuracy']:.1f}%", f"{metrics_lstm['Directional_Accuracy']:.1f}%"],
                textposition='auto',
            )
        ])
        
        acc_fig.update_layout(
            height=350,
            template="plotly_dark",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 100],
            margin=dict(t=20, b=20, l=50, r=50)
        )
        
        st.plotly_chart(acc_fig, config={'displayModeBar': True, 'scrollZoom': False})
        
        # Determine best performing model
        best_model = "PyTorch LSTM"
        best_acc = metrics_lstm['Directional_Accuracy']
        if metrics_xgb['Directional_Accuracy'] > best_acc:
            best_model = "XGBoost"
            best_acc = metrics_xgb['Directional_Accuracy']
        if metrics_rf['Directional_Accuracy'] > metrics_xgb['Directional_Accuracy'] and metrics_rf['Directional_Accuracy'] > metrics_lstm['Directional_Accuracy']:
            best_model = "Random Forest"
            best_acc = metrics_rf['Directional_Accuracy']
            
        st.info(f"💡 **Performance Insight**: **{best_model}** exhibited the highest Directional Accuracy on the test set (**{best_acc:.1f}%**). This suggests it is currently the most reliable model for predicting trend direction for **{info['longName']}** under these training parameters.")