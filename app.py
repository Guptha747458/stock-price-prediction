import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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
from src.assistant import get_assistant_response

@st.dialog("💬 AlphaPulse AI Assistant", width="large")
def chat_dialog(ticker, info, chat_context):
    st.markdown("### 💬 AlphaPulse AI Assistant")
    
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    # Display indicator of status
    if not groq_api_key:
        st.caption("⚠️ **Local Analyzer Mode**: Set `GROQ_API_KEY` in `.env` for full AI assistant features.")
    else:
        st.caption("🟢 **Groq Active** (Connected via LangChain)")
        
    # Initialize or reset chat history when switching stock tickers
    if "chat_messages" not in st.session_state or st.session_state.get("current_chat_ticker") != ticker:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": f"Hello! I am your AlphaPulse AI Assistant. Ask me anything about **{info['longName']} ({ticker})** technical indicators, model predictions, or company performance!"}
        ]
        st.session_state["current_chat_ticker"] = ticker
    # Clear history button
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": f"Hello! I am your AlphaPulse AI Assistant. Ask me anything about **{info['longName']} ({ticker})** technical indicators, model predictions, or company performance!"}
        ]
        st.rerun()

    # Display chat messages from history on app rerun
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_query = st.chat_input("Ask AlphaPulse AI...")
        
    if user_query:
        # Display user message in chat message container
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing stock patterns & generating response..."):
                resp = get_assistant_response(
                    prompt=user_query,
                    chat_history=st.session_state["chat_messages"][:-1],
                    context=chat_context,
                    api_key=groq_api_key
                )
                st.markdown(resp)
                st.session_state["chat_messages"].append({"role": "assistant", "content": resp})



def select_search_ticker(chosen_label):
    chosen_symbol = chosen_label.split("  —  ")[0].strip()
    st.session_state["custom_ticker"]       = chosen_symbol
    st.session_state["custom_ticker_input"] = chosen_symbol
    st.session_state["market_category"]     = "🏗️ Custom (Any Exchange)"
    st.session_state["has_run"]             = False
    st.session_state["last_seen_ticker"]    = chosen_symbol.upper()
    if "search_results" in st.session_state:
        del st.session_state["search_results"]



# --- Page Configuration ---
st.set_page_config(
    page_title="AlphaPulse - Stock Price Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fixed Theme Configuration (Bloomberg Terminal) ---
bg_color = "#0D0D0D"
sidebar_bg = "#111111"
card_bg = "#141414"
card_hover_bg = "#181818"
accent_color = "#FF6600"
text_color = "#C8C8C8"
heading_color = "#E8E8E8"
border_color = "#2A2A2A"
input_bg = "#141414"
input_border = "#2A2A2A"
input_text = "#E8E8E8"
tab_text_unselected = "#666666"
df_bg = "#141414"
df_header_bg = "#1A1A1A"
chat_bg = "#141414"
plotly_template = "plotly_dark"
plotly_paper_bg = "#0D0D0D"
plotly_plot_bg = "#141414"
plotly_grid_color = "#2A2A2A"


# --- Professional Finance Styling (Bloomberg Terminal) ---
_CSS = f"""
<style>
/* --- AlphaPulse — Bloomberg Terminal Design --- */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {{
    --bg-color: {bg_color};
    --sidebar-bg: {sidebar_bg};
    --card-bg: {card_bg};
    --card-hover-bg: {card_hover_bg};
    --accent-color: {accent_color};
    --text-color: {text_color};
    --heading-color: {heading_color};
    --border-color: {border_color};
    --input-bg: {input_bg};
    --input-border: {input_border};
    --input-text: {input_text};
    --tab-text-unselected: {tab_text_unselected};
    --df-bg: {df_bg};
    --df-header-bg: {df_header_bg};
    --chat-bg: {chat_bg};
}}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {{
    background: var(--bg-color);
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-color);
}}

/* Top status bar accent — the signature element */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color);
    z-index: 9999;
    pointer-events: none;
}}

/* ── Typography ── */
h1, h2, h3, h4 {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: var(--heading-color);
    text-transform: uppercase;
}}

.main-title {{
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent-color);
    font-size: 2.1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.1rem;
    text-transform: uppercase;
}}

.subtitle {{
    color: var(--tab-text-unselected);
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    border-left: 2px solid var(--accent-color);
    padding-left: 0.6rem;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border-color);
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1rem;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--tab-text-unselected);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ── Metric cards — sharp, borderline, no rounding ── */
.metric-card {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-top: 2px solid var(--accent-color);
    border-radius: 0;
    padding: 0.9rem 1rem;
    transition: border-color 0.15s ease, background 0.15s ease;
}}
.metric-card:hover {{
    background: var(--card-hover-bg);
    border-color: var(--accent-color);
    border-top-color: var(--accent-color);
}}

.metric-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--tab-text-unselected);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}}
.metric-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--heading-color);
    line-height: 1.1;
}}

/* ── Recommendation badges — flat, terminal style ── */
.rec-badge {{
    display: inline-block;
    padding: 0.3rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 0;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border: none;
}}
.rec-strong-buy  {{ background: #00AA44; color: #0D0D0D; }}
.rec-buy         {{ background: #007733; color: #CCFFDD; }}
.rec-hold        {{ background: #333333; color: #AAAAAA; }}
.rec-sell        {{ background: #CC2200; color: #FFE0DD; }}
.rec-strong-sell {{ background: #990000; color: #FFE0DD; }}

/* ── Status ── */
.status-ok {{
    color: #00CC55;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
}}

/* ── Buttons ── */
.stButton > button {{
    background: var(--card-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 0;
    padding: 0.45rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    transition: all 0.15s ease;
}}
.stButton > button:hover {{
    background: var(--accent-color);
    color: #0D0D0D;
    border-color: var(--accent-color);
}}

/* ── Inputs & selects ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div {{
    background: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 0 !important;
    color: var(--input-text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: var(--accent-color) !important;
    box-shadow: none !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tab-text-unselected);
    border-radius: 0;
    padding: 0.5rem 1rem;
    border-bottom: 2px solid transparent;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: var(--accent-color);
    border-bottom: 2px solid var(--accent-color);
    background: transparent;
}}

/* ── DataFrames / tables ── */
[data-testid="stDataFrame"] {{
    border-radius: 0;
    border: 1px solid var(--border-color);
}}
[data-testid="stDataFrame"] th {{
    background: var(--df-header-bg) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--tab-text-unselected) !important;
}}
[data-testid="stDataFrame"] td {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--text-color) !important;
}}

/* ── Plotly chart containers ── */
.js-plotly-plot {{
    border: 1px solid var(--border-color);
    border-radius: 0;
}}

/* ── Info / alert boxes ── */
[data-testid="stAlert"] {{
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-left: 3px solid var(--accent-color) !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-color) !important;
}}

/* ── Dividers ── */
hr {{
    border: none;
    border-top: 1px solid var(--border-color);
    margin: 1rem 0;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-color); }}
::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 0; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent-color); }}

/* ── Streamlit metric widgets ── */
[data-testid="metric-container"] {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-top: 2px solid var(--border-color);
    padding: 0.75rem 1rem;
    border-radius: 0;
}}
[data-testid="metric-container"] label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--tab-text-unselected) !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important;
    color: var(--heading-color) !important;
}}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{
    background: var(--chat-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Sans', sans-serif;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] p {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--tab-text-unselected);
}}

/* ── Welcome Container & Feature Cards ── */
.welcome-container {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 2.5rem;
    margin-top: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
}}
.feature-card {{
    background: var(--card-hover-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.2rem;
}}
.feature-card h4 {{
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-size: 1.05rem;
    color: var(--accent-color);
}}
.feature-card p {{
    color: var(--text-color);
    font-size: 0.92rem;
    line-height: 1.4;
    margin: 0;
}}
.start-instructions {{
    background: var(--sidebar-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}}
.start-instructions p {{
    margin: 0;
}}
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

chat_button_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")

# ── Stock Selection Mode ──────────────────────────────────────────────────────
st.sidebar.markdown("### 🌍 Stock Selection")

market_category = st.sidebar.selectbox(
    "Market / Category:",
    list(GLOBAL_STOCKS.keys()),
    key="market_category"
)

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
    
    # Save the curated selection to pre-populate Custom mode when switched
    st.session_state["custom_ticker"] = ticker
    st.session_state["custom_ticker_input"] = ticker

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
            st.button(
                "✅ Use this ticker", 
                key="use_search_ticker", 
                use_container_width=True, 
                on_click=select_search_ticker, 
                args=(chosen_label,)
            )
        elif do_search:
            st.warning("No results found. Try a different name or check spelling.")

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

# Fix #14: Validate date range before allowing the user to proceed
if start_date >= end_date:
    st.sidebar.error("⚠️ Start date must be before the end date.")

# Model configuration options
st.sidebar.markdown("### 🤖 Model Settings")
sequence_length = st.sidebar.slider("LSTM Sequence Length (Days):", min_value=5, max_value=30, value=10)
lstm_epochs = st.sidebar.slider("LSTM Training Epochs:", min_value=5, max_value=30, value=15)
train_split = st.sidebar.slider("Train/Test Split Ratio:", min_value=0.6, max_value=0.9, value=0.8, step=0.05)

st.sidebar.markdown("---")
run_button = st.sidebar.button("⚡ Fetch & Train Models", use_container_width=True, disabled=(start_date >= end_date))



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

# Fix #5: Removed auto-trigger on ticker change — it wasted compute when users
# were still browsing the dropdown. Only the explicit button now starts training.
if "last_seen_ticker" not in st.session_state:
    st.session_state["last_seen_ticker"] = ticker

# If user clicks the run button, trigger execution
if run_button:
    st.session_state["has_run"] = True
    st.session_state["last_seen_ticker"] = ticker

has_run = st.session_state.get("has_run", False)
open_chat_clicked = chat_button_placeholder.button(
    "💬 AlphaPulse AI Assistant", 
    use_container_width=True, 
    disabled=not has_run,
    help="You must fetch and train models first." if not has_run else "Open the AI Chatbot"
)

# Render Welcome Landing Page if not run yet
if not st.session_state["has_run"]:
    st.markdown(
'<div class="welcome-container">'
'<h2 style="color: var(--accent-color); font-weight: 800; font-size: 2.2rem; margin-bottom: 0.5rem; font-family: \'Outfit\', sans-serif;">Welcome to AlphaPulse! 📈</h2>'
'<p style="color: var(--text-color); font-size: 1.15rem; margin-bottom: 2rem; opacity: 0.85;">An advanced, AI-powered stock prediction &amp; quantitative analysis platform utilizing machine learning models to forecast price trends.</p>'
'<h3 style="color: var(--heading-color); font-size: 1.3rem; margin-bottom: 1rem; font-family: \'Outfit\', sans-serif;">🎯 Core Features &amp; Capabilities</h3>'
'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2.5rem;">'
'<div class="feature-card">'
'<h4>🧠 Machine Learning Predictions</h4>'
'<p>Trains three separate models in real-time on historical prices: <b>Random Forest</b>, <b>XGBoost</b>, and a deep learning <b>PyTorch LSTM Network</b>.</p>'
'</div>'
'<div class="feature-card">'
'<h4>📊 Technical Charts &amp; Overlays</h4>'
'<p>Generates interactive candlestick charts with overlays for SMAs, EMAs, Bollinger Bands, and RSI indicators.</p>'
'</div>'
'<div class="feature-card">'
'<h4>🔮 Autoregressive Forecasts</h4>'
'<p>Rolls predictions recursively to project a <b>7-day price forecast</b> showing trends and model consensus.</p>'
'</div>'
'<div class="feature-card">'
'<h4>🚦 Consensus Trading Signals</h4>'
'<p>Combines models and technical indicator states to compute quantitative buy/hold/sell trading signals.</p>'
'</div>'
'</div>'
'<div class="start-instructions">'
'<p style="color: var(--heading-color); font-weight: 600; font-size: 1.1rem; margin-bottom: 0.3rem;">👈 How to start:</p>'
'<p style="color: var(--text-color); font-size: 0.95rem; margin-bottom: 0; opacity: 0.85;">Select a stock from the left sidebar or search by company name, then click <b>⚡ Fetch &amp; Train Models</b> to load the interactive dashboard.</p>'
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
    st.markdown(f"## {info['longName']} ({info['symbol']})")
    st.markdown(f"**Sector**: {info['sector']} | **Industry**: {info['industry']} | **Currency**: {info['currency']}")
            
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
        # Format volume in abbreviated form to prevent overflow in metric card
        if volume >= 1e9:
            volume_str = f"{volume/1e9:.2f}B"
        elif volume >= 1e6:
            volume_str = f"{volume/1e6:.2f}M"
        elif volume >= 1e3:
            volume_str = f"{volume/1e3:.1f}K"
        else:
            volume_str = f"{volume:,.0f}"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volume</div>
            <div class="metric-value">{volume_str}</div>
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
    # Fix #2: Pass df as a frozen bytes hash-key so @st.cache_data can detect changes
    # without holding a mutable reference to the live dataframe.
    @st.cache_data(show_spinner=False)
    def prepare_all_data(df_processed, seq_len, split_ratio):
        """Cache-friendly: returns serialisable data (numpy arrays + scalers)."""
        lag_days = 5
        reg_data = prepare_regression_data(df_processed, lag_days=lag_days, train_split=split_ratio)
        lstm_data = prepare_lstm_data(df_processed, sequence_length=seq_len, train_split=split_ratio)
        return reg_data, lstm_data, lag_days

    # Fix #3: Removed session_state access from inside @st.cache_resource.
    # The df is now passed directly as a parameter so the function is pure.
    # Fix #13: LSTM uses hidden_size=32 / num_layers=1 intentionally — lighter
    # model that trains faster in the browser; dropout is auto-disabled for single layer.
    @st.cache_resource(show_spinner=False)
    def train_all_models(df_hash_key, seq_len, epochs_num, split_ratio, df_processed):
        """Cache-friendly: returns non-serialisable model objects.
        df_hash_key is a tuple (ticker, start, end, ...) so Streamlit invalidates on param change.
        df_processed is passed directly — never read from session_state here.
        """
        reg_data, lstm_data, _lag_days = prepare_all_data(df_processed, seq_len, split_ratio)
        X_tr_reg, y_tr_reg, X_te_reg, y_te_reg, feat_scaler_reg, scaler_reg, feat_cols_reg = reg_data
        X_tr_lstm, y_tr_lstm, X_te_lstm, y_te_lstm, feat_scaler_lstm, scaler_lstm, feat_cols_lstm = lstm_data
        
        rf_model = train_random_forest(X_tr_reg, y_tr_reg)
        xgb_model = train_xgboost(X_tr_reg, y_tr_reg)
        
        lstm_model = PyTorchLSTMRegressor(
            input_size=len(feat_cols_lstm),
            hidden_size=32,   # Intentionally lighter than default (64) for faster in-browser training
            num_layers=1,     # Single layer — dropout is auto-disabled by LSTMNetwork when num_layers=1
            epochs=epochs_num,
            batch_size=32
        )
        lstm_model.fit(X_tr_lstm, y_tr_lstm, val_data=(X_te_lstm, y_te_lstm))
        
        return {'rf': rf_model, 'xgb': xgb_model, 'lstm': lstm_model}

    # Derive hash key from current parameters to bust model cache on param change
    _cache_key = (ticker, str(start_date), str(end_date), sequence_length, lstm_epochs, train_split)

    # Detect whether this is a genuinely new training run (params changed or button pressed)
    _needs_training = (
        run_button or
        st.session_state.get('_last_trained_key') != _cache_key
    )

    if run_button:
        # Fix #4: Clear only the specific cache functions instead of the global cache
        # to avoid invalidating other users' trained models in multi-user deployments.
        prepare_all_data.clear()
        train_all_models.clear()
        load_data.clear()

    if _needs_training:
        # Show spinner only when actually training (first load or param change)
        with st.spinner("🧠 Training Random Forest, XGBoost, and PyTorch LSTM models on stock history..."):
            reg_data, lstm_data, lag_days = prepare_all_data(df, sequence_length, train_split)
            models_dict = train_all_models(_cache_key, sequence_length, lstm_epochs, train_split, df)
        st.session_state['_last_trained_key'] = _cache_key
        st.toast("Models trained successfully!", icon="✅")
    else:
        # Tab switch / sidebar interaction — return instantly from cache, no spinner
        reg_data, lstm_data, lag_days = prepare_all_data(df, sequence_length, train_split)
        models_dict = train_all_models(_cache_key, sequence_length, lstm_epochs, train_split, df)

    # Unpack model data
    rf_model = models_dict['rf']
    xgb_model = models_dict['xgb']
    lstm_model = models_dict['lstm']

    # Unpack data — use .copy() to prevent inadvertent mutation of cached arrays
    X_tr_reg, y_tr_reg, X_te_reg, y_te_reg, feat_scaler_reg, scaler_reg, feat_cols_reg = reg_data
    X_tr_reg, y_tr_reg, X_te_reg, y_te_reg = X_tr_reg.copy(), y_tr_reg.copy(), X_te_reg.copy(), y_te_reg.copy()

    X_tr_lstm, y_tr_lstm, X_te_lstm, y_te_lstm, feat_scaler_lstm, scaler_lstm, feat_cols_lstm = lstm_data
    X_tr_lstm, y_tr_lstm, X_te_lstm, y_te_lstm = X_tr_lstm.copy(), y_tr_lstm.copy(), X_te_lstm.copy(), y_te_lstm.copy()

    # --- Generate Predictions (cached in session_state to avoid re-running on every tab switch / chat send) ---
    if st.session_state.get('_pred_cache_key') != _cache_key:
        # Align clean datasets and split indices
        df_lags_clean = create_lagged_features(df, lag_days=lag_days)
        df_clean_reg = df_lags_clean.dropna().iloc[:-1]
        split_idx_reg = int(len(df_clean_reg) * train_split)

        df_clean_lstm = df.dropna().iloc[:-1]
        split_idx_lstm = int((len(df_clean_lstm) - sequence_length) * train_split)

        # 1. XGBoost & Random Forest test set predictions
        y_pred_rf_scaled = rf_model.predict(X_te_reg)
        y_pred_xgb_scaled = xgb_model.predict(X_te_reg)

        y_pred_rf   = scaler_reg.inverse_transform(y_pred_rf_scaled.reshape(-1, 1)).ravel()
        y_pred_xgb  = scaler_reg.inverse_transform(y_pred_xgb_scaled.reshape(-1, 1)).ravel()
        y_true_reg  = scaler_reg.inverse_transform(y_te_reg.reshape(-1, 1)).ravel()

        # 2. LSTM test set predictions
        y_pred_lstm_scaled = lstm_model.predict(X_te_lstm)
        y_pred_lstm  = scaler_lstm.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).ravel()
        y_true_lstm  = scaler_lstm.inverse_transform(y_te_lstm.reshape(-1, 1)).ravel()

        # 3. Tomorrow's predictions (1-day ahead)
        # A. Regression tomorrow
        df_lags = create_lagged_features(df, lag_days=lag_days)
        latest_feat_reg_df    = df_lags[feat_cols_reg].dropna().iloc[-1:]
        latest_feat_reg_scaled = feat_scaler_reg.transform(latest_feat_reg_df.values)

        pred_tomorrow_rf  = scaler_reg.inverse_transform(rf_model.predict(latest_feat_reg_scaled).reshape(-1, 1))[0][0]
        pred_tomorrow_xgb = scaler_reg.inverse_transform(xgb_model.predict(latest_feat_reg_scaled).reshape(-1, 1))[0][0]

        # B. LSTM tomorrow
        df_no_nan = df.dropna()
        latest_sequence_lstm        = df_no_nan[feat_cols_lstm].iloc[-sequence_length:].values
        latest_sequence_lstm_scaled = feat_scaler_lstm.transform(latest_sequence_lstm)
        pred_tomorrow_lstm = scaler_lstm.inverse_transform(
            lstm_model.predict(np.expand_dims(latest_sequence_lstm_scaled, axis=0)).reshape(-1, 1)
        )[0][0]

        # 4. Rolling 7-day forecast for all three models
        future_days  = 7
        future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(1, future_days + 1)]

        # LSTM rolling forecast
        close_idx         = feat_cols_lstm.index('Close')
        lstm_rolling_seq  = latest_sequence_lstm_scaled.copy()
        lstm_forecast_raw = []
        for _ in range(future_days):
            pred_s = lstm_model.predict(np.expand_dims(lstm_rolling_seq, axis=0))
            lstm_forecast_raw.append(pred_s[0])
            new_row          = lstm_rolling_seq[-1].copy()
            new_row[close_idx] = pred_s[0]
            lstm_rolling_seq = np.vstack([lstm_rolling_seq[1:], new_row])
        lstm_forecast_prices = scaler_lstm.inverse_transform(
            np.array(lstm_forecast_raw).reshape(-1, 1)
        ).ravel()

        # XGBoost rolling forecast
        lag_indices       = [feat_cols_reg.index(f'Close_Lag_{i}') for i in range(1, lag_days + 1)]
        close_idx_reg     = feat_cols_reg.index('Close')
        current_xgb_feat  = latest_feat_reg_df.values[0].copy()
        xgb_forecast      = []
        for _ in range(future_days):
            pred_price = scaler_reg.inverse_transform(
                xgb_model.predict(feat_scaler_reg.transform([current_xgb_feat])).reshape(-1, 1)
            )[0][0]
            xgb_forecast.append(pred_price)
            for i in range(lag_days - 1, 0, -1):
                current_xgb_feat[lag_indices[i]] = current_xgb_feat[lag_indices[i - 1]]
            current_xgb_feat[lag_indices[0]]  = current_xgb_feat[close_idx_reg]
            current_xgb_feat[close_idx_reg]   = pred_price

        # Random Forest rolling forecast
        current_rf_feat = latest_feat_reg_df.values[0].copy()
        rf_forecast     = []
        for _ in range(future_days):
            pred_price = scaler_reg.inverse_transform(
                rf_model.predict(feat_scaler_reg.transform([current_rf_feat])).reshape(-1, 1)
            )[0][0]
            rf_forecast.append(pred_price)
            for i in range(lag_days - 1, 0, -1):
                current_rf_feat[lag_indices[i]] = current_rf_feat[lag_indices[i - 1]]
            current_rf_feat[lag_indices[0]]  = current_rf_feat[close_idx_reg]
            current_rf_feat[close_idx_reg]   = pred_price

        # Store everything in session_state so future reruns (tab switches, chat) skip all this
        st.session_state['_pred_cache_key'] = _cache_key
        st.session_state['_pred_results'] = {
            'df_clean_reg':          df_clean_reg,
            'df_clean_lstm':         df_clean_lstm,
            'split_idx_reg':         split_idx_reg,
            'split_idx_lstm':        split_idx_lstm,
            'y_pred_rf':             y_pred_rf,
            'y_pred_xgb':            y_pred_xgb,
            'y_pred_lstm':           y_pred_lstm,
            'y_true_reg':            y_true_reg,
            'y_true_lstm':           y_true_lstm,
            'pred_tomorrow_rf':      pred_tomorrow_rf,
            'pred_tomorrow_xgb':     pred_tomorrow_xgb,
            'pred_tomorrow_lstm':    pred_tomorrow_lstm,
            'future_dates':          future_dates,
            'rf_forecast':           rf_forecast,
            'xgb_forecast':          xgb_forecast,
            'lstm_forecast_prices':  lstm_forecast_prices,
            'feat_cols_reg':         feat_cols_reg,
            'feat_cols_lstm':        feat_cols_lstm,
            'latest_feat_reg_df':    latest_feat_reg_df,
            'latest_sequence_lstm_scaled': latest_sequence_lstm_scaled,
            'lag_days':              lag_days,
        }
    else:
        # Restore instantly from session_state — zero computation on reruns
        _pr = st.session_state['_pred_results']
        df_clean_reg          = _pr['df_clean_reg']
        df_clean_lstm         = _pr['df_clean_lstm']
        split_idx_reg         = _pr['split_idx_reg']
        split_idx_lstm        = _pr['split_idx_lstm']
        y_pred_rf             = _pr['y_pred_rf']
        y_pred_xgb            = _pr['y_pred_xgb']
        y_pred_lstm           = _pr['y_pred_lstm']
        y_true_reg            = _pr['y_true_reg']
        y_true_lstm           = _pr['y_true_lstm']
        pred_tomorrow_rf      = _pr['pred_tomorrow_rf']
        pred_tomorrow_xgb     = _pr['pred_tomorrow_xgb']
        pred_tomorrow_lstm    = _pr['pred_tomorrow_lstm']
        future_dates          = _pr['future_dates']
        rf_forecast           = _pr['rf_forecast']
        xgb_forecast          = _pr['xgb_forecast']
        lstm_forecast_prices  = _pr['lstm_forecast_prices']
        feat_cols_reg         = _pr['feat_cols_reg']
        feat_cols_lstm        = _pr['feat_cols_lstm']
        latest_feat_reg_df    = _pr['latest_feat_reg_df']
        latest_sequence_lstm_scaled = _pr['latest_sequence_lstm_scaled']
        lag_days              = _pr['lag_days']

    # --- App Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "📈 Technical Charts & Indicators", 
        "🔮 Forecast & Recommendations", 
        "📊 Model Comparison"
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
            template=plotly_template,
            paper_bgcolor=plotly_paper_bg,
            plot_bgcolor=plotly_plot_bg,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.08,
                xanchor="center",
                x=0.5,
                font=dict(size=11, color=text_color),
                bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(t=30, b=80, l=50, r=50),
            xaxis=dict(gridcolor=plotly_grid_color, linecolor=plotly_grid_color),
            yaxis=dict(gridcolor=plotly_grid_color, linecolor=plotly_grid_color),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor=plotly_grid_color)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[10, 90], gridcolor=plotly_grid_color)
        fig.update_yaxes(title_text="Volume", row=3, col=1, gridcolor=plotly_grid_color)
        
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
        # Guard against off-by-one: ensure dates length matches prediction array length
        _lstm_raw_dates = df_clean_lstm.index[sequence_length + split_idx_lstm:]
        test_dates_lstm = _lstm_raw_dates[:len(y_pred_lstm)]
        
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
            template=plotly_template,
            paper_bgcolor=plotly_paper_bg,
            plot_bgcolor=plotly_plot_bg,
            xaxis_title="Date",
            yaxis_title=f"Close Price ({info['currency']})",
            margin=dict(t=30, b=80, l=50, r=50),
            legend=dict(
                orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                font=dict(color=text_color), bgcolor="rgba(0,0,0,0)"
            ),
            xaxis=dict(gridcolor=plotly_grid_color),
            yaxis=dict(gridcolor=plotly_grid_color),
        )
        
        st.plotly_chart(pred_fig, config={'displayModeBar': True, 'scrollZoom': True})
        
        st.markdown("---")
        st.markdown("### Next 7 Days Rolling Price Forecast")
        
        # Fix #12: future_days already defined at line 808 inside the prediction block;
        # redundant re-assignment removed — use the value from session state cache instead.
        future_days = len(future_dates)
            
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
            template=plotly_template,
            paper_bgcolor=plotly_paper_bg,
            plot_bgcolor=plotly_plot_bg,
            xaxis_title="Date",
            yaxis_title=f"Price ({info['currency']})",
            margin=dict(t=30, b=80, l=50, r=50),
            legend=dict(
                orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                font=dict(color=text_color), bgcolor="rgba(0,0,0,0)"
            ),
            xaxis=dict(gridcolor=plotly_grid_color),
            yaxis=dict(gridcolor=plotly_grid_color),
        )
        
        st.plotly_chart(forecast_fig, config={'displayModeBar': True, 'scrollZoom': True})

    # ==================== TAB 3: MODEL COMPARISON ====================
    # --- Compute model metrics + chat_context (cached to avoid re-running on tab switches) ---
    if st.session_state.get('_metrics_cache_key') != _cache_key:
        y_today_reg = df_clean_reg['Close'].iloc[split_idx_reg:].values
        metrics_rf  = calculate_metrics(y_true_reg, y_pred_rf,  y_today_reg)
        metrics_xgb = calculate_metrics(y_true_reg, y_pred_xgb, y_today_reg)

        # LSTM y_today — guard length to match prediction array
        _lstm_y_today_raw = df_clean_lstm['Close'].iloc[sequence_length + split_idx_lstm:].values
        y_today_lstm  = _lstm_y_today_raw[:len(y_pred_lstm)]
        metrics_lstm  = calculate_metrics(y_true_lstm, y_pred_lstm, y_today_lstm)

        chat_context = {
            'symbol': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'description': info.get('longBusinessSummary', 'No description available.'),
            'current_price': current_price,
            'currency': info.get('currency', 'USD'),
            'website': info.get('website', 'N/A'),
            'consensus': {
                'recommendation': rec,
                'score': score,
                'reasons': reasons,
                'metrics': {
                    'rsi': recommendation_data['metrics']['rsi'],
                    'bb_upper': recommendation_data['metrics']['bb_upper'],
                    'bb_lower': recommendation_data['metrics']['bb_lower']
                }
            },
            'forecast': {
                'tomorrow': {
                    'rf': pred_tomorrow_rf,
                    'xgb': pred_tomorrow_xgb,
                    'lstm': pred_tomorrow_lstm
                },
                'fc_7d': {
                    'rf':   rf_forecast.tolist()           if hasattr(rf_forecast,           'tolist') else list(rf_forecast),
                    'xgb':  xgb_forecast.tolist()          if hasattr(xgb_forecast,          'tolist') else list(xgb_forecast),
                    'lstm': lstm_forecast_prices.tolist()  if hasattr(lstm_forecast_prices,  'tolist') else list(lstm_forecast_prices)
                }
            },
            'model_metrics': {'rf': metrics_rf, 'xgb': metrics_xgb, 'lstm': metrics_lstm}
        }

        st.session_state['_metrics_cache_key'] = _cache_key
        st.session_state['_metrics_results']   = {
            'metrics_rf':   metrics_rf,
            'metrics_xgb':  metrics_xgb,
            'metrics_lstm': metrics_lstm,
            'chat_context': chat_context,
        }
    else:
        _mr         = st.session_state['_metrics_results']
        metrics_rf  = _mr['metrics_rf']
        metrics_xgb = _mr['metrics_xgb']
        metrics_lstm = _mr['metrics_lstm']
        chat_context = _mr['chat_context']

    with tab3:
        st.markdown("### Model Performance Analysis")
        
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
            template=plotly_template,
            paper_bgcolor=plotly_paper_bg,
            plot_bgcolor=plotly_plot_bg,
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 100],
            margin=dict(t=20, b=20, l=50, r=50),
            xaxis=dict(gridcolor=plotly_grid_color),
            yaxis=dict(gridcolor=plotly_grid_color),
        )
        
        st.plotly_chart(acc_fig, config={'displayModeBar': True, 'scrollZoom': False})
        
        # Fix #10: Use max() for a correct, unambiguous 3-way comparison
        _model_accs = {
            'Random Forest': metrics_rf['Directional_Accuracy'],
            'XGBoost':       metrics_xgb['Directional_Accuracy'],
            'PyTorch LSTM':  metrics_lstm['Directional_Accuracy'],
        }
        best_model = max(_model_accs, key=_model_accs.get)
        best_acc   = _model_accs[best_model]
            
        st.info(f"💡 **Performance Insight**: **{best_model}** exhibited the highest Directional Accuracy on the test set (**{best_acc:.1f}%**). This suggests it is currently the most reliable model for predicting trend direction for **{info['longName']}** under these training parameters.")

    # ==================== CHATBOT DIALOG ====================
    if open_chat_clicked:
        chat_dialog(ticker, info, chat_context)