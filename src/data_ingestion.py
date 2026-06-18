import pandas as pd
import yfinance as yf
import requests
from typing import Tuple, Dict, Any, Optional, List


def search_tickers(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search Yahoo Finance for stocks, ETFs, or crypto matching a query string.

    Parameters:
    query     (str): Company name or partial ticker (e.g. "Apple", "TSLA", "Bitcoin")
    max_results (int): Maximum number of results to return

    Returns:
    List of dicts, each with keys: symbol, name, exchange, type_display
    """
    if not query or len(query.strip()) < 1:
        return []

    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {
        "q": query.strip(),
        "quotesCount": max_results,
        "newsCount": 0,
        "listsCount": 0,
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("quotes", []):
            symbol = item.get("symbol", "")
            name   = item.get("longname") or item.get("shortname") or symbol
            exch   = item.get("exchDisp", item.get("exchange", ""))
            qtype  = item.get("quoteType", "")
            # Map quoteType to readable label
            type_label = {
                "EQUITY":       "Stock",
                "ETF":          "ETF",
                "MUTUALFUND":   "Fund",
                "CRYPTOCURRENCY": "Crypto",
                "INDEX":        "Index",
                "FUTURE":       "Futures",
                "CURRENCY":     "FX",
            }.get(qtype, qtype)
            if symbol:
                results.append({
                    "symbol":       symbol,
                    "name":         name,
                    "exchange":     exch,
                    "type_display": type_label,
                })
        return results
    except Exception:
        return []

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches daily stock data for a given ticker and date range.
    
    Parameters:
    ticker (str): Stock symbol (e.g., AAPL, TSLA, RELIANCE.NS)
    start_date (str): Start date string (YYYY-MM-DD)
    end_date (str): End date string (YYYY-MM-DD)
    
    Returns:
    pd.DataFrame: Stock historical data with columns [Open, High, Low, Close, Adj Close, Volume]
                 indexed by Date.
    """
    ticker = ticker.strip().upper()
    try:
        # Fetch data using yfinance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            # Fallback to download in case history doesn't return anything (e.g. indexing issues)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Flatten MultiIndex columns returned by newer yfinance versions (e.g. ('Close', 'AAPL'))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        if df.empty:
            raise ValueError(f"No stock data found for ticker '{ticker}' in the range {start_date} to {end_date}.")
            
        # Clean index name and sort
        df.index.name = 'Date'
        df = df.sort_index()
        
        # Normalise column names to title-case (yfinance can return lower-case on some builds)
        df.columns = [c.title() if c.lower() != 'adj close' else 'Adj Close' for c in df.columns]

        # Keep only necessary columns if they exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' is missing from the fetched data.")
        
        # Ensure we have Adj Close (fall back to Close if absent)
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
                
        # Keep only required columns
        df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        # Fill missing values if any
        df = df.ffill().bfill()
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error fetching data for ticker '{ticker}': {str(e)}")

def get_ticker_info(ticker: str) -> Dict[str, Any]:
    """
    Fetches basic company information/metadata.
    
    Parameters:
    ticker (str): Stock symbol (e.g., AAPL)
    
    Returns:
    Dict[str, Any]: Company metadata (name, currency, business summary, etc.)
    """
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Select key metadata fields with safe fallbacks
        metadata = {
            'symbol': ticker,
            'longName': info.get('longName', ticker),
            'currency': info.get('currency', 'USD'),
            'currentPrice': info.get('currentPrice', info.get('regularMarketPrice', None)),
            'dayHigh': info.get('dayHigh', None),
            'dayLow': info.get('dayLow', None),
            'volume': info.get('volume', None),
            'marketCap': info.get('marketCap', None),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'longBusinessSummary': info.get('longBusinessSummary', 'No description available.'),
            'website': info.get('website', 'N/A'),
            'logo_url': f"https://logo.clearbit.com/{info.get('website', '').replace('https://', '').replace('http://', '').split('/')[0]}" if info.get('website') else None
        }
        return metadata
    except Exception:
        # Return bare-minimum details on error so application doesn't crash
        return {
            'symbol': ticker,
            'longName': ticker,
            'currency': 'USD',
            'currentPrice': None,
            'dayHigh': None,
            'dayLow': None,
            'volume': None,
            'marketCap': None,
            'sector': 'N/A',
            'industry': 'N/A',
            'longBusinessSummary': 'Company details could not be retrieved.',
            'website': 'N/A',
            'logo_url': None
        }
