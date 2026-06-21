import numpy as np
import pandas as pd
from typing import Dict, Any, List

def generate_trading_signals(
    df: pd.DataFrame, 
    predicted_next_price: float, 
    model_name: str
) -> Dict[str, Any]:
    """
    Generates trading signals and recommendation reasons based on indicators and model predictions.
    
    Parameters:
    df (pd.DataFrame): Stock data containing technical indicators, must include the last row.
    predicted_next_price (float): The price predicted for tomorrow.
    model_name (str): The name of the model that generated this prediction.
    
    Returns:
    Dict[str, Any]: Recommendation details containing:
                    - 'recommendation': Strong Buy, Buy, Hold, Sell, Strong Sell
                    - 'score': Numeric score from -5.0 to 5.0
                    - 'reasons': List of textual reasons for the signal
                    - 'metrics': Key indicator values used
    """
    # Get the latest row of data (today)
    latest_row = df.iloc[-1]
    current_price = latest_row['Close']
    
    # 1. Model Prediction change
    pred_pct_change = ((predicted_next_price - current_price) / current_price) * 100.0
    
    # Technical Indicators — with safe NaN fallbacks for stocks with short history
    rsi     = latest_row['RSI']     if not pd.isna(latest_row.get('RSI',     float('nan'))) else 50.0
    bb_upper = latest_row['BB_Upper'] if not pd.isna(latest_row.get('BB_Upper', float('nan'))) else current_price * 1.02
    bb_lower = latest_row['BB_Lower'] if not pd.isna(latest_row.get('BB_Lower', float('nan'))) else current_price * 0.98
    bb_mid   = latest_row['BB_Middle'] if not pd.isna(latest_row.get('BB_Middle', float('nan'))) else current_price
    sma_10  = latest_row['SMA_10']  if not pd.isna(latest_row.get('SMA_10',  float('nan'))) else current_price
    sma_50  = latest_row['SMA_50']  if not pd.isna(latest_row.get('SMA_50',  float('nan'))) else current_price
    
    # Calculate scores and reasons
    score = 0.0
    reasons = []
    
    # A. Model Prediction Score (weight: up to 2.0)
    reasons.append(f"**Model Prediction ({model_name})**: Predicts tomorrow's price will be **{predicted_next_price:.2f}** ({'+' if pred_pct_change >= 0 else ''}{pred_pct_change:.2f}% change).")
    if pred_pct_change >= 2.0:
        score += 2.0
        reasons.append("📈 Bullish: Model predicts a strong price increase (>2.0%).")
    elif pred_pct_change >= 0.5:
        score += 1.0
        reasons.append("📈 Moderate Bullish: Model predicts a moderate price increase (0.5% - 2.0%).")
    elif pred_pct_change <= -2.0:
        score -= 2.0
        reasons.append("📉 Bearish: Model predicts a strong price decline (<-2.0%).")
    elif pred_pct_change <= -0.5:
        score -= 1.0
        reasons.append("📉 Moderate Bearish: Model predicts a moderate price decline (-0.5% - -2.0%).")
    else:
        reasons.append("⚖️ Neutral: Model predicts a very minor price fluctuation.")
        
    # B. RSI Score (weight: up to 1.5)
    reasons.append(f"**RSI (Relative Strength Index)**: Currently at **{rsi:.1f}**.")
    if rsi < 30:
        score += 1.5
        reasons.append("🔥 Strong Buy Signal: RSI is under 30, indicating the stock is heavily oversold and a price rebound is likely.")
    elif rsi < 40:
        score += 0.75
        reasons.append("👍 Mild Buy Signal: RSI is under 40, leaning towards oversold territory.")
    elif rsi > 70:
        score -= 1.5
        reasons.append("⚠️ Strong Sell Signal: RSI is over 70, indicating the stock is overbought and a price correction/pullback is likely.")
    elif rsi > 60:
        score -= 0.75
        reasons.append("👎 Mild Sell Signal: RSI is over 60, leaning towards overbought territory.")
    else:
        reasons.append("⚖️ Neutral: RSI is in the neutral range (40 - 60).")
        
    # C. Bollinger Bands Score (weight: up to 1.0)
    # Calculate how close the price is to the bands
    bb_width = bb_upper - bb_lower
    pct_from_lower = ((current_price - bb_lower) / bb_width) if bb_width > 0 else 0.5
    
    reasons.append(f"**Bollinger Bands**: Price is at **{current_price:.2f}** (Upper: {bb_upper:.2f}, Lower: {bb_lower:.2f}).")
    if current_price <= bb_lower:
        score += 1.0
        reasons.append("🔥 Buy Signal: Price has crossed below the Lower Bollinger Band, suggesting it is temporarily undervalued.")
    elif current_price >= bb_upper:
        score -= 1.0
        reasons.append("⚠️ Sell Signal: Price has crossed above the Upper Bollinger Band, suggesting it is temporarily overextended.")
    elif pct_from_lower < 0.15:
        score += 0.5
        reasons.append("👍 Mild Buy Signal: Price is trading near the Lower Bollinger Band.")
    elif pct_from_lower > 0.85:
        score -= 0.5
        reasons.append("👎 Mild Sell Signal: Price is trading near the Upper Bollinger Band.")
        
    # D. Moving Average Crossover (weight: up to 1.0)
    # Golden cross vs Death cross
    reasons.append(f"**Moving Averages**: 10-day SMA is **{sma_10:.2f}** and 50-day SMA is **{sma_50:.2f}**.")
    if sma_10 > sma_50:
        score += 1.0
        reasons.append("📈 Bullish Trend: The 10-day SMA is above the 50-day SMA, indicating upward short-term momentum.")
    else:
        score -= 1.0
        reasons.append("📉 Bearish Trend: The 10-day SMA is below the 50-day SMA, indicating downward short-term momentum.")
        
    # Map score to recommendations
    if score >= 3.0:
        recommendation = "Strong Buy"
    elif score >= 1.0:
        recommendation = "Buy"
    elif score <= -3.0:
        recommendation = "Strong Sell"
    elif score <= -1.0:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
        
    metrics = {
        'current_price': float(current_price),
        'pred_change': float(pred_pct_change),
        'rsi': float(rsi),
        'sma_10': float(sma_10),
        'sma_50': float(sma_50),
        'bb_middle': float(bb_mid),
        'bb_upper': float(bb_upper),
        'bb_lower': float(bb_lower)
    }
    
    return {
        'recommendation': recommendation,
        'score': float(score),
        'reasons': reasons,
        'metrics': metrics
    }
