import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_local_fallback_response(prompt: str, context: Dict[str, Any]) -> str:
    """
    Analyzes the user's prompt and generates an intelligent, structured financial analysis
    using the active stock data, indicators, and model predictions.
    """
    p_lower = prompt.lower()
    symbol = context.get('symbol', 'N/A')
    name = context.get('name', 'this asset')
    curr_price = context.get('current_price', 0.0)
    currency = context.get('currency', 'USD')
    consensus = context.get('consensus') or {}
    forecast = context.get('forecast') or {}
    metrics = context.get('metrics') or {}
    model_metrics = context.get('model_metrics') or {}

    # 1. Technical Indicators / RSI / Bollinger Bands
    if "technical" in p_lower or "rsi" in p_lower or "bollinger" in p_lower or "indicator" in p_lower:
        rsi_val = consensus.get('metrics', {}).get('rsi', 50.0)
        bb_upper = consensus.get('metrics', {}).get('bb_upper', 0.0)
        bb_lower = consensus.get('metrics', {}).get('bb_lower', 0.0)
        
        rsi_desc = "Neutral (no extreme conditions)"
        if rsi_val >= 70:
            rsi_desc = "Overbought (potential bearish reversal signal)"
        elif rsi_val <= 30:
            rsi_desc = "Oversold (potential bullish rebound signal)"
            
        bb_pos = "within the bands"
        if curr_price >= bb_upper:
            bb_pos = "above the Upper Bollinger Band (potentially overvalued)"
        elif curr_price <= bb_lower:
            bb_pos = "below the Lower Bollinger Band (potentially undervalued)"

        return f"""### 📊 Technical Indicator Analysis for {name} ({symbol})

* **Current Price**: {currency} {curr_price:,.2f}
* **Relative Strength Index (RSI)**: `{rsi_val:.2f}` — **{rsi_desc}**
* **Bollinger Bands**:
  * Upper Band: `{bb_upper:,.2f}`
  * Lower Band: `{bb_lower:,.2f}`
  * Current position: The price is currently **{bb_pos}**.

**Analysis & Interpretation**:
RSI at `{rsi_val:.1f}` suggests that buying momentum is {rsi_desc.lower()}. Bollinger Bands represent volatility thresholds; since the price is {bb_pos}, it might suggest a trend extension or imminent mean-reversion depending on volumes. Check the chart tab for interactive visual overlays of these indicators.
"""

    # 2. Recommendations / Buy / Sell / Hold / Consensus
    elif "buy" in p_lower or "sell" in p_lower or "hold" in p_lower or "recommend" in p_lower or "consensus" in p_lower:
        rec_rating = consensus.get('recommendation', 'Hold')
        score = consensus.get('score', 0.0)
        reasons = consensus.get('reasons', [])
        
        reasons_md = "\n".join([f"- {r}" for r in reasons])
        
        return f"""### 🚦 Quantitative Consensus Rating for {name} ({symbol})

* **Rating**: **{rec_rating}**
* **Quantitative Score**: `{score:+.1f} / +5.0`

**Key Supporting Factors**:
{reasons_md}

*Note: This rating is computed using a combination of the 7-day model price prediction trends and technical indicator states (RSI, Bollinger Bands, Moving Averages).*
"""

    # 3. Model Predictions / Forecast / Future / Tomorrow
    elif "predict" in p_lower or "forecast" in p_lower or "future" in p_lower or "tomorrow" in p_lower:
        pred_tomorrow = forecast.get('tomorrow', {})
        fc_7d = forecast.get('fc_7d', {})
        
        rf_tomorrow = pred_tomorrow.get('rf', 0.0)
        xgb_tomorrow = pred_tomorrow.get('xgb', 0.0)
        lstm_tomorrow = pred_tomorrow.get('lstm', 0.0)
        avg_tomorrow = (rf_tomorrow + xgb_tomorrow + lstm_tomorrow) / 3.0
        
        change_tomorrow = avg_tomorrow - curr_price
        change_pct_tomorrow = (change_tomorrow / curr_price) * 100.0 if curr_price > 0 else 0.0
        dir_str = "increase" if change_tomorrow >= 0 else "decrease"
        
        return f"""### 🔮 Model Price Predictions & 7-Day Forecast for {name} ({symbol})

* **Current Price**: {currency} {curr_price:,.2f}
* **Tomorrow's Prediction Consensus (Average)**: {currency} {avg_tomorrow:,.2f} (`{change_pct_tomorrow:+.2f}%` predicted {dir_str})
  * **Random Forest**: {currency} {rf_tomorrow:,.2f}
  * **XGBoost**: {currency} {xgb_tomorrow:,.2f}
  * **PyTorch LSTM**: {currency} {lstm_tomorrow:,.2f}

**7-Day Model Consensus Trends**:
* **Random Forest**: {'Bullish Upward' if fc_7d.get('rf', [0])[-1] > fc_7d.get('rf', [0])[0] else 'Bearish Downward'} trend.
* **XGBoost**: {'Bullish Upward' if fc_7d.get('xgb', [0])[-1] > fc_7d.get('xgb', [0])[0] else 'Bearish Downward'} trend.
* **PyTorch LSTM**: {'Bullish Upward' if fc_7d.get('lstm', [0])[-1] > fc_7d.get('lstm', [0])[0] else 'Bearish Downward'} trend.

*Please refer to the "Forecast & Consensus Recommendations" tab for the visual time-series forecast plot.*
"""

    # 4. Model Accuracy / Performance / RMSE / R2
    elif "accuracy" in p_lower or "rmse" in p_lower or "r2" in p_lower or "performance" in p_lower or "compare" in p_lower:
        best_model = "LSTM"
        return f"""### 📊 Machine Learning Model Comparison & Diagnostics

Below are the performance metrics computed on the historical test partition:

* **Random Forest**:
  * Directional Accuracy: `{model_metrics.get('rf', {}).get('Directional_Accuracy', 0.0):.1f}%`
  * RMSE: `{model_metrics.get('rf', {}).get('RMSE', 0.0):.4f}`
  * R2 Score: `{model_metrics.get('rf', {}).get('R2', 0.0):.4f}`
* **XGBoost**:
  * Directional Accuracy: `{model_metrics.get('xgb', {}).get('Directional_Accuracy', 0.0):.1f}%`
  * RMSE: `{model_metrics.get('xgb', {}).get('RMSE', 0.0):.4f}`
  * R2 Score: `{model_metrics.get('xgb', {}).get('R2', 0.0):.4f}`
* **PyTorch LSTM Regressor**:
  * Directional Accuracy: `{model_metrics.get('lstm', {}).get('Directional_Accuracy', 0.0):.1f}%`
  * RMSE: `{model_metrics.get('lstm', {}).get('RMSE', 0.0):.4f}`
  * R2 Score: `{model_metrics.get('lstm', {}).get('R2', 0.0):.4f}`

For a full breakdown of directional predictions and error comparison, please check the **"Model Comparison & Analytics"** tab.
"""

    # 5. General Info / Summary / About / Business Description
    elif "about" in p_lower or "summary" in p_lower or "description" in p_lower or "company" in p_lower or "sector" in p_lower or "industry" in p_lower:
        desc = context.get('description', 'No description available.')
        sector = context.get('sector', 'N/A')
        industry = context.get('industry', 'N/A')
        website = context.get('website', 'N/A')
        return f"""### 📖 Company Profile: {name} ({symbol})

* **Sector**: {sector}
* **Industry**: {industry}
* **Website**: {website}

**Business Description**:
{desc}
"""

    # 6. Default Fallback response (Help menu)
    else:
        return f"""### 💬 AlphaPulse AI Assistant (Local Analyzer Mode)

I am currently running in **Local Analyzer Mode** because no OpenAI API key was provided or configured. In this mode, I can analyze the active dataset, model forecasts, and technical indicators for **{name} ({symbol})**.

Here are some specific queries you can ask me:
* **"Analyze the technical indicators"** (returns RSI, Bollinger Bands, and support zones)
* **"Explain tomorrow's price predictions"** (returns RF, XGBoost, and PyTorch LSTM predictions)
* **"What is the buy/sell consensus rating?"** (returns quantitative rating and reasons)
* **"Show model accuracy comparisons"** (returns RMSE, R2, and directional accuracy metrics)
* **"Give me information about this company"** (returns sector, industry, and description)

*To unlock general conversational features and advanced financial reasoning, please enter your Groq API key in the sidebar panel.*
"""


def get_assistant_response(
    prompt: str, 
    chat_history: List[Dict[str, str]], 
    context: Dict[str, Any], 
    api_key: str = None
) -> str:
    """
    Get assistant response using LangChain Groq if API key is provided,
    otherwise fallback to the local rule-based stock query analyzer.
    """
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        return get_local_fallback_response(prompt, context)

    try:
        # Build prompt messages safely extracting nested values
        consensus_dict = context.get('consensus') or {}
        metrics_dict = consensus_dict.get('metrics') or {}
        forecast_dict = context.get('forecast') or {}
        tomorrow_dict = forecast_dict.get('tomorrow') or {}
        model_metrics_dict = context.get('model_metrics') or {}

        system_message_content = f"""You are AlphaPulse AI Assistant, a premium quantitative finance advisor.
You are assisting a user in analyzing a specific stock/cryptocurrency: {context.get('name')} ({context.get('symbol')}).

Here is the current live context of the asset as computed by AlphaPulse:
- **Asset**: {context.get('name')} ({context.get('symbol')})
- **Sector**: {context.get('sector')} | Industry: {context.get('industry')}
- **Description**: {context.get('description')}
- **Current Close Price**: {context.get('currency')} {context.get('current_price')}
- **Technical Indicators**:
  - RSI: {metrics_dict.get('rsi')}
  - Bollinger Bands: Upper={metrics_dict.get('bb_upper')}, Lower={metrics_dict.get('bb_lower')}
- **Model Predictions for Tomorrow**:
  - Random Forest Prediction: {tomorrow_dict.get('rf')}
  - XGBoost Prediction: {tomorrow_dict.get('xgb')}
  - PyTorch LSTM Prediction: {tomorrow_dict.get('lstm')}
- **Model Consensus Rating**: {consensus_dict.get('recommendation')} (Score: {consensus_dict.get('score')})
  - Supporting Reasons: {consensus_dict.get('reasons')}
- **Performance Diagnostics (Directional Accuracy / RMSE)**:
  - Random Forest: Accuracy={model_metrics_dict.get('rf', {}).get('Directional_Accuracy')}%, RMSE={model_metrics_dict.get('rf', {}).get('RMSE')}
  - XGBoost: Accuracy={model_metrics_dict.get('xgb', {}).get('Directional_Accuracy')}%, RMSE={model_metrics_dict.get('xgb', {}).get('RMSE')}
  - PyTorch LSTM: Accuracy={model_metrics_dict.get('lstm', {}).get('Directional_Accuracy')}%, RMSE={model_metrics_dict.get('lstm', {}).get('RMSE')}

Provide detailed, technical, professional, and visually structured markdown responses. Be precise, cite the figures in the context, and maintain a high-quality tone. Do not make up any data not provided in the context unless you clearly state it is a general knowledge elaboration.
"""
        
        messages = [SystemMessage(content=system_message_content)]
        
        # Add chat history
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        # Append current user prompt
        messages.append(HumanMessage(content=prompt))
        
        # Call LangChain ChatGroq
        chat = ChatGroq(
            api_key=api_key, 
            model="llama-3.3-70b-versatile", 
            temperature=0.2
        )
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"⚠️ **Error connecting to Groq via LangChain**: {str(e)}\n\n*Falling back to Local Analyzer Mode results:*\n\n{get_local_fallback_response(prompt, context)}"
