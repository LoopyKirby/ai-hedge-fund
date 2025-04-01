from graph.state import AgentState
from tools.binance_data import binance
import logging

logger = logging.getLogger(__name__)

def analyze_fundamentals(volume: float, price_change: float, weighted_avg_price: float) -> dict:
    """分析基本面数据"""
    signal = "neutral"
    confidence = 50.0
    reasons = []
    
    # 分析交易量
    if volume > 1000000:  # 大交易量
        confidence += 10
        reasons.append(f"高交易量: {volume:,.0f}")
    
    # 分析价格变化
    if price_change > 5:
        signal = "bullish"
        confidence += 15
        reasons.append(f"显著的价格上涨: {price_change:.2f}%")
    elif price_change < -5:
        signal = "bearish"
        confidence += 15
        reasons.append(f"显著的价格下跌: {price_change:.2f}%")
    
    # 分析加权平均价格
    if weighted_avg_price > 0:
        reasons.append(f"加权平均价格: {weighted_avg_price:.2f}")
    
    return {
        "signal": signal,
        "confidence": min(confidence, 100),
        "reasoning": " | ".join(reasons)
    }

def fundamentals_agent(state: AgentState):
    """基本面分析代理 - 加密货币版本"""
    data = state["data"]
    tickers = data["tickers"]
    analysis_results = {}
    
    for ticker in tickers:
        try:
            # 获取市场数据
            ticker_24h = binance.get_ticker_24h(ticker)
            
            # 分析数据
            volume = float(ticker_24h['volume'])
            price_change = float(ticker_24h['priceChangePercent'])
            weighted_avg_price = float(ticker_24h['weightedAvgPrice'])
            
            # 分析基本面
            analysis = analyze_fundamentals(volume, price_change, weighted_avg_price)
            
            analysis_results[ticker] = {
                "signal": analysis["signal"],
                "confidence": analysis["confidence"],
                "reasoning": analysis["reasoning"],
                "metrics": {
                    "volume_24h": volume,
                    "price_change_24h": price_change,
                    "weighted_avg_price": weighted_avg_price
                }
            }
            
        except Exception as e:
            logger.error(f"分析{ticker}时出错: {e}")
            analysis_results[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": f"分析出错: {str(e)}",
                "metrics": {}
            }
    
    return {
        "data": {
            **state["data"],
            "analyst_signals": {
                **state["data"].get("analyst_signals", {}),
                "fundamentals_agent": analysis_results
            }
        }
    }