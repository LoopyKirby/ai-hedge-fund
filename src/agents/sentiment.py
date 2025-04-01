from graph.state import AgentState
from tools.binance_data import binance
import logging

logger = logging.getLogger(__name__)

def analyze_sentiment(buy_sell_ratio: float, price_change: float) -> dict:
    """分析市场情绪"""
    signal = "neutral"
    confidence = 50.0
    reasons = []
    
    # 分析买卖比率
    if buy_sell_ratio > 1.2:
        signal = "bullish"
        confidence += 15
        reasons.append(f"买盘压力大于卖盘 ({buy_sell_ratio:.2f})")
    elif buy_sell_ratio < 0.8:
        signal = "bearish"
        confidence += 15
        reasons.append(f"卖盘压力大于买盘 ({buy_sell_ratio:.2f})")
    
    # 分析价格变化
    if price_change > 5:
        signal = "bullish"
        confidence += 10
        reasons.append(f"价格强势上涨 ({price_change:.2f}%)")
    elif price_change < -5:
        signal = "bearish"
        confidence += 10
        reasons.append(f"价格大幅下跌 ({price_change:.2f}%)")
    
    return {
        "signal": signal,
        "confidence": min(confidence, 100),
        "reasoning": " | ".join(reasons)
    }

def sentiment_agent(state: AgentState):
    """情绪分析代理"""
    data = state["data"]
    tickers = data["tickers"]
    analysis_results = {}
    
    for ticker in tickers:
        try:
            # 获取市场数据
            ticker_24h = binance.get_ticker_24h(ticker)
            trades = binance.get_trades(ticker, limit=1000)
            
            # 计算买卖比率
            buy_trades = sum(1 for t in trades if not t['isBuyerMaker'])
            sell_trades = sum(1 for t in trades if t['isBuyerMaker'])
            buy_sell_ratio = buy_trades / sell_trades if sell_trades > 0 else 1.0
            
            # 获取价格变化
            price_change = float(ticker_24h['priceChangePercent'])
            
            # 分析情绪
            sentiment = analyze_sentiment(buy_sell_ratio, price_change)
            
            analysis_results[ticker] = {
                "signal": sentiment["signal"],
                "confidence": sentiment["confidence"],
                "reasoning": sentiment["reasoning"],
                "metrics": {
                    "buy_sell_ratio": buy_sell_ratio,
                    "price_change_24h": price_change,
                    "buy_trades": buy_trades,
                    "sell_trades": sell_trades
                }
            }
            
        except Exception as e:
            logger.error(f"分析{ticker}时出错: {str(e)}")
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
                "sentiment_agent": analysis_results
            }
        }
    }
