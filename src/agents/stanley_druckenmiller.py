from graph.state import AgentState
from tools.binance_data import binance
import logging

logger = logging.getLogger(__name__)

def stanley_druckenmiller_agent(state: AgentState):
    """Stanley Druckenmiller 分析代理 - 加密货币版本"""
    data = state["data"]
    tickers = data["tickers"]
    analysis_results = {}
    
    for ticker in tickers:
        try:
            # 获取市场数据
            ticker_24h = binance.get_ticker_24h(ticker)
            klines = binance.get_klines(ticker, interval='1d', limit=30)
            
            # 分析数据
            current_price = float(ticker_24h['lastPrice'])
            volume = float(ticker_24h['volume'])
            price_change = float(ticker_24h['priceChangePercent'])
            
            # 计算动量指标
            closing_prices = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            # Druckenmiller 风格分析（关注动量和趋势）
            signal = "neutral"
            confidence = 50.0
            reasons = []
            
            # 分析价格动量
            price_momentum = price_change
            if price_momentum > 5:
                signal = "bullish"
                confidence += 25
                reasons.append(f"强劲的价格动量 ({price_momentum:.2f}%)")
            elif price_momentum < -5:
                signal = "bearish"
                confidence += 20
                reasons.append(f"负面的价格动量 ({price_momentum:.2f}%)")
            
            # 分析交易量趋势
            volume_trend = (sum(volumes[-7:]) / 7) / (sum(volumes[-30:-7]) / 23)
            if volume_trend > 1.3 and price_momentum > 0:
                confidence += 15
                reasons.append("交易量支持上涨趋势")
            
            analysis_results[ticker] = {
                "signal": signal,
                "confidence": min(confidence, 100),
                "reasoning": " | ".join(reasons),
                "metrics": {
                    "current_price": current_price,
                    "volume_24h": volume,
                    "price_momentum": price_momentum,
                    "volume_trend": volume_trend
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
                "stanley_druckenmiller_agent": analysis_results
            }
        }
    }
