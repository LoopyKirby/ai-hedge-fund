from graph.state import AgentState
from tools.binance_data import binance
import logging

logger = logging.getLogger(__name__)

def cathie_wood_agent(state: AgentState):
    """Cathie Wood 分析代理 - 加密货币版本"""
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
            
            # 计算趋势
            closing_prices = [float(k[4]) for k in klines]
            price_trend = (closing_prices[-1] - closing_prices[0]) / closing_prices[0] * 100
            
            # Cathie Wood 风格分析（关注创新和高增长）
            signal = "neutral"
            confidence = 50.0
            reasons = []
            
            # 分析价格趋势
            if price_trend > 30:  # 强劲上涨趋势
                signal = "bullish"
                confidence += 20
                reasons.append(f"显著的上涨趋势 ({price_trend:.2f}%)")
            elif price_trend < -30:
                signal = "bearish"
                confidence += 15
                reasons.append(f"显著的下跌趋势 ({price_trend:.2f}%)")
            
            # 分析交易量
            avg_volume = sum(float(k[5]) for k in klines) / len(klines)
            if volume > avg_volume * 2:
                confidence += 15
                reasons.append("交易量大幅增加，表明市场兴趣提升")
            
            # 分析短期动能
            if price_change > 5:
                signal = "bullish"
                confidence += 10
                reasons.append(f"短期价格动能强劲 ({price_change:.2f}%)")
            elif price_change < -5:
                signal = "bearish"
                confidence += 10
                reasons.append(f"短期价格动能减弱 ({price_change:.2f}%)")
            
            analysis_results[ticker] = {
                "signal": signal,
                "confidence": min(confidence, 100),
                "reasoning": " | ".join(reasons),
                "metrics": {
                    "current_price": current_price,
                    "volume_24h": volume,
                    "price_change_24h": price_change,
                    "price_trend_30d": price_trend
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
                "cathie_wood_agent": analysis_results
            }
        }
    }