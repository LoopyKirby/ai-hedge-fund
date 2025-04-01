from graph.state import AgentState
from tools.binance_data import binance
import logging

logger = logging.getLogger(__name__)

def ben_graham_agent(state: AgentState):
    """Ben Graham 分析代理 - 加密货币版本"""
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
            
            # 计算30天平均价格作为参考价
            avg_price = sum(float(k[4]) for k in klines) / len(klines)
            
            # Graham 风格分析（寻找被低估的资产）
            signal = "neutral"
            confidence = 50.0
            reasons = []
            
            # 价格分析（相对于30天平均）
            price_to_avg = current_price / avg_price
            if price_to_avg < 0.8:
                signal = "bullish"
                confidence += 20
                reasons.append(f"价格显著低于30天平均值 ({price_to_avg:.2f})")
            elif price_to_avg > 1.2:
                signal = "bearish"
                confidence += 15
                reasons.append(f"价格显著高于30天平均值 ({price_to_avg:.2f})")
            
            analysis_results[ticker] = {
                "signal": signal,
                "confidence": min(confidence, 100),
                "reasoning": " | ".join(reasons),
                "metrics": {
                    "current_price": current_price,
                    "avg_price_30d": avg_price,
                    "price_to_avg_ratio": price_to_avg,
                    "volume_24h": volume
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
                "ben_graham_agent": analysis_results
            }
        }
    }