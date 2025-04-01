from graph.state import AgentState
from tools.binance_data import binance
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def risk_management_agent(state: AgentState) -> Dict[str, Any]:
    """风险管理代理"""
    data = state["data"]
    tickers = data["tickers"]
    risk_analysis = {}
    
    for ticker in tickers:
        try:
            ticker_24h = binance.get_ticker_24h(ticker)
            
            # 计算基本风险指标
            price = float(ticker_24h['lastPrice'])
            volume = float(ticker_24h['volume'])
            price_change = float(ticker_24h['priceChangePercent'])
            
            # 风险评估
            risk_level = "medium"
            max_position = 1.0  # 默认允许100%仓位
            
            # 基于价格波动调整风险
            if abs(price_change) > 10:
                risk_level = "high"
                max_position = 0.3  # 高风险时限制到30%
            elif abs(price_change) < 3:
                risk_level = "low"
                max_position = 1.0  # 低风险时允许满仓
            
            # 基于成交量调整
            avg_volume = float(ticker_24h['quoteVolume']) / price
            if volume > avg_volume * 2:
                risk_level = "high"
                max_position *= 0.8  # 成交量异常时降低20%仓位
            
            risk_analysis[ticker] = {
                "risk_level": risk_level,
                "max_position": max_position,
                "current_price": price,  # 添加当前价格
                "remaining_position_limit": max_position * 100000,  # 假设总资金100000
                "metrics": {
                    "price": price,
                    "volume_24h": volume,
                    "price_change_24h": price_change
                }
            }
            
        except Exception as e:
            logger.error(f"分析{ticker}时出错: {str(e)}")
            risk_analysis[ticker] = {
                "risk_level": "high",
                "max_position": 0,
                "current_price": 0,
                "remaining_position_limit": 0,
                "metrics": {}
            }
    
    return {
        "data": {
            **state["data"],
            "analyst_signals": {
                **state["data"].get("analyst_signals", {}),
                "risk_management_agent": risk_analysis
            }
        }
    }