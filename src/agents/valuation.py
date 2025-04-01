from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import json
from tools.binance_data import binance
import logging

logger = logging.getLogger(__name__)


##### Valuation Agent #####
def valuation_agent(state: AgentState):
    """估值分析代理 - 加密货币版本"""
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
            
            # 计算估值指标
            avg_price_30d = sum(float(k[4]) for k in klines) / len(klines)
            avg_volume_30d = sum(float(k[5]) for k in klines) / len(klines)
            
            # 估值分析
            signal = "neutral"
            confidence = 50.0
            reasons = []
            
            # 价格相对于30天均价的偏离
            price_deviation = (current_price - avg_price_30d) / avg_price_30d * 100
            if price_deviation < -20:
                signal = "bullish"
                confidence += 20
                reasons.append(f"价格显著低于30天均价 ({price_deviation:.2f}%)")
            elif price_deviation > 20:
                signal = "bearish"
                confidence += 20
                reasons.append(f"价格显著高于30天均价 ({price_deviation:.2f}%)")
            
            # 交易量分析
            volume_ratio = volume / avg_volume_30d
            if volume_ratio > 2:
                confidence += 15
                reasons.append(f"交易量显著高于平均水平 ({volume_ratio:.2f}x)")
            
            analysis_results[ticker] = {
                "signal": signal,
                "confidence": min(confidence, 100),
                "reasoning": " | ".join(reasons),
                "metrics": {
                    "current_price": current_price,
                    "avg_price_30d": avg_price_30d,
                    "price_deviation": price_deviation,
                    "volume_ratio": volume_ratio
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
                "valuation_agent": analysis_results
            }
        }
    }


def calculate_owner_earnings_value(
    net_income: float,
    depreciation: float,
    capex: float,
    working_capital_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """
    Calculates the intrinsic value using Buffett's Owner Earnings method.

    Owner Earnings = Net Income
                    + Depreciation/Amortization
                    - Capital Expenditures
                    - Working Capital Changes

    Args:
        net_income: Annual net income
        depreciation: Annual depreciation and amortization
        capex: Annual capital expenditures
        working_capital_change: Annual change in working capital
        growth_rate: Expected growth rate
        required_return: Required rate of return (Buffett typically uses 15%)
        margin_of_safety: Margin of safety to apply to final value
        num_years: Number of years to project

    Returns:
        float: Intrinsic value with margin of safety
    """
    if not all([isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]]):
        return 0

    # Calculate initial owner earnings
    owner_earnings = net_income + depreciation - capex - working_capital_change

    if owner_earnings <= 0:
        return 0

    # Project future owner earnings
    future_values = []
    for year in range(1, num_years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        discounted_value = future_value / (1 + required_return) ** year
        future_values.append(discounted_value)

    # Calculate terminal value (using perpetuity growth formula)
    terminal_growth = min(growth_rate, 0.03)  # Cap terminal growth at 3%
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / (1 + required_return) ** num_years

    # Sum all values and apply margin of safety
    intrinsic_value = sum(future_values) + terminal_value_discounted
    value_with_safety_margin = intrinsic_value * (1 - margin_of_safety)

    return value_with_safety_margin


def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    Computes the discounted cash flow (DCF) for a given company based on the current free cash flow.
    Use this function to calculate the intrinsic value of a stock.
    """
    # Estimate the future cash flows based on the growth rate
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]

    # Calculate the present value of projected cash flows
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)

    # Calculate the terminal value
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    # Sum up the present values and terminal value
    dcf_value = sum(present_values) + terminal_present_value

    return dcf_value


def calculate_working_capital_change(
    current_working_capital: float,
    previous_working_capital: float,
) -> float:
    """
    Calculate the absolute change in working capital between two periods.
    A positive change means more capital is tied up in working capital (cash outflow).
    A negative change means less capital is tied up (cash inflow).

    Args:
        current_working_capital: Current period's working capital
        previous_working_capital: Previous period's working capital

    Returns:
        float: Change in working capital (current - previous)
    """
    return current_working_capital - previous_working_capital
