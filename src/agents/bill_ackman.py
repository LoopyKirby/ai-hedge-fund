from langchain_openai import ChatOpenAI
from graph.state import AgentState, show_agent_reasoning
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
from tools.binance_data import binance
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def bill_ackman_agent(state: AgentState):
    """Bill Ackman 分析代理 - 加密货币版本"""
    data = state["data"]
    tickers = data["tickers"]
    analysis_results = {}
    
    for ticker in tickers:
        try:
            # 获取市场数据
            ticker_24h = binance.get_ticker_24h(ticker)
            klines = binance.get_klines(ticker, interval='1d', limit=30)  # 30天数据
            klines_4h = binance.get_klines(ticker, interval='4h', limit=30)  # 4小时数据，用于短期分析
            
            # 分析数据
            current_price = float(ticker_24h['lastPrice'])
            volume = float(ticker_24h['volume'])
            price_change = float(ticker_24h['priceChangePercent'])
            weighted_avg_price = float(ticker_24h['weightedAvgPrice'])
            
            # 计算趋势
            closing_prices = [float(k[4]) for k in klines]
            price_trend = (closing_prices[-1] - closing_prices[0]) / closing_prices[0] * 100
            
            # 计算短期趋势（4小时）
            short_term_prices = [float(k[4]) for k in klines_4h]
            short_term_trend = (short_term_prices[-1] - short_term_prices[0]) / short_term_prices[0] * 100
            
            # 计算波动性
            daily_changes = [(float(k[4]) - float(k[1])) / float(k[1]) * 100 for k in klines]
            volatility = sum(abs(x) for x in daily_changes) / len(daily_changes)
            
            # Ackman 风格分析（激进投资）
            signal = "neutral"
            confidence = 50.0
            reasons = []
            
            # 1. 趋势分析（权重：40%）
            if price_trend > 20 and short_term_trend > 5:
                signal = "bullish"
                confidence += 20
                reasons.append(f"强劲上涨趋势（月度：{price_trend:.2f}%，短期：{short_term_trend:.2f}%）")
            elif price_trend < -20 and short_term_trend < -5:
                signal = "bearish"
                confidence += 20
                reasons.append(f"显著下跌趋势（月度：{price_trend:.2f}%，短期：{short_term_trend:.2f}%）")
            
            # 2. 价格动量（权重：20%）
            if price_change > 5 and volume > float(ticker_24h['volume']) * 1.2:
                if signal == "bullish":
                    confidence += 10
                    reasons.append(f"强劲买入动能（24h变化：{price_change:.2f}%，成交量增加）")
                else:
                    signal = "bullish"
                    confidence += 5
                    reasons.append(f"潜在反转信号（24h变化：{price_change:.2f}%）")
            elif price_change < -5 and volume > float(ticker_24h['volume']) * 1.2:
                if signal == "bearish":
                    confidence += 10
                    reasons.append(f"强劲卖出动能（24h变化：{price_change:.2f}%，成交量增加）")
                else:
                    signal = "bearish"
                    confidence += 5
                    reasons.append(f"潜在反转信号（24h变化：{price_change:.2f}%）")
            
            # 3. 波动性分析（权重：20%）
            if volatility > 5:  # 高波动性
                confidence = max(30, confidence - 10)  # 降低信心度
                reasons.append(f"高波动性环境（日均波动：{volatility:.2f}%）需要谨慎")
            else:  # 低波动性
                confidence += 10
                reasons.append(f"市场波动性较低（{volatility:.2f}%），信号更可靠")
            
            # 4. 价格相对于加权平均价（权重：20%）
            price_to_wavg = (current_price - weighted_avg_price) / weighted_avg_price * 100
            if abs(price_to_wavg) > 3:
                if price_to_wavg > 0 and signal == "bullish":
                    confidence += 10
                    reasons.append(f"价格高于加权平均价 {price_to_wavg:.2f}%，确认上涨趋势")
                elif price_to_wavg < 0 and signal == "bearish":
                    confidence += 10
                    reasons.append(f"价格低于加权平均价 {abs(price_to_wavg):.2f}%，确认下跌趋势")
            
            # 确保信心度不超过100
            confidence = min(confidence, 100)
            
            # 如果没有明确信号，保持中性
            if confidence < 40:
                signal = "neutral"
                reasons.append("信号强度不足，保持中性观望")
            
            analysis_results[ticker] = BillAckmanSignal(
                signal=signal,
                confidence=confidence,
                reasoning=" | ".join(reasons)
            ).dict()
            
        except Exception as e:
            logger.error(f"分析{ticker}时出错: {str(e)}")
            analysis_results[ticker] = BillAckmanSignal(
                signal="neutral",
                confidence=0,
                reasoning=f"分析出错: {str(e)}"
            ).dict()
    
    return {
        "data": {
            **state["data"],
            "bill_ackman": analysis_results  # 移除_agent后缀
        }
    }


def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages, and potential for long-term growth.
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze business quality"
        }
    
    # 1. Multi-period revenue growth analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2:
        # Check if overall revenue grew from first to last
        initial, final = revenues[0], revenues[-1]
        if initial and final and final > initial:
            # Simple growth rate
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:  # e.g., 50% growth over the available time
                score += 2
                details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period.")
            else:
                score += 1
                details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
        else:
            details.append("Revenue did not grow significantly or data insufficient.")
    else:
        details.append("Not enough revenue data for multi-period trend.")
    
    # 2. Operating margin and free cash flow consistency
    # We'll check if operating_margin or free_cash_flow are consistently positive/improving
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if op_margin_vals:
        # Check if the majority of operating margins are > 15%
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins have often exceeded 15%.")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data across periods.")
    
    if fcf_vals:
        # Check if free cash flow is positive in most periods
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive free cash flow.")
        else:
            details.append("Free cash flow not consistently positive.")
    else:
        details.append("No free cash flow data across periods.")
    
    # 3. Return on Equity (ROE) check from the latest metrics
    # (If you want multi-period ROE, you'd need that in financial_line_items as well.)
    latest_metrics = metrics[0]
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        details.append(f"High ROE of {latest_metrics.return_on_equity:.1%}, indicating potential moat.")
    elif latest_metrics.return_on_equity:
        details.append(f"ROE of {latest_metrics.return_on_equity:.1%} is not indicative of a strong moat.")
    else:
        details.append("ROE data not available in metrics.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze financial discipline"
        }
    
    # 1. Multi-period debt ratio or debt_to_equity
    # Check if the company's leverage is stable or improving
    debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    
    # If we have multi-year data, see if D/E ratio has gone down or stayed <1 across most periods
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
            score += 2
            details.append("Debt-to-equity < 1.0 for the majority of periods.")
        else:
            details.append("Debt-to-equity >= 1.0 in many periods.")
    else:
        # Fallback to total_liabilities/total_assets if D/E not available
        liab_to_assets = []
        for item in financial_line_items:
            if item.total_liabilities and item.total_assets and item.total_assets > 0:
                liab_to_assets.append(item.total_liabilities / item.total_assets)
        
        if liab_to_assets:
            below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
            if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                score += 2
                details.append("Liabilities-to-assets < 50% for majority of periods.")
            else:
                details.append("Liabilities-to-assets >= 50% in many periods.")
        else:
            details.append("No consistent leverage ratio data available.")
    
    # 2. Capital allocation approach (dividends + share counts)
    # If the company paid dividends or reduced share count over time, it may reflect discipline
    dividends_list = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if dividends_list:
        # Check if dividends were paid (i.e., negative outflows to shareholders) in most periods
        paying_dividends_count = sum(1 for d in dividends_list if d < 0)
        if paying_dividends_count >= (len(dividends_list) // 2 + 1):
            score += 1
            details.append("Company has a history of returning capital to shareholders (dividends).")
        else:
            details.append("Dividends not consistently paid or no data.")
    else:
        details.append("No dividend data found across periods.")
    
    # Check for decreasing share count (simple approach):
    # We can compare first vs last if we have at least two data points
    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    if len(shares) >= 2:
        if shares[-1] < shares[0]:
            score += 1
            details.append("Outstanding shares have decreased over time (possible buybacks).")
        else:
            details.append("Outstanding shares have not decreased over the available periods.")
    else:
        details.append("No multi-period share count data to assess buybacks.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    We can do a simplified DCF or an FCF-based approach.
    This function currently uses the latest free cash flow only, 
    but you could expand it to use an average or multi-year FCF approach.
    """
    if not financial_line_items or market_cap is None:
        return {
            "score": 0,
            "details": "Insufficient data to perform valuation"
        }
    
    # Example: use the most recent item for FCF
    latest = financial_line_items[-1]  # the last one is presumably the most recent
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0
    
    # For demonstration, let's do a naive approach:
    growth_rate = 0.06
    discount_rate = 0.10
    terminal_multiple = 15
    projection_years = 5
    
    if fcf <= 0:
        return {
            "score": 0,
            "details": f"No positive FCF for valuation; FCF = {fcf}",
            "intrinsic_value": None
        }
    
    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv
    
    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) \
                     / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value
    
    # Compare with market cap => margin of safety
    margin_of_safety = (intrinsic_value - market_cap) / market_cap
    
    score = 0
    if margin_of_safety > 0.3:
        score += 3
    elif margin_of_safety > 0.1:
        score += 1
    
    details = [
        f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
        f"Market cap: ~{market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}"
    ]
    
    return {
        "score": score,
        "details": "; ".join(details),
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety
    }


def generate_ackman_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BillAckmanSignal:
    """
    Generates investment decisions in the style of Bill Ackman.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Bill Ackman AI agent, making investment decisions using his principles:

            1. Seek high-quality businesses with durable competitive advantages (moats).
            2. Prioritize consistent free cash flow and growth potential.
            3. Advocate for strong financial discipline (reasonable leverage, efficient capital allocation).
            4. Valuation matters: target intrinsic value and margin of safety.
            5. Invest with high conviction in a concentrated portfolio for the long term.
            6. Potential activist approach if management or operational improvements can unlock value.
            

            Rules:
            - Evaluate brand strength, market position, or other moats.
            - Check free cash flow generation, stable or growing earnings.
            - Analyze balance sheet health (reasonable debt, good ROE).
            - Buy at a discount to intrinsic value; higher discount => stronger conviction.
            - Engage if management is suboptimal or if there's a path for strategic improvements.
            - Provide a rational, data-driven recommendation (bullish, bearish, or neutral).
            
            When providing your reasoning, be thorough and specific by:
            1. Explaining the quality of the business and its competitive advantages in detail
            2. Highlighting the specific financial metrics that most influenced your decision (FCF, margins, leverage)
            3. Discussing any potential for operational improvements or management changes
            4. Providing a clear valuation assessment with numerical evidence
            5. Identifying specific catalysts that could unlock value
            6. Using Bill Ackman's confident, analytical, and sometimes confrontational style
            
            For example, if bullish: "This business generates exceptional free cash flow with a 15% margin and has a dominant market position that competitors can't easily replicate. Trading at only 12x FCF, there's a 40% discount to intrinsic value, and management's recent capital allocation decisions suggest..."
            For example, if bearish: "Despite decent market position, FCF margins have deteriorated from 12% to 8% over three years. Management continues to make poor capital allocation decisions by pursuing low-ROIC acquisitions. Current valuation at 18x FCF provides no margin of safety given the operational challenges..."
            """
        ),
        (
            "human",
            """Based on the following analysis, create an Ackman-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return the trading signal in this JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=BillAckmanSignal, 
        agent_name="bill_ackman_agent", 
        default_factory=create_default_bill_ackman_signal,
    )
