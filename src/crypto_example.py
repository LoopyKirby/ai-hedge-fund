from datetime import datetime, timedelta
import logging
from agents.technicals import technical_analyst_agent
from graph.state import AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 设置分析参数
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # 设置要分析的加密货币
    tickers = ['BTCUSDT', 'ETHUSDT']
    
    # 创建初始状态
    state = AgentState({
        "messages": [],
        "data": {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {}
        },
        "metadata": {
            "show_reasoning": True
        }
    })
    
    # 运行技术分析
    logger.info(f"开始分析加密货币: {', '.join(tickers)}")
    result = technical_analyst_agent(state)
    
    # 显示分析结果
    for ticker in tickers:
        signals = result["data"]["analyst_signals"]["technical_analyst_agent"][ticker]
        logger.info(f"\n{ticker} 技术分析结果:")
        logger.info(f"整体信号: {signals['signal']}")
        logger.info(f"置信度: {signals['confidence']}%")
        
        # 显示各个策略的信号
        logger.info("\n各策略信号:")
        for strategy, data in signals["strategy_signals"].items():
            logger.info(f"{strategy}:")
            logger.info(f"  信号: {data['signal']}")
            logger.info(f"  置信度: {data['confidence']}%")
            if 'metrics' in data:
                logger.info(f"  指标: {data['metrics']}")

if __name__ == "__main__":
    main() 