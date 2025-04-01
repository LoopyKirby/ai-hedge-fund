from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
from tools.binance_data import binance
from data.models import Price, FinancialMetrics, FinancialMetricsResponse
import os
import requests
import logging

logger = logging.getLogger(__name__)

def get_prices(ticker: str, start_date: str, end_date: str) -> List[Price]:
    """获取价格历史数据"""
    try:
        if ticker.endswith('USDT'):
            # 转换日期为时间戳
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # 获取K线数据
            klines = binance.get_klines(
                symbol=ticker,
                interval='1d',
                start_time=start_ts,
                end_time=end_ts
            )
            
            # 转换为Price对象
            prices = []
            for k in klines:
                price = Price(
                    time=datetime.fromtimestamp(k[0]/1000).strftime('%Y-%m-%d'),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    ticker=ticker
                )
                prices.append(price)
            return prices
            
        return []
        
    except Exception as e:
        logger.error(f"获取价格数据失败: {e}")
        return []

def prices_to_df(prices: List[Price]) -> pd.DataFrame:
    """将价格列表转换为DataFrame"""
    if not prices:
        return pd.DataFrame()
        
    data = []
    for p in prices:
        data.append({
            'time': p.time,
            'open': p.open,
            'high': p.high,
            'low': p.low,
            'close': p.close,
            'volume': p.volume,
            'ticker': p.ticker
        })
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def get_financial_metrics(ticker: str, end_date: str, period: str = "annual", limit: int = 5) -> List[FinancialMetrics]:
    """获取财务指标，支持加密货币"""
    try:
        if ticker.endswith('USDT'):
            ticker_24h = binance.get_ticker_24h(ticker)
            klines = binance.get_klines(ticker, interval='1d', limit=30)
            
            metrics = FinancialMetrics(
                ticker=ticker,
                time=end_date,
                revenue=float(ticker_24h['volume']),
                gross_profit=float(ticker_24h['priceChangePercent']),
                operating_income=float(ticker_24h['weightedAvgPrice']),
                net_income=float(ticker_24h['lastPrice']),
                eps=0.0,
                shares_outstanding=float(ticker_24h['volume']),
                free_cash_flow=float(ticker_24h['quoteVolume'])
            )
            return [metrics]
            
        # 如果是传统股票
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key
        
        url = f"https://api.financialdatasets.ai/financials/?ticker={ticker}&period={period}&limit={limit}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return []
            
        metrics_response = FinancialMetricsResponse(**response.json())
        return metrics_response.financials
        
    except Exception as e:
        logger.error(f"获取财务数据失败: {e}")
        return []

def get_market_cap(ticker: str) -> Optional[float]:
    """获取市值，支持加密货币"""
    try:
        if ticker.endswith('USDT'):
            ticker_24h = binance.get_ticker_24h(ticker)
            # 对于加密货币，使用当前价格 * 24小时成交量作为市值估算
            return float(ticker_24h['lastPrice']) * float(ticker_24h['volume'])
        
        # 如果是传统股票
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key
            
        url = f"https://api.financialdatasets.ai/market_cap/?ticker={ticker}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return float(response.json()["market_cap"])
        return None
        
    except Exception as e:
        logger.error(f"获取市值失败: {e}")
        return None

def search_line_items(ticker: str) -> Dict:
    """搜索财务项目，支持加密货币"""
    try:
        if ticker.endswith('USDT'):
            ticker_24h = binance.get_ticker_24h(ticker)
            return {
                'price': float(ticker_24h['lastPrice']),
                'volume': float(ticker_24h['volume']),
                'price_change': float(ticker_24h['priceChangePercent']),
                'weighted_avg_price': float(ticker_24h['weightedAvgPrice']),
                'quote_volume': float(ticker_24h['quoteVolume'])
            }
            
        # 如果是传统股票
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key
            
        url = f"https://api.financialdatasets.ai/line_items/?ticker={ticker}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        return {}
        
    except Exception as e:
        logger.error(f"搜索财务项目失败: {e}")
        return {}

def get_insider_trades(ticker: str, limit: int = 100) -> List[Dict]:
    """获取内部交易数据，支持加密货币"""
    try:
        if ticker.endswith('USDT'):
            # 对于加密货币，使用大额交易作为"内部交易"的替代
            trades = binance.get_trades(ticker, limit=limit)
            large_trades = [
                {
                    'date': datetime.fromtimestamp(t['time']/1000).strftime('%Y-%m-%d'),
                    'type': 'buy' if t['isBuyerMaker'] else 'sell',
                    'amount': float(t['qty']),
                    'price': float(t['price'])
                }
                for t in trades
                if float(t['qty']) * float(t['price']) > 100000  # 只返回大于10万USDT的交易
            ]
            return large_trades
            
        # 如果是传统股票
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key
            
        url = f"https://api.financialdatasets.ai/insider_trades/?ticker={ticker}&limit={limit}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()["trades"]
        return []
        
    except Exception as e:
        logger.error(f"获取内部交易数据失败: {e}")
        return []

def get_news_sentiment(ticker: str, days: int = 30) -> Dict:
    """获取新闻情绪数据，支持加密货币"""
    try:
        if ticker.endswith('USDT'):
            # 对于加密货币，使用价格变化和交易量作为市场情绪的替代指标
            klines = binance.get_klines(ticker, interval='1d', limit=days)
            
            # 计算平均涨跌幅
            price_changes = [
                (float(k[4]) - float(k[1])) / float(k[1]) * 100 
                for k in klines
            ]
            avg_change = sum(price_changes) / len(price_changes)
            
            # 计算成交量趋势
            volumes = [float(k[5]) for k in klines]
            volume_trend = (sum(volumes[-7:]) / 7) / (sum(volumes[-days:-7]) / (days-7))
            
            return {
                'sentiment_score': avg_change / 10,  # 将涨跌幅转换为-1到1之间的情绪分数
                'volume_trend': volume_trend,
                'price_trend': avg_change,
                'news_count': 0  # 加密货币没有传统新闻源数据
            }
            
        # 如果是传统股票
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key
            
        url = f"https://api.financialdatasets.ai/news_sentiment/?ticker={ticker}&days={days}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        return {}
        
    except Exception as e:
        logger.error(f"获取新闻情绪数据失败: {e}")
        return {}
    
# ... 其他导入和函数保持不变 ...

def get_company_news(ticker: str, days: int = 30) -> List[Dict]:
    """获取公司新闻，支持加密货币"""
    try:
        if ticker.endswith('USDT'):
            # 对于加密货币，使用最近的大额交易作为"新闻"
            trades = binance.get_trades(ticker, limit=50)
            news = [
                {
                    'date': datetime.fromtimestamp(t['time']/1000).strftime('%Y-%m-%d'),
                    'title': f"Large {t['isBuyerMaker'] and 'Buy' or 'Sell'} Order",
                    'summary': f"Trade amount: {float(t['qty']):.2f} at price {float(t['price']):.2f}",
                    'sentiment': 'positive' if t['isBuyerMaker'] else 'negative'
                }
                for t in trades
                if float(t['qty']) * float(t['price']) > 100000  # 只返回大于10万USDT的交易
            ]
            return news
            
        # 如果是传统股票
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key
            
        url = f"https://api.financialdatasets.ai/company_news/?ticker={ticker}&days={days}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()["news"]
        return []
        
    except Exception as e:
        logger.error(f"获取公司新闻失败: {e}")
        return []