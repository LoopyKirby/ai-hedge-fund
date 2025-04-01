import requests
from datetime import datetime
from typing import List, Dict, Optional
from data.models import Price
import logging

logger = logging.getLogger(__name__)

def get_binance_klines(symbol: str, interval: str = '1d', 
                      start_time: Optional[int] = None, 
                      end_time: Optional[int] = None,
                      limit: int = 1000) -> List[Dict]:
    """从币安获取K线数据"""
    endpoint = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time
        
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"获取币安数据失败: {e}")
        return []

def get_crypto_prices(ticker: str, start_date: str, end_date: str) -> List[Price]:
    """获取加密货币价格数据"""
    # 转换日期为时间戳
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    # 获取币安数据
    klines = get_binance_klines(
        symbol=ticker,
        start_time=start_ts,
        end_time=end_ts
    )
    
    # 转换为Price对象列表
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