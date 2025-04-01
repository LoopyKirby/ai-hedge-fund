import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BinanceAPI:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
        # 创建会话对象
        self.session = requests.Session()
        
        # 配置重试策略
        retries = Retry(
            total=5,  # 最多重试5次
            backoff_factor=1,  # 重试间隔时间
            status_forcelist=[500, 502, 503, 504],  # 需要重试的HTTP状态码
            allowed_methods=["GET"]  # 只对GET请求重试
        )
        
        # 将重试策略应用到会话
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """发送请求并处理错误"""
        for attempt in range(3):  # 最多尝试3次
            try:
                url = f"{self.base_url}{endpoint}"
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/3): {str(e)}")
                if attempt == 2:  # 最后一次尝试
                    raise
                time.sleep(2 ** attempt)  # 指数退避
    
    def get_ticker_24h(self, symbol: str) -> Dict:
        """获取24小时价格变动情况"""
        try:
            return self._make_request("/ticker/24hr", {"symbol": symbol})
        except Exception as e:
            logger.error(f"获取24小时数据失败 ({symbol}): {str(e)}")
            # 返回默认值而不是抛出异常
            return {
                "lastPrice": "0",
                "volume": "0",
                "priceChangePercent": "0",
                "weightedAvgPrice": "0",
                "quoteVolume": "0"
            }

    def get_klines(self, symbol: str, interval: str = '1d', 
                  limit: int = 30, start_time: Optional[int] = None, 
                  end_time: Optional[int] = None) -> List:
        """获取K线数据"""
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
                
            return self._make_request("/klines", params)
        except Exception as e:
            logger.error(f"获取K线数据失败 ({symbol}): {str(e)}")
            # 返回空列表而不是抛出异常
            return []

    def get_trades(self, symbol: str, limit: int = 1000) -> List:
        """获取最近的交易"""
        try:
            return self._make_request("/trades", {"symbol": symbol, "limit": limit})
        except Exception as e:
            logger.error(f"获取交易数据失败 ({symbol}): {str(e)}")
            # 返回空列表而不是抛出异常
            return []

    def get_depth(self, symbol: str, limit: int = 100) -> Dict:
        """获取订单簿深度"""
        endpoint = f"{self.base_url}/depth"
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_agg_trades(self, symbol: str, 
                      start_time: Optional[int] = None,
                      end_time: Optional[int] = None,
                      limit: int = 500) -> List[Dict]:
        """获取归集交易"""
        endpoint = f"{self.base_url}/aggTrades"
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

# 创建单例实例
binance = BinanceAPI() 