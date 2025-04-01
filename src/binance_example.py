from binance_client import BinanceClient
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_ticker_data(data):
    logger.info(f"收到BTC/USDT价格数据: {data}")
    if isinstance(data, dict):
        price = data.get("c", "N/A")
        volume = data.get("v", "N/A")
        logger.info(f"BTC/USDT - 最新价格: {price} USDT, 24小时成交量: {volume} BTC")

def main():
    logger.info("启动币安实时数据监控...")
    api_key = "R3pyFteTDZvD3kEaY2JkHDfFCb6r2J4U3gp0xetYAS33xkLjIZ2xdgCiMlZiR9Od"
    api_secret = ""
    client = BinanceClient(api_key, api_secret)
    logger.info("币安客户端初始化完成")
    client.connect()
    logger.info("正在连接到币安WebSocket服务器...")
    time.sleep(2)
    streams = ["btcusdt@ticker"]
    client.subscribe(streams, handle_ticker_data)
    logger.info(f"已订阅数据流: {streams}")
    try:
        logger.info("开始监控BTC/USDT价格...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("正在关闭连接...")
        client.close()
        logger.info("程序已退出")

if __name__ == "__main__":
    main()