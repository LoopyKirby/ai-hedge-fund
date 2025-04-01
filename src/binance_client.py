import json
import websocket
import threading
from typing import Callable, Dict, List, Optional
import logging
import ssl
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws: Optional[websocket.WebSocketApp] = None
        self.subscribed_streams: List[str] = []
        self.callbacks: Dict[str, Callable] = {}
        self.base_endpoint = "wss://stream.binance.com:9443"
        self.reconnect_delay = 5
        self.running = True
        
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            logger.info(f"收到数据: {data}")
            if "stream" in data:
                stream_name = data["stream"]
                if stream_name in self.callbacks:
                    self.callbacks[stream_name](data["data"])
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket错误: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket连接已关闭")
        if self.running:
            logger.info(f"{self.reconnect_delay}秒后尝试重新连接...")
            time.sleep(self.reconnect_delay)
            self.connect()

    def _on_open(self, ws):
        logger.info("WebSocket连接已建立")
        if self.subscribed_streams:
            self._subscribe(self.subscribed_streams)

    def connect(self):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            "wss://stream.binance.com:443/stream",
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        ws_thread = threading.Thread(
            target=lambda: self.ws.run_forever(
                sslopt={
                    "cert_reqs": ssl.CERT_NONE,
                    "check_hostname": False
                }
            )
        )
        ws_thread.daemon = True
        ws_thread.start()

    def subscribe(self, streams: List[str], callback: Callable):
        self.subscribed_streams.extend(streams)
        for stream in streams:
            self.callbacks[stream] = callback
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self._subscribe(streams)

    def _subscribe(self, streams: List[str]):
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        self.ws.send(json.dumps(subscribe_message))

    def unsubscribe(self, streams: List[str]):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            unsubscribe_message = {
                "method": "UNSUBSCRIBE",
                "params": streams,
                "id": 1
            }
            self.ws.send(json.dumps(unsubscribe_message))
            for stream in streams:
                if stream in self.subscribed_streams:
                    self.subscribed_streams.remove(stream)
                if stream in self.callbacks:
                    del self.callbacks[stream]

    def close(self):
        self.running = False
        if self.ws:
            self.ws.close()