from rich.progress import Progress as RichProgress
from rich.console import Console

class Progress:
    def __init__(self):
        self.console = Console()
        self.progress = None
        self.tasks = {}
        
    def start(self):
        """启动进度显示"""
        self.progress = RichProgress()
        self.progress.start()
        
    def update_status(self, agent_name: str, ticker: str, status: str):
        """更新代理状态"""
        if self.progress:
            self.console.print(f"[bold blue]{agent_name}[/bold blue] - {ticker}: {status}")

    def stop(self):
        """停止进度显示"""
        if self.progress:
            self.progress.stop()

    def get_status(self, agent_name: str, ticker: str) -> str:
        return self.current_status.get(agent_name, {}).get(ticker, "")

progress = Progress()
