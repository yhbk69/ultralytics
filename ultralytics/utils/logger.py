# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from ultralytics.utils import LOGGER, MACOS, RANK
from ultralytics.utils.checks import check_requirements


class ConsoleLogger:
    """控制台输出捕获，支持批量流式传输到文件、API 或自定义回调。

    捕获 stdout/stderr 输出，并通过智能去重和可配置的批量处理进行流式传输。

    属性:
        destination (str | Path | None): 流式传输的目标（URL、Path，或 None 表示仅回调）。
        batch_size (int): 刷新前批量的行数（默认: 1 表示立即刷新）。
        flush_interval (float): 自动刷新之间的秒数（默认: 5.0）。
        on_flush (callable | None): 刷新时调用的可选回调函数，传入批量内容。
        active (bool): 控制台捕获是否当前活动。

    示例:
        文件日志（立即）:
        >>> logger = ConsoleLogger("training.log")
        >>> logger.start_capture()
        >>> print("这将被记录")
        >>> logger.stop_capture()

        带批量的 API 流式传输:
        >>> logger = ConsoleLogger("https://api.example.com/logs", batch_size=10)
        >>> logger.start_capture()

        带批量的自定义回调:
        >>> def my_handler(content, line_count, chunk_id):
        ...     print(f"收到 {line_count} 行")
        >>> logger = ConsoleLogger(on_flush=my_handler, batch_size=5)
        >>> logger.start_capture()
    """

    def __init__(self, destination=None, batch_size=1, flush_interval=5.0, on_flush=None):
        """初始化控制台日志器，可选批量处理。

        参数:
            destination (str | Path | None): API 端点 URL（http/https）、本地文件路径，或 None。
            batch_size (int): 刷新前累积的行数（1 = 立即，更高 = 批量）。
            flush_interval (float): 批量模式下刷新之间的最大秒数。
            on_flush (callable | None): 回调(content: str, line_count: int, chunk_id: int)用于自定义处理。
        """
        self.destination = destination
        self.is_api = isinstance(destination, str) and destination.startswith(("http://", "https://"))
        if destination is not None and not self.is_api:
            self.destination = Path(destination)

        # 批量配置
        self.batch_size = max(1, batch_size)
        self.flush_interval = flush_interval
        self.on_flush = on_flush

        # 控制台捕获状态
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.active = False
        self._log_handler = None  # 跟踪处理器以便清理

        # 批量缓冲区
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.flush_thread = None
        self.chunk_id = 0

        # 去重状态
        self.last_line = ""
        self.last_time = 0.0
        self.last_progress_line = ""  # 跟踪进程序列键以去重
        self.last_was_progress = False  # 跟踪上一行是否为进度条

    def start_capture(self):
        """开始捕获控制台输出并重定向 stdout/stderr。

        注意:
            在 DDP 训练中，仅在 rank 0/-1 上激活，以防止重复日志。
        """
        if self.active or RANK not in {-1, 0}:
            return

        self.active = True
        sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
        sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)

        # 挂载 Ultralytics 日志器
        try:
            self._log_handler = self._LogHandler(self._queue_log)
            logging.getLogger("ultralytics").addHandler(self._log_handler)
        except Exception:
            pass

        # 启动批量模式的后台刷新线程
        if self.batch_size > 1:
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()

    def stop_capture(self):
        """停止捕获控制台输出并刷新剩余缓冲区。"""
        if not self.active:
            return

        self.active = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # 移除日志处理器以防止内存泄漏
        if self._log_handler:
            try:
                logging.getLogger("ultralytics").removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None

        # 最终刷新
        self._flush_buffer()

    def _queue_log(self, text):
        """将控制台文本加入队列，进行去重和时间戳处理。"""
        if not self.active:
            return

        current_time = time.time()

        # 处理回车符并处理行
        if "\r" in text:
            text = text.split("\r")[-1]

        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()

        for line in lines:
            line = line.rstrip()

            # 跳过仅有细进度条的行（部分进度）
            if "─" in line:  # 有细线但无粗线
                continue

            # 仅显示进度条 100% 完成行
            if " ━━" in line:
                is_complete = "100%" in line

                # 跳过所有未完成的进度行
                if not is_complete:
                    continue

                # 提取序列键以去重同一序列的多个 100% 行
                parts = line.split()
                seq_key = ""
                if parts:
                    # 检查 epoch 模式（开头的 X/Y）
                    if "/" in parts[0] and parts[0].replace("/", "").isdigit():
                        seq_key = parts[0]  # 如 "1/3"
                    elif parts[0] == "Class" and len(parts) > 1:
                        seq_key = f"{parts[0]}_{parts[1]}"  # 如 "Class_train:" 或 "Class_val:"
                    elif parts[0] in ("train:", "val:"):
                        seq_key = parts[0]  # 阶段标识符

                # 如果已显示此序列的 100% 则跳过
                if seq_key and self.last_progress_line == f"{seq_key}:done":
                    continue

                # 标记此序列为完成
                if seq_key:
                    self.last_progress_line = f"{seq_key}:done"

                self.last_was_progress = True
            else:
                # 跳过进度条后的空行
                if not line and self.last_was_progress:
                    self.last_was_progress = False
                    continue
                self.last_was_progress = False

            # 一般去重
            if line == self.last_line and current_time - self.last_time < 0.1:
                continue

            self.last_line = line
            self.last_time = current_time

            # 如果需要则添加时间戳
            if not line.startswith("[20"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"[{timestamp}] {line}"

            # 添加到缓冲区并检查是否需要刷新
            should_flush = False
            with self.buffer_lock:
                self.buffer.append(line)
                if len(self.buffer) >= self.batch_size:
                    should_flush = True

            # 在锁外刷新以避免死锁
            if should_flush:
                self._flush_buffer()

    def _flush_worker(self):
        """定期刷新缓冲区的后台工作线程。"""
        while self.active:
            time.sleep(self.flush_interval)
            if self.active:
                self._flush_buffer()

    def _flush_buffer(self):
        """将缓冲行刷新到目标和/或回调。"""
        with self.buffer_lock:
            if not self.buffer:
                return
            lines = self.buffer.copy()
            self.buffer.clear()
            self.chunk_id += 1
            chunk_id = self.chunk_id  # 在锁内捕获以避免竞态

        content = "\n".join(lines)
        line_count = len(lines)

        # 如果提供了自定义回调则调用
        if self.on_flush:
            try:
                self.on_flush(content, line_count, chunk_id)
            except Exception:
                pass  # 静默忽略回调错误以避免淹没 stderr

        # 写入目标（文件或 API）
        if self.destination is not None:
            self._write_destination(content)

    def _write_destination(self, content):
        """将内容写入文件或 API 目标。"""
        try:
            if self.is_api:
                import requests

                payload = {"timestamp": datetime.now().isoformat(), "message": content}
                requests.post(str(self.destination), json=payload, timeout=5)
            else:
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                with self.destination.open("a", encoding="utf-8") as f:
                    f.write(content + "\n")
        except Exception as e:
            print(f"Console logger write error: {e}", file=self.original_stderr)

        """轻量级 stdout/stderr 捕获。"""

        __slots__ = ("callback", "original")

        def __init__(self, original, callback):
            """初始化流包装器，将写入重定向到回调，同时保留原始流。"""
            self.original = original
            self.callback = callback

        def write(self, text):
            """将文本写入原始流并转发到捕获回调。"""
            self.original.write(text)
            self.callback(text)

        def flush(self):
            """刷新包装的流，在控制台捕获期间及时传播缓冲输出。"""
            self.original.flush()

        def isatty(self):
            """将 isatty 检查委托给原始流。"""
            return self.original.isatty()

        """轻量级日志处理器。"""

        __slots__ = ("callback",)

        def __init__(self, callback):
            """初始化轻量级 logging.Handler，将日志记录转发到提供的回调。"""
            super().__init__()
            self.callback = callback

        def emit(self, record):
            """格式化并将 LogRecord 消息转发到捕获回调以实现统一日志流。"""
            self.callback(self.format(record) + "\n")


class SystemLogger:
    """用于训练监控的日志动态系统指标。

    捕获实时系统指标，包括 CPU、RAM、磁盘 I/O、网络 I/O 和 NVIDIA GPU 统计信息，
    用于训练性能监控和分析。

    属性:
        pynvml: 成功导入的 NVIDIA pynvml 模块实例，否则为 None。
        nvidia_initialized (bool): NVIDIA GPU 监控是否可用并已初始化。
        net_start: 用于计算累积使用量的初始网络 I/O 计数器。
        disk_start: 用于计算累积使用量的初始磁盘 I/O 计数器。

    示例:
        基本用法:
        >>> logger = SystemLogger()
        >>> metrics = logger.get_metrics()
        >>> print(f"CPU: {metrics['cpu']}%, RAM: {metrics['ram']}%")
        >>> if metrics["gpus"]:
        ...     gpu0 = metrics["gpus"]["0"]
        ...     print(f"GPU0: {gpu0['usage']}% 使用率, {gpu0['temp']}°C")

        训练循环集成:
        >>> system_logger = SystemLogger()
        >>> for epoch in range(epochs):
        ...     # 训练代码
        ...     metrics = system_logger.get_metrics()
        ...     # 记录到数据库/文件
    """

    def __init__(self):
        """初始化系统日志器。"""
        import psutil  # 限定作用域，因为导入较慢

        self.pynvml = None
        self.nvidia_initialized = self._init_nvidia()
        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()

        # 用于速率计算
        self._prev_net = self.net_start
        self._prev_disk = self.disk_start
        self._prev_time = time.time()

    def _init_nvidia(self):
        """使用 pynvml 初始化 NVIDIA GPU 监控。"""
        if MACOS:
            return False

        try:
            check_requirements("nvidia-ml-py>=12.0.0")
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            return True
        except Exception as e:
            import torch

            if torch.cuda.is_available():
                LOGGER.warning(f"SystemLogger NVML init failed: {e}")
            return False

    def get_metrics(self, rates=False):
        """获取当前系统指标，包括 CPU、RAM、磁盘、网络和 GPU 使用率。

        收集综合系统指标，包括 CPU 使用率、RAM 使用率、磁盘 I/O 统计、
        网络 I/O 统计和 GPU 指标（如果可用）。

        示例输出（rates=False，默认）:
        ```python
        {
            "cpu": 45.2,
            "ram": 78.9,
            "disk": {"read_mb": 156.7, "write_mb": 89.3, "used_gb": 256.8},
            "network": {"recv_mb": 157.2, "sent_mb": 89.1},
            "gpus": {
                "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
                "1": {"usage": 94.1, "memory": 82.7, "temp": 70, "power": 278},
            },
        }
        ```

        示例输出（rates=True）:
        ```python
        {
            "cpu": 45.2,
            "ram": 78.9,
            "disk": {"read_mbs": 12.5, "write_mbs": 8.3, "used_gb": 256.8},
            "network": {"recv_mbs": 5.2, "sent_mbs": 1.1},
            "gpus": {
                "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
            },
        }
        ```

        参数:
            rates (bool): 如果为 True，磁盘/网络返回 MB/s 速率而不是累积 MB。

        返回:
            (dict): 包含 cpu、ram、disk、network 和 gpus 键的指标字典。

        示例:
            >>> logger = SystemLogger()
            >>> logger.get_metrics()["cpu"]  # CPU 百分比
            >>> logger.get_metrics(rates=True)["network"]["recv_mbs"]  # MB/s 下载速率
        """
        import psutil  # 限定作用域，因为导入较慢

        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        memory = psutil.virtual_memory()
        disk_usage = shutil.disk_usage("/")
        now = time.time()

        metrics = {
            "cpu": round(psutil.cpu_percent(), 3),
            "ram": round(memory.percent, 3),
            "gpus": {},
        }

        # 计算自上次调用以来的时间
        elapsed = max(0.1, now - self._prev_time)  # 避免除零

        if rates:
            # 计算自上次调用以来的 MB/s 速率
            metrics["disk"] = {
                "read_mbs": round(max(0, (disk.read_bytes - self._prev_disk.read_bytes) / (1 << 20) / elapsed), 3),
                "write_mbs": round(max(0, (disk.write_bytes - self._prev_disk.write_bytes) / (1 << 20) / elapsed), 3),
                "used_gb": round(disk_usage.used / (1 << 30), 3),
            }
            metrics["network"] = {
                "recv_mbs": round(max(0, (net.bytes_recv - self._prev_net.bytes_recv) / (1 << 20) / elapsed), 3),
                "sent_mbs": round(max(0, (net.bytes_sent - self._prev_net.bytes_sent) / (1 << 20) / elapsed), 3),
            }
        else:
            # 初始化以来的累积 MB（原始行为）
            metrics["disk"] = {
                "read_mb": round((disk.read_bytes - self.disk_start.read_bytes) / (1 << 20), 3),
                "write_mb": round((disk.write_bytes - self.disk_start.write_bytes) / (1 << 20), 3),
                "used_gb": round(disk_usage.used / (1 << 30), 3),
            }
            metrics["network"] = {
                "recv_mb": round((net.bytes_recv - self.net_start.bytes_recv) / (1 << 20), 3),
                "sent_mb": round((net.bytes_sent - self.net_start.bytes_sent) / (1 << 20), 3),
            }

        # 始终更新前一个值，以便下次调用时准确计算速率
        self._prev_net = net
        self._prev_disk = disk
        self._prev_time = now

        # 添加 GPU 指标（仅 NVIDIA）
        if self.nvidia_initialized:
            metrics["gpus"].update(self._get_nvidia_metrics())

        return metrics

    def _get_nvidia_metrics(self):
        """获取 NVIDIA GPU 指标，包括利用率、内存、温度和功耗。"""
        gpus = {}
        if not self.nvidia_initialized or not self.pynvml:
            return gpus
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                power = self.pynvml.nvmlDeviceGetPowerUsage(handle) // 1000

                gpus[str(i)] = {
                    "usage": round(util.gpu, 3),
                    "memory": round((memory.used / memory.total) * 100, 3),
                    "temp": temp,
                    "power": power,
                }
        except Exception:
            pass
        return gpus


if __name__ == "__main__":
    print("SystemLogger Real-time Metrics Monitor")
    print("Press Ctrl+C to stop\n")

    logger = SystemLogger()

    try:
        while True:
            metrics = logger.get_metrics()

            # 清屏（适用于大多数终端）
            print("\033[H\033[J", end="")

            # 显示系统指标
            print(f"CPU: {metrics['cpu']:5.1f}%")
            print(f"RAM: {metrics['ram']:5.1f}%")
            print(f"Disk Read: {metrics['disk']['read_mb']:8.1f} MB")
            print(f"Disk Write: {metrics['disk']['write_mb']:7.1f} MB")
            print(f"Disk Used: {metrics['disk']['used_gb']:8.1f} GB")
            print(f"Net Recv: {metrics['network']['recv_mb']:9.1f} MB")
            print(f"Net Sent: {metrics['network']['sent_mb']:9.1f} MB")

            # 若可用则显示 GPU 指标
            if metrics["gpus"]:
                print("\nGPU Metrics:")
                for gpu_id, gpu_data in metrics["gpus"].items():
                    print(
                        f"  GPU {gpu_id}: {gpu_data['usage']:3}% | "
                        f"Mem: {gpu_data['memory']:5.1f}% | "
                        f"Temp: {gpu_data['temp']:2}°C | "
                        f"Power: {gpu_data['power']:3}W"
                    )
            else:
                print("\nGPU: No NVIDIA GPUs detected")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")
