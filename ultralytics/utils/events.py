# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import random
import time
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

from ultralytics import SETTINGS, __version__
from ultralytics.utils import ARGV, ENVIRONMENT, GIT, IS_PIP_PACKAGE, ONLINE, PYTHON_VERSION, RANK, TESTS_RUNNING
from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
from ultralytics.utils.torch_utils import get_cpu_info


def _post(url: str, data: dict, timeout: float = 5.0) -> None:
    """发送一次性 JSON POST 请求。"""
    try:
        body = json.dumps(data, separators=(",", ":")).encode()  # 紧凑 JSON
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        urlopen(req, timeout=timeout).close()
    except Exception:
        pass


class Events:
    """收集和发送匿名使用分析数据，具有速率限制。

    当设置中启用了同步、当前进程的 rank 为 -1 或 0、测试未运行、
    环境在线、安装来源为 pip 或官方 Ultralytics GitHub 仓库时，
    事件收集和传输才会启用。

    属性:
        url (str): 用于接收匿名事件的 Measurement Protocol 端点。
        events (list[dict]): 等待传输的事件负载内存队列。
        rate_limit (float): POST 请求之间的最短时间间隔（秒）。
        t (float): 上次传输的时间戳（自纪元以来的秒数）。
        metadata (dict): 描述运行时、安装来源和环境的静态元数据。
        enabled (bool): 指示分析收集是否激活的标志。

    方法:
        __init__: 初始化事件队列、速率限制器和运行时元数据。
        __call__: 将事件入队并在速率限制到期时触发非阻塞发送。
    """

    url = "https://www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&api_secret=QLQrATrNSwGRFRLE-cbHJw"

    def __init__(self) -> None:
        """初始化 Events 实例，包括队列、速率限制器和环境元数据。"""
        self.events = []  # 待发送的事件
        self.rate_limit = 30.0  # 速率限制（秒）
        self.t = 0.0  # 上次发送时间戳（秒）
        self.metadata = {
            "cli": Path(ARGV[0]).name == "yolo",
            "install": "git" if GIT.is_repo else "pip" if IS_PIP_PACKAGE else "other",
            "python": PYTHON_VERSION.rsplit(".", 1)[0],  # 即 3.13
            "CPU": get_cpu_info(),
            # "GPU": get_gpu_info(index=0) if cuda else None,
            "version": __version__,
            "env": ENVIRONMENT,
            "session_id": round(random.random() * 1e15),
            "engagement_time_msec": 1000,
        }
        self.enabled = (
            SETTINGS["sync"]
            and RANK in {-1, 0}
            and not TESTS_RUNNING
            and ONLINE
            and (IS_PIP_PACKAGE or GIT.origin == "https://github.com/ultralytics/ultralytics.git")
        )

    def __call__(self, cfg, device=None, backend=None) -> None:
        """将事件入队，当速率限制到期时异步刷新队列。

        参数:
            cfg (IterableSimpleNamespace): 包含模式和任务信息的配置对象。
            device (torch.device | str, 可选): 设备类型（如 'cpu', 'cuda'）。
            backend (object | None, 可选): 预测期间使用的推理后端实例。
        """
        if not self.enabled:
            # 事件已禁用，不做任何操作
            return

        # 尝试将新事件入队
        if len(self.events) < 25:  # 队列限制为 25 个事件，以限制内存和流量
            params = {
                **self.metadata,
                "task": cfg.task,
                "model": cfg.model if cfg.model in GITHUB_ASSETS_NAMES else "custom",
                "device": str(device),
            }
            if cfg.mode == "export":
                params["format"] = cfg.format
            if cfg.mode == "predict":
                params["backend"] = type(backend).__name__ if backend is not None else None
            self.events.append({"name": cfg.mode, "params": params})

        # 检查速率限制，如果未到限制时间则提前返回
        t = time.time()
        if (t - self.t) < self.rate_limit:
            return

        # 超过速率限制：在后台线程中发送队列事件的快照
        payload_events = list(self.events)  # 快照以避免与队列重置的竞态条件
        Thread(
            target=_post,
            args=(self.url, {"client_id": SETTINGS["uuid"], "events": payload_events}),  # SHA-256 匿名化
            daemon=True,
        ).start()

        # 重置队列和速率限制计时器
        self.events = []
        self.t = t


events = Events()
