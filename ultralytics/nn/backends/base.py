# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseBackend(ABC):
    """所有推理后端的基类。

    此抽象类定义了所有推理后端必须实现的接口，提供模型加载、元数据处理
    和设备管理的通用功能。

    Attributes:
        model: 底层推理模型或运行时会话。
        device (torch.device): 执行推理的设备。
        fp16 (bool): 是否使用 FP16（半精度）推理。
        nhwc (bool): 模型是否期望 NHWC 输入格式（而非 NCHW）。
        stride (int): 模型步长，YOLO 模型通常为 32。
        names (dict): 类别索引到类别名称的映射字典。
        task (str | None): 任务类型（detect/segment/classify/pose/obb）。
        batch (int): 推理批次大小。
        imgsz (tuple): 输入图像尺寸 (height, width)。
        channels (int): 输入通道数，RGB 图像通常为 3。
        end2end (bool): 模型是否包含端到端 NMS 后处理。
        dynamic (bool): 模型是否支持动态输入形状。
        metadata (dict): 模型元数据字典，包含导出时的配置信息。
    """

    def __init__(self, weight: str | torch.nn.Module, device: torch.device | str, fp16: bool = False):
        """初始化基础后端，设置通用属性并加载模型。

        Args:
            weight (str | torch.nn.Module): 模型权重文件路径或 PyTorch 模块实例。
            device (torch.device | str): 执行推理的设备（如 'cpu'、'cuda:0'）。
            fp16 (bool): 是否使用 FP16 半精度推理。
        """
        self.device = device
        self.fp16 = fp16
        self.nhwc = False
        self.stride = 32
        self.names = {}
        self.task = None
        self.batch = 1
        self.channels = 3
        self.end2end = False
        self.dynamic = False
        self.metadata = {}
        self.model = None
        self.load_model(weight)

    @abstractmethod
    def load_model(self, weight: str | torch.nn.Module) -> None:
        """从权重文件或模块实例中加载模型。

        Args:
            weight (str | torch.nn.Module): 模型权重路径或 PyTorch 模块。
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, im: torch.Tensor) -> Any:
        """对输入图像张量执行推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (Any): 模型前向传播的原始输出，可能需要后处理。
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        """允许直接调用后端实例执行推理，参数转发给 `forward` 方法。"""
        return self.forward(*args, **kwargs)

    def apply_metadata(self, metadata: dict | None) -> None:
        """处理并应用模型元数据到后端属性。

        对常用元数据字段（如 stride、batch、names）进行类型转换，
        并设为实例属性。同时从导出参数中解析端到端 NMS 和动态形状设置。

        Args:
            metadata (dict | None): 包含模型导出配置键值对的元数据字典。
        """
        if not metadata:
            return

        # 存储原始元数据
        self.metadata = metadata

        # 对已知字段进行类型转换
        for k, v in metadata.items():
            if k in {"stride", "batch", "channels"}:
                metadata[k] = int(v)
            elif k in {"imgsz", "names", "kpt_shape", "kpt_names", "args", "end2end"} and isinstance(v, str):
                metadata[k] = ast.literal_eval(v)

        # 处理带端到端 NMS 导出的模型
        metadata["end2end"] = metadata.get("end2end", False) or metadata.get("args", {}).get("nms", False)
        metadata["dynamic"] = metadata.get("args", {}).get("dynamic", self.dynamic)

        # 将所有元数据字段设为后端属性
        for k, v in metadata.items():
            setattr(self, k, v)
