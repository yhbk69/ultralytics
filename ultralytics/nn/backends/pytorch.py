# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils import IS_JETSON, LOGGER, is_jetson

from .base import BaseBackend


class PyTorchBackend(BaseBackend):
    """原生 PyTorch 推理后端，用于直接执行 PyTorch 模型。

    加载并运行原生 PyTorch 模型（.pt 检查点文件）或预加载的 nn.Module 实例。
    支持模型层融合、FP16 精度及 NVIDIA Jetson 兼容性。
    """

    def __init__(
        self,
        weight: str | Path | nn.Module,
        device: torch.device,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """初始化 PyTorch 后端。

        Args:
            weight (str | Path | nn.Module): .pt 模型文件路径或预加载的 nn.Module 实例。
            device (torch.device): 执行推理的设备（如 'cpu'、'cuda:0'）。
            fp16 (bool): 是否使用 FP16 半精度推理。
            fuse (bool): 是否融合 Conv2D + BatchNorm 层以优化推理速度。
            verbose (bool): 是否打印详细的模型加载信息。
        """
        self.fuse = fuse
        self.verbose = verbose
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | torch.nn.Module) -> None:
        """从检查点文件或 nn.Module 实例中加载 PyTorch 模型。

        Args:
            weight (str | torch.nn.Module): .pt 检查点路径或预加载的模块。
        """
        from ultralytics.nn.tasks import load_checkpoint

        if isinstance(weight, torch.nn.Module):
            # 若传入已加载的模块，则直接融合并移至目标设备
            if self.fuse and hasattr(weight, "fuse"):
                if IS_JETSON and is_jetson(jetpack=5):
                    weight = weight.to(self.device)
                weight = weight.fuse(verbose=self.verbose)
            model = weight.to(self.device)
        else:
            # 从 .pt 文件加载检查点
            model, _ = load_checkpoint(weight, device=self.device, fuse=self.fuse)

        # 提取模型属性
        if hasattr(model, "kpt_shape"):
            self.kpt_shape = model.kpt_shape
        self.stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
        self.names = model.module.names if hasattr(model, "module") else getattr(model, "names", {})
        self.channels = model.yaml.get("channels", 3) if hasattr(model, "yaml") else 3
        # 按精度设置模型数据类型
        model.half() if self.fp16 else model.float()

        # 关闭参数梯度计算，节省推理内存
        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.end2end = getattr(model, "end2end", False)

    def forward(
        self, im: torch.Tensor, augment: bool = False, visualize: bool = False, embed: list | None = None, **kwargs: Any
    ) -> torch.Tensor | list[torch.Tensor]:
        """执行原生 PyTorch 推理，支持测试时增强、特征图可视化和嵌入提取。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。
            augment (bool): 是否应用测试时增强（TTA）。
            visualize (bool): 是否可视化中间特征图。
            embed (list | None): 需要提取嵌入向量的层索引列表，None 表示不提取。
            **kwargs (Any): 传递给模型 forward 方法的其他关键字参数。

        Returns:
            (torch.Tensor | list[torch.Tensor]): 模型预测结果张量。
        """
        return self.model(im, augment=augment, visualize=visualize, embed=embed, **kwargs)


class TorchScriptBackend(BaseBackend):
    """PyTorch TorchScript 推理后端，用于运行序列化模型。

    加载并运行通过 torch.jit.trace 或 torch.jit.script 创建的 TorchScript 模型
    （.torchscript 文件）。支持 FP16 精度和内嵌元数据提取。
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False):
        """初始化 TorchScript 后端。

        Args:
            weight (str | Path): .torchscript 模型文件路径。
            device (torch.device): 执行推理的设备（如 'cpu'、'cuda:0'）。
            fp16 (bool): 是否使用 FP16 半精度推理。
        """
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str) -> None:
        """从 .torchscript 文件加载 TorchScript 模型，并提取内嵌元数据。

        Args:
            weight (str): .torchscript 模型文件路径。
        """
        import json

        import torchvision  # noqa - TorchScript 模型反序列化时需要此库

        LOGGER.info(f"Loading {weight} for TorchScript inference...")
        extra_files = {"config.txt": ""}
        self.model = torch.jit.load(weight, _extra_files=extra_files, map_location=self.device)
        self.model.half() if self.fp16 else self.model.float()

        # 若模型内嵌了 config.txt，则解析其中的元数据
        if extra_files["config.txt"]:
            self.apply_metadata(json.loads(extra_files["config.txt"], object_hook=lambda x: dict(x.items())))

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """执行 TorchScript 推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (torch.Tensor | list[torch.Tensor]): 模型预测结果张量。
        """
        return self.model(im)
