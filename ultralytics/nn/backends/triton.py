# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class TritonBackend(BaseBackend):
    """NVIDIA Triton 推理服务器后端，用于远程模型服务。

    通过 HTTP 或 gRPC 协议连接并运行托管在 NVIDIA Triton 推理服务器上的模型。
    模型通过 triton:// URL 方案指定。
    """

    def load_model(self, weight: str | Path) -> None:
        """连接到 NVIDIA Triton 推理服务器上的远程模型。

        Args:
            weight (str | Path): Triton 模型 URL（如 'triton://host:8000/model_name'）。
        """
        check_requirements("tritonclient[all]")
        from ultralytics.utils.triton import TritonRemoteModel

        self.model = TritonRemoteModel(weight)

        # 从 Triton 模型中复制元数据（如有）
        if hasattr(self.model, "metadata"):
            self.apply_metadata(self.model.metadata)

    def forward(self, im: torch.Tensor) -> list:
        """通过 NVIDIA Triton 推理服务器执行远程推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (list): 来自 Triton 服务器的模型预测结果列表（numpy 数组）。
        """
        return self.model(im.cpu().numpy())
