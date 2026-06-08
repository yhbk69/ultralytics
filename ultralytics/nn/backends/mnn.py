# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class MNNBackend(BaseBackend):
    """MNN（Mobile Neural Network）推理后端。

    使用阿里巴巴 MNN 框架加载并运行 MNN 模型（.mnn 文件）。
    针对移动端和边缘设备部署优化，支持可配置的线程数和推理精度。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .mnn 文件加载阿里巴巴 MNN 模型。

        Args:
            weight (str | Path): .mnn 模型文件路径。
        """
        LOGGER.info(f"Loading {weight} for MNN inference...")
        check_requirements("MNN")
        import MNN

        # 使用低精度 CPU 模式，线程数取 CPU 核数的一半（平衡性能与功耗）
        config = {"precision": "low", "backend": "CPU", "numThread": (os.cpu_count() + 1) // 2}
        rt = MNN.nn.create_runtime_manager((config,))
        self.net = MNN.nn.load_module_from_file(weight, [], [], runtime_manager=rt, rearrange=True)
        self.expr = MNN.expr

        # 从模型的 bizCode 字段中读取元数据（JSON 格式）
        info = self.net.get_info()
        if "bizCode" in info:
            try:
                self.apply_metadata(json.loads(info["bizCode"]))
            except json.JSONDecodeError:
                pass

    def forward(self, im: torch.Tensor) -> list:
        """使用 MNN 运行时执行推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (list): 模型预测结果列表，每个元素为 numpy 数组。
        """
        input_var = self.expr.const(im.data_ptr(), im.shape)
        output_var = self.net.onForward([input_var])
        # 注意：必须调用 copy()，否则在 ARM 设备上会得到错误结果
        return [x.read().copy() for x in output_var]
