# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class CoreMLBackend(BaseBackend):
    """苹果 CoreML 推理后端，用于 Apple 硬件加速。

    使用 coremltools 库加载并运行 CoreML 模型（.mlpackage 文件）。
    支持静态和动态输入形状，以及包含 NMS 的模型输出处理。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .mlpackage 文件加载 CoreML 模型。

        Args:
            weight (str | Path): .mlpackage 模型文件路径。
        """
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        LOGGER.info(f"Loading {weight} for CoreML inference...")
        self.model = ct.models.MLModel(weight)
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        # multiArrayType 表示动态多维数组输入（非固定尺寸图像类型）
        self.dynamic = spec.description.input[0].type.HasField("multiArrayType")

        # 从模型用户自定义元数据中加载配置
        self.apply_metadata(dict(self.model.user_defined_metadata))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """执行 CoreML 推理，自动处理输入格式转换。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BHWC（由 AutoBackend 从 BCHW 转换而来）。

        Returns:
            (np.ndarray | list[np.ndarray]): 模型预测结果 numpy 数组。
        """
        im = im.cpu().numpy()
        h, w = im.shape[1:3]

        # 动态输入：直接转置为 BCHW；静态输入：转为 PIL Image（CoreML 接口要求）
        im = im.transpose(0, 3, 1, 2) if self.dynamic else Image.fromarray((im[0] * 255).astype("uint8"))
        y = self.model.predict({self.input_name: im})
        if "confidence" in y:
            # 模型内置 NMS：将 xywh 坐标还原为 xyxy 并拼接置信度和类别
            from ultralytics.utils.ops import xywh2xyxy

            box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])
            cls = y["confidence"].argmax(1, keepdims=True)
            y = np.concatenate((box, np.take_along_axis(y["confidence"], cls, axis=1), cls), 1)[None]
        else:
            y = list(y.values())
        # 分割模型输出顺序修正：确保 (det, proto) 顺序正确
        if len(y) == 2 and len(y[1].shape) != 4:
            y = list(reversed(y))
        return y
