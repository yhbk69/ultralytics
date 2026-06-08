# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class DeepXBackend(BaseBackend):
    """DeepX NPU 推理后端，用于 DeepX 硬件加速器。

    加载已编译的 DeepX 模型（.dxnn 文件），并使用 DeepX DX-Runtime 执行推理。
    """

    def load_model(self, weight: str | Path) -> None:
        """从包含 .dxnn 文件的目录中加载 DeepX 模型。

        Args:
            weight (str | Path): 包含 .dxnn 二进制文件的 DeepX 模型目录路径。

        Raises:
            ImportError: 若未安装 ``dx_engine`` Python 包，则抛出此异常。
            FileNotFoundError: 若在给定目录中未找到 .dxnn 文件，则抛出此异常。
        """
        try:
            from dx_engine import InferenceEngine
        except ImportError as e:
            raise ImportError(
                "DeepX inference requires the DeepX DX-Runtime and `dx_engine` Python package. "
                "See https://docs.ultralytics.com/integrations/deepx/#runtime-installation for installation instructions."
            ) from e

        LOGGER.info(f"Loading {weight} for DeepX inference...")

        w = Path(weight)
        # 递归查找目录中的 .dxnn 模型文件
        found = next(w.rglob("*.dxnn"), None)
        if found is None:
            raise FileNotFoundError(f"No .dxnn file found in: {w}")

        self.model = InferenceEngine(str(found))

        # 加载同目录下的 metadata.yaml 元数据
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """在 DeepX NPU 上执行推理。

        DeepX 运行时要求输入为 HWC uint8 [0, 255] 格式，
        因此需将 BCHW float [0, 1] 逐帧转换后逐帧推理，最后沿 batch 维度拼接输出。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (np.ndarray | list[np.ndarray]): 模型预测结果，单输出返回数组，多输出返回列表。
        """
        outputs = []
        for sample in im.cpu().numpy():
            # CHW float → HWC uint8：转置 + 反归一化 + 裁剪 + 类型转换，保证连续内存布局
            sample = np.ascontiguousarray(np.clip(np.transpose(sample, (1, 2, 0)) * 255, 0, 255).astype(np.uint8))
            for i, out in enumerate(map(np.asarray, self.model.run([sample]))):
                if i == len(outputs):
                    outputs.append([])
                # 若输出已有 batch 维度则直接追加，否则补充一维
                outputs[i].append(out if out.ndim and out.shape[0] == 1 else out[None])
        # 沿 batch 轴拼接各帧结果
        y = [np.concatenate(x, axis=0) for x in outputs]
        return y[0] if len(y) == 1 else y
