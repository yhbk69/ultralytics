# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements, is_rockchip

from .base import BaseBackend


class RKNNBackend(BaseBackend):
    """瑞芯微 RKNN 推理后端，专用于搭载 NPU 硬件的 Rockchip 设备。

    通过 RKNN-Toolkit-Lite2 运行时加载并执行 RKNN 模型（.rknn 文件）。
    仅支持带有 NPU 的 Rockchip 设备（如 RK3588、RK3566 等开发板/嵌入式平台）。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .rknn 文件或模型目录中加载瑞芯微 RKNN 模型。

        Args:
            weight (str | Path): .rknn 文件路径，或包含模型文件的目录路径。

        Raises:
            OSError: 若当前设备不是 Rockchip 设备，则抛出此异常。
            RuntimeError: 若模型加载或运行时初始化失败，则抛出此异常。
        """
        # 检查是否运行在 Rockchip 设备上，非 Rockchip 平台不支持 RKNN 推理
        if not is_rockchip():
            raise OSError("RKNN inference is only supported on Rockchip devices.")

        LOGGER.info(f"Loading {weight} for RKNN inference...")

        # 检查并安装 rknn-toolkit-lite2 依赖（轻量级运行时库）
        check_requirements("rknn-toolkit-lite2")
        from rknnlite.api import RKNNLite

        w = Path(weight)
        # 若传入的是目录路径，则递归查找其中第一个 .rknn 文件
        if not w.is_file():
            w = next(w.rglob("*.rknn"))

        # 初始化 RKNNLite 实例并加载模型文件
        self.model = RKNNLite()
        ret = self.model.load_rknn(str(w))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        # 初始化 NPU 运行时环境（绑定硬件驱动）
        ret = self.model.init_runtime()
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {ret}")

        # 加载同目录下的 metadata.yaml，写入模型元信息（类别名、输入尺寸等）
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        """在 Rockchip NPU 上执行前向推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]（float）。

        Returns:
            (list): 模型推理结果列表，每个元素对应一个输出头的 numpy 数组。
        """
        # RKNN 运行时要求输入为 uint8 类型（0~255），将 float [0,1] 反归一化并转换
        im = (im.cpu().numpy() * 255).astype("uint8")

        # RKNN inference 接口要求输入为列表格式，若已是列表/元组则直接传入
        im = im if isinstance(im, (list, tuple)) else [im]
        return self.model.inference(inputs=im)
