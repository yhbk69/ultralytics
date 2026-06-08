# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class NCNNBackend(BaseBackend):
    """腾讯 NCNN 推理后端，面向移动端和嵌入式部署。

    加载并运行腾讯 NCNN 模型（*_ncnn_model/ 目录）。
    针对移动平台优化，在可用时支持 Vulkan GPU 加速。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .param/.bin 文件对或模型目录中加载 NCNN 模型。

        Args:
            weight (str | Path): .param 文件路径或包含 NCNN 模型文件的目录路径。
        """
        LOGGER.info(f"Loading {weight} for NCNN inference...")
        check_requirements("ncnn", cmds="--no-deps")
        import ncnn as pyncnn

        self.pyncnn = pyncnn
        self.net = pyncnn.Net()

        # 若指定 vulkan 设备，则启用 Vulkan GPU 加速
        if isinstance(self.device, str) and self.device.startswith("vulkan"):
            self.net.opt.use_vulkan_compute = True
            self.net.set_vulkan_device(int(self.device.split(":")[1]))
            self.device = torch.device("cpu")
        else:
            self.net.opt.use_vulkan_compute = False

        w = Path(weight)
        # 若传入的是目录路径，则查找其中的 .param 文件
        if not w.is_file():
            w = next(w.glob("*.param"))

        # .param 为网络结构文件，.bin 为权重文件
        self.net.load_param(str(w))
        self.net.load_model(str(w.with_suffix(".bin")))

        # 加载同目录下的 metadata.yaml 元数据
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """使用 NCNN 运行时执行推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (list[np.ndarray]): 模型预测结果列表，每个元素为一个输出层的 numpy 数组。
        """
        # NCNN 只接受单张图像的 Mat 对象，取 batch 中的第一张
        mat_in = self.pyncnn.Mat(im[0].cpu().numpy())
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], mat_in)
            # 对输出名称排序以规避 pnnx 的输出顺序问题
            y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]
        return y
