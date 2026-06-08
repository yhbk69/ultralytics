# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class PaddleBackend(BaseBackend):
    """百度飞桨 PaddlePaddle 推理后端。

    加载并运行百度飞桨模型（*_paddle_model/ 目录）。
    支持 CPU 和 GPU 执行，自动进行设备配置和显存池初始化。
    """

    def load_model(self, weight: str | Path) -> None:
        """从包含 .json 和 .pdiparams 文件的目录中加载飞桨模型。

        Args:
            weight (str | Path): 模型目录路径或 .pdiparams 文件路径。
        """
        cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"
        LOGGER.info(f"Loading {weight} for PaddlePaddle inference...")
        # 根据运行环境安装对应版本的 paddlepaddle
        if cuda:
            check_requirements("paddlepaddle-gpu>=3.0.0,<3.3.0")
        elif ARM64:
            check_requirements("paddlepaddle==3.0.0")
        else:
            check_requirements("paddlepaddle>=3.0.0,<3.3.0")

        import paddle.inference as pdi

        w = Path(weight)
        model_file, params_file = None, None

        if w.is_dir():
            # 目录模式：递归查找 .json 结构文件和 .pdiparams 权重文件
            model_file = next(w.rglob("*.json"), None)
            params_file = next(w.rglob("*.pdiparams"), None)
        elif w.suffix == ".pdiparams":
            # 直接传入权重文件：结构文件与其同名
            model_file = w.with_name("model.json")
            params_file = w

        if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
            raise FileNotFoundError(f"Paddle model not found in {w}. Both .json and .pdiparams files are required.")

        config = pdi.Config(str(model_file), str(params_file))
        if cuda:
            # 启用 GPU 推理并初始化显存池（2048 MB）
            config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=self.device.index or 0)

        self.predictor = pdi.create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output_names = self.predictor.get_output_names()

        # 加载 metadata.yaml 元数据
        metadata_file = (w if w.is_dir() else w.parent) / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """执行百度飞桨推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (list[np.ndarray]): 模型预测结果列表，每个元素对应一个输出句柄的 numpy 数组。
        """
        # 将张量转为 float32 numpy 数组后传入 CPU 输入句柄
        self.input_handle.copy_from_cpu(im.cpu().numpy().astype(np.float32))
        self.predictor.run()
        return [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
