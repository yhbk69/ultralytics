# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_executorch_requirements

from .base import BaseBackend


class ExecuTorchBackend(BaseBackend):
    """Meta ExecuTorch 推理后端，用于设备端部署。

    使用 ExecuTorch 运行时加载并运行 Meta ExecuTorch 模型（.pte 文件）。
    支持独立的 .pte 文件和包含元数据的目录式模型包。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .pte 文件或目录中加载 ExecuTorch 模型。

        Args:
            weight (str | Path): .pte 模型文件路径或包含模型的目录路径。
        """
        LOGGER.info(f"Loading {weight} for ExecuTorch inference...")
        check_executorch_requirements()

        from executorch.runtime import Runtime

        w = Path(weight)
        if w.is_dir():
            # 目录模式：递归查找 .pte 文件，并在同目录查找元数据
            model_file = next(w.rglob("*.pte"))
            metadata_file = w / "metadata.yaml"
        else:
            # 文件模式：直接使用 .pte 文件，在父目录查找元数据
            model_file = w
            metadata_file = w.parent / "metadata.yaml"

        # 加载程序并提取 forward 方法用于推理
        program = Runtime.get().load_program(str(model_file))
        self.model = program.load_method("forward")

        # 加载元数据（如存在）
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        """使用 ExecuTorch 运行时执行推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (list): ExecuTorch 模型输出值列表。
        """
        return self.model.execute([im])
