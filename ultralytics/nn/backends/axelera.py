# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class AxeleraBackend(BaseBackend):
    """Axelera AI 推理后端，用于 Axelera Metis AI 加速器。

    加载已编译的 Axelera 模型（.axm 文件），并使用 Axelera AI 运行时 SDK 执行推理。
    """

    def load_model(self, weight: str | Path) -> None:
        """从包含 .axm 文件的目录中加载 Axelera 模型。

        Args:
            weight (str | Path): 包含 .axm 二进制文件的 Axelera 模型目录路径。
        """
        try:
            from axelera.runtime import op
        except ImportError:
            # 自动从 Axelera 私有 PyPI 源安装运行时
            check_requirements(
                "axelera-rt==1.6.0",
                cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
            )

        from axelera.runtime import op

        w = Path(weight)
        # 递归查找目录中的 .axm 模型文件
        found = next(w.rglob("*.axm"), None)
        if found is None:
            raise FileNotFoundError(f"No .axm file found in: {w}")

        # 加载并优化模型（optimized() 触发硬件特定编译优化）
        self.model = op.load(str(found)).optimized()

        # 加载同目录下的 metadata.yaml 元数据
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> list:
        """在 Axelera 硬件加速器上执行推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (list): 模型预测结果列表。
        """
        return self.model(im.cpu())
