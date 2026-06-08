# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, LINUX, LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class OpenVINOBackend(BaseBackend):
    """英特尔 OpenVINO 推理后端，用于英特尔硬件加速。

    加载并运行英特尔 OpenVINO IR 模型（*_openvino_model/ 目录）。
    支持自动设备选择、英特尔特定设备目标以及异步推理以优化吞吐量。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 .xml/.bin 文件对或模型目录中加载英特尔 OpenVINO IR 模型。

        Args:
            weight (str | Path): .xml 文件路径或包含 OpenVINO 模型文件的目录路径。
        """
        LOGGER.info(f"Loading {weight} for OpenVINO inference...")
        check_requirements("openvino>=2024.0.0")
        import openvino as ov

        core = ov.Core()
        # 若只有 CPU 可用则默认使用 CPU，否则启用 AUTO 自动选择最优设备
        fallback_device = "CPU" if core.available_devices == ["CPU"] else "AUTO"
        device_name = fallback_device

        # 若用户指定了 intel:XXX 设备名，则提取并验证
        if isinstance(self.device, str) and self.device.startswith("intel"):
            device_name = self.device.split(":")[1].upper()
            self.device = torch.device("cpu")
            if device_name not in core.available_devices:
                LOGGER.warning(f"OpenVINO device '{device_name}' not available. Using '{fallback_device}' instead.")
                device_name = fallback_device

        w = Path(weight)
        # 若传入的是目录，则查找其中的 .xml 文件
        if not w.is_file():
            w = next(w.glob("*.xml"))

        ov_model = core.read_model(model=str(w), weights=w.with_suffix(".bin"))
        # 若模型未设置布局，手动设置为 NCHW
        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout("NCHW"))

        # 加载同目录下的 metadata.yaml 元数据
        metadata_file = w.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

        # 根据是否为动态形状及批次大小选择推理模式
        self.inference_mode = "CUMULATIVE_THROUGHPUT" if self.dynamic and self.batch > 1 else "LATENCY"
        config = {"PERFORMANCE_HINT": self.inference_mode}
        # ARM64 Linux 上强制使用 FP32 精度以保证准确性
        if LINUX and ARM64 and device_name == "CPU":
            config["EXECUTION_MODE_HINT"] = ov.properties.hint.ExecutionMode.ACCURACY
            config["INFERENCE_PRECISION_HINT"] = ov.Type.f32

        self.ov_compiled_model = core.compile_model(
            ov_model,
            device_name=device_name,
            config=config,
        )
        LOGGER.info(
            f"Using OpenVINO {self.inference_mode} mode for batch={self.batch} inference on "
            f"{', '.join(self.ov_compiled_model.get_property('EXECUTION_DEVICES'))}..."
        )
        self.input_name = self.ov_compiled_model.input().get_any_name()
        self.ov = ov

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """根据推理模式选择同步或异步方式执行英特尔 OpenVINO 推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (list[np.ndarray]): 模型预测结果列表，每个元素为一个输出层的 numpy 数组。
        """
        im = im.cpu().numpy().astype(np.float32)

        if self.inference_mode in {"THROUGHPUT", "CUMULATIVE_THROUGHPUT"}:
            # 异步推理：逐帧提交，并行处理大批次，提高吞吐量
            n = im.shape[0]
            results = [None] * n

            def callback(request, userdata):
                """将异步推理结果存入预分配的结果列表对应索引位置。"""
                results[userdata] = request.results

            async_queue = self.ov.AsyncInferQueue(self.ov_compiled_model)
            async_queue.set_callback(callback)

            for i in range(n):
                async_queue.start_async(inputs={self.input_name: im[i : i + 1]}, userdata=i)
            async_queue.wait_all()

            # 将各帧结果按输出通道合并
            y = [list(r.values()) for r in results]
            y = [np.concatenate(x) for x in zip(*y)]
        else:
            # 同步推理：LATENCY 模式，适合低延迟单帧推理
            y = list(self.ov_compiled_model(im).values())
        return y
