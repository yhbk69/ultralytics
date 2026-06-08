# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class ONNXBackend(BaseBackend):
    """微软 ONNX Runtime 推理后端，可选 OpenCV DNN 支持。

    使用微软 ONNX Runtime（支持 CUDA/CoreML 执行提供者）或 OpenCV DNN（轻量级 CPU 推理）
    加载并运行 ONNX 模型（.onnx 文件）。支持 IO Binding 以在静态输入形状下优化 GPU 推理。
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False, format: str = "onnx"):
        """初始化 ONNX 后端。

        Args:
            weight (str | Path): .onnx 模型文件路径。
            device (torch.device): 执行推理的设备。
            fp16 (bool): 是否使用 FP16 半精度推理。
            format (str): 推理引擎，"onnx" 表示 ONNX Runtime，"dnn" 表示 OpenCV DNN。
        """
        assert format in {"onnx", "dnn"}, f"Unsupported ONNX format: {format}."
        self.format = format
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """使用 ONNX Runtime 或 OpenCV DNN 加载 ONNX 模型。

        Args:
            weight (str | Path): .onnx 模型文件路径。
        """
        cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"

        if self.format == "dnn":
            # 使用 OpenCV DNN 加载模型（轻量 CPU 后端）
            LOGGER.info(f"Loading {weight} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            import cv2

            self.net = cv2.dnn.readNetFromONNX(weight)
        else:
            # 使用 ONNX Runtime 加载模型
            LOGGER.info(f"Loading {weight} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            # 选择执行提供者（优先 CUDA，其次 CoreML，最后 CPU）
            available = onnxruntime.get_available_providers()
            if cuda and "CUDAExecutionProvider" in available:
                providers = [("CUDAExecutionProvider", {"device_id": self.device.index}), "CPUExecutionProvider"]
            elif self.device.type == "mps" and "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                if cuda:
                    LOGGER.warning("CUDA requested but CUDAExecutionProvider not available. Using CPU...")
                    self.device = torch.device("cpu")
                    cuda = False

            LOGGER.info(
                f"Using ONNX Runtime {onnxruntime.__version__} with "
                f"{providers[0] if isinstance(providers[0], str) else providers[0][0]}"
            )

            self.session = onnxruntime.InferenceSession(weight, providers=providers)
            self.output_names = [x.name for x in self.session.get_outputs()]

            # 从模型自定义元数据中加载配置信息
            metadata_map = self.session.get_modelmeta().custom_metadata_map
            if metadata_map:
                self.apply_metadata(dict(metadata_map))

            # 检查是否为动态输入形状（batch 维度为字符串时表示动态）
            self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
            self.fp16 = "float16" in self.session.get_inputs()[0].type

            # 仅在 CUDA 且静态形状时启用 IO Binding 以提升 GPU 推理效率
            self.use_io_binding = not self.dynamic and cuda
            if self.use_io_binding:
                self.io = self.session.io_binding()
                self.bindings = []
                for output in self.session.get_outputs():
                    out_fp16 = "float16" in output.type
                    # 预分配 GPU 输出缓冲区，避免每次推理重新分配内存
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if out_fp16 else torch.float32).to(
                        self.device
                    )
                    self.io.bind_output(
                        name=output.name,
                        device_type=self.device.type,
                        device_id=self.device.index if cuda else 0,
                        element_type=np.float16 if out_fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    self.bindings.append(y_tensor)

    def forward(self, im: torch.Tensor) -> torch.Tensor | list[torch.Tensor] | np.ndarray:
        """使用 IO Binding（CUDA）或标准 Session 执行 ONNX 推理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (torch.Tensor | list[torch.Tensor] | np.ndarray): 模型预测结果张量或数组。
        """
        if self.format == "dnn":
            # OpenCV DNN 推理路径
            self.net.setInput(im.cpu().numpy())
            return self.net.forward()

        # ONNX Runtime 推理路径
        if self.use_io_binding:
            # IO Binding 路径：直接绑定 GPU 内存，避免数据拷贝
            if self.device.type == "cpu":
                im = im.cpu()
            self.io.bind_input(
                name="images",
                device_type=im.device.type,
                device_id=im.device.index if im.device.type == "cuda" else 0,
                element_type=np.float16 if self.fp16 else np.float32,
                shape=tuple(im.shape),
                buffer_ptr=im.data_ptr(),
            )
            self.session.run_with_iobinding(self.io)
            return self.bindings
        else:
            # 标准路径：将张量转为 numpy 后传入 Session
            return self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})


class ONNXIMXBackend(ONNXBackend):
    """面向 NXP i.MX 处理器的 ONNX IMX 推理后端。

    继承 `ONNXBackend`，增加对量化模型的支持，针对 NXP i.MX 边缘设备进行优化。
    使用 MCT（模型压缩工具包）量化器和自定义 NMS 操作实现高效推理。
    """

    def load_model(self, weight: str | Path) -> None:
        """从 IMX 模型目录中加载量化 ONNX 模型。

        Args:
            weight (str | Path): 包含 .onnx 文件的 IMX 模型目录路径。
        """
        check_requirements(("model-compression-toolkit>=2.4.1", "edge-mdt-cl<1.1.0", "onnxruntime-extensions"))
        check_requirements(("onnx", "onnxruntime"))
        import mct_quantizers as mctq
        import onnxruntime
        from edgemdt_cl.pytorch.nms import nms_ort  # noqa - 注册自定义 NMS 算子

        w = Path(weight)
        onnx_file = next(w.glob("*.onnx"))
        LOGGER.info(f"Loading {onnx_file} for ONNX IMX inference...")

        # 获取 MCT 量化器的 Session 配置，并禁用内存复用（量化模型要求）
        session_options = mctq.get_ort_session_options()
        session_options.enable_mem_reuse = False

        self.session = onnxruntime.InferenceSession(onnx_file, session_options, providers=["CPUExecutionProvider"])
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.session.get_inputs()[0].type
        metadata_map = self.session.get_modelmeta().custom_metadata_map
        if metadata_map:
            self.apply_metadata(dict(metadata_map))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]:
        """执行 IMX 推理，并对不同任务类型拼接输出。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BCHW，值域为 [0, 1]。

        Returns:
            (np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]): 按任务格式整理的模型预测结果。
        """
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im.cpu().numpy()})

        if self.task == "detect":
            # 检测任务：拼接 boxes、置信度、类别
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None]], axis=-1)
        elif self.task == "pose":
            # 关键点任务：拼接 boxes、置信度、关键点
            return np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype)
        elif self.task == "segment":
            # 分割任务：返回 (检测结果拼接, 原型掩码)
            return (
                np.concatenate([y[0], y[1][:, :, None], y[2][:, :, None], y[3]], axis=-1, dtype=y[0].dtype),
                y[4],
            )
        return y
