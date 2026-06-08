# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.utils.checks import check_suffix
from ultralytics.utils.downloads import is_url

from .backends import (
    AxeleraBackend,
    CoreMLBackend,
    DeepXBackend,
    ExecuTorchBackend,
    MNNBackend,
    NCNNBackend,
    ONNXBackend,
    ONNXIMXBackend,
    OpenVINOBackend,
    PaddleBackend,
    PyTorchBackend,
    RKNNBackend,
    TensorFlowBackend,
    TensorRTBackend,
    TorchScriptBackend,
    TritonBackend,
)


def check_class_names(names: list | dict) -> dict[int, str]:
    """检查类名并在需要时转换为字典格式。

    Args:
        names (list | dict): 列表或字典格式的类名。

    Returns:
        (dict): 以整数为键、字符串为值的字典格式类名。

    Raises:
        KeyError: 如果类索引对数据集大小无效则抛出。
    """
    if isinstance(names, list):  # names 是列表
        names = dict(enumerate(names))  # 转换为字典
    if isinstance(names, dict):
        # 转换 1) 字符串键为整数，如 '0' -> 0，以及非字符串值为字符串，如 True -> 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet 类编码，如 'n01440764'
            from ultralytics.utils import ROOT, YAML

            names_map = YAML.load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]  # 人类可读的名称
            names = {k: names_map[v] for k, v in names.items()}
    return names


def default_class_names(data: str | Path | None = None) -> dict[int, str]:
    """从 YAML 文件加载类名或返回数字形式的类名。

    Args:
        data (str | Path, optional): 包含类名的 YAML 文件路径。

    Returns:
        (dict): 将类索引映射为类名的字典。
    """
    if data:
        try:
            from ultralytics.utils import YAML
            from ultralytics.utils.checks import check_yaml

            return YAML.load(check_yaml(data))["names"]
        except Exception:
            pass
    return {i: f"class{i}" for i in range(999)}  # 如果上述出错则返回默认值


class AutoBackend(nn.Module):
    """处理使用 Ultralytics YOLO 模型进行推理时后端动态选择的抽象层。

    AutoBackend 类旨在为各种推理引擎提供抽象层。它支持多种格式，每种格式都有如下特定的命名约定：

        支持的格式与命名约定：
            | 格式                  | 文件后缀            |
            | --------------------- | ----------------- |
            | PyTorch               | *.pt              |
            | TorchScript           | *.torchscript     |
            | ONNX Runtime          | *.onnx            |
            | ONNX OpenCV DNN       | *.onnx (dnn=True) |
            | OpenVINO              | *openvino_model/  |
            | CoreML                | *.mlpackage       |
            | TensorRT              | *.engine          |
            | TensorFlow SavedModel | *_saved_model/    |
            | TensorFlow GraphDef   | *.pb              |
            | TensorFlow Lite       | *.tflite          |
            | TensorFlow Edge TPU   | *_edgetpu.tflite  |
            | PaddlePaddle          | *_paddle_model/   |
            | MNN                   | *.mnn             |
            | NCNN                  | *_ncnn_model/     |
            | IMX                   | *_imx_model/      |
            | RKNN                  | *_rknn_model/     |
            | Triton Inference      | triton://model    |
            | ExecuTorch            | *.pte             |
            | Axelera AI            | *_axelera_model/  |
            | DeepX                 | *_deepx_model/    |

    Attributes:
        backend (BaseBackend): 加载的推理后端实例。
        format (str): 模型格式（如 'pt', 'onnx', 'engine'）。
        model: 底层模型（对 PyTorch 后端为 nn.Module，否则为后端实例）。
        device (torch.device): 模型加载所在的设备（CPU 或 GPU）。
        task (str): 模型执行的任务类型（detect, segment, classify, pose）。
        names (dict): 模型可检测的类名字典。
        stride (int): 模型步幅，YOLO 模型通常为 32。
        fp16 (bool): 模型是否使用半精度（FP16）推理。
        nhwc (bool): 模型是否期望 NHWC 输入格式而非 NCHW。

    Methods:
        forward: 对输入图像执行推理。
        from_numpy: 将 NumPy 数组转换为模型设备上的张量。
        warmup: 使用虚拟输入预热模型。
        _model_type: 通过文件路径确定模型类型。

    Examples:
        >>> model = AutoBackend(model="yolo26n.pt", device="cuda")
        >>> results = model(img)
    """

    _BACKEND_MAP = {
        "pt": PyTorchBackend,
        "torchscript": TorchScriptBackend,
        "onnx": ONNXBackend,
        "dnn": ONNXBackend,  # 特殊情况：使用 DNN 的 ONNX
        "openvino": OpenVINOBackend,
        "engine": TensorRTBackend,
        "coreml": CoreMLBackend,
        "saved_model": TensorFlowBackend,
        "pb": TensorFlowBackend,
        "tflite": TensorFlowBackend,
        "edgetpu": TensorFlowBackend,
        "paddle": PaddleBackend,
        "mnn": MNNBackend,
        "ncnn": NCNNBackend,
        "imx": ONNXIMXBackend,
        "rknn": RKNNBackend,
        "triton": TritonBackend,
        "executorch": ExecuTorchBackend,
        "axelera": AxeleraBackend,
        "deepx": DeepXBackend,
    }

    @torch.no_grad()
    def __init__(
        self,
        model: str | torch.nn.Module = "yolo26n.pt",
        device: torch.device = torch.device("cpu"),
        dnn: bool = False,
        data: str | Path | None = None,
        fp16: bool = False,
        fuse: bool = True,
        verbose: bool = True,
    ):
        """初始化 AutoBackend 用于推理。

        Args:
            model (str | torch.nn.Module): 模型权重文件路径或模块实例。
            device (torch.device): 运行模型的设备。
            dnn (bool): 使用 OpenCV DNN 模块进行 ONNX 推理。
            data (str | Path, optional): 包含类名的附加 data.yaml 文件路径。
            fp16 (bool): 启用半精度推理。仅特定后端支持。
            fuse (bool): 融合 Conv2D + BatchNorm 层以进行优化。
            verbose (bool): 启用详细日志。
        """
        super().__init__()
        # 通过路径/URL 确定模型格式
        format = "pt" if isinstance(model, nn.Module) else self._model_type(model, dnn)

        # 检查格式是否支持 FP16
        fp16 &= format in {"pt", "torchscript", "onnx", "openvino", "engine", "triton"}

        # 设置设备
        if (
            isinstance(device, torch.device)
            and torch.cuda.is_available()
            and device.type != "cpu"
            and format not in {"pt", "torchscript", "engine", "onnx", "paddle"}
        ):
            device = torch.device("cpu")

        # 选择并初始化合适的后端
        backend_kwargs = {"device": device, "fp16": fp16}

        if format == "tfjs":
            raise NotImplementedError("Ultralytics TF.js inference is not currently supported.")
        if format not in self._BACKEND_MAP:
            from ultralytics.engine.exporter import export_formats

            raise TypeError(
                f"model='{model}' is not a supported model format. "
                f"Ultralytics supports: {export_formats()['Format']}\n"
                f"See https://docs.ultralytics.com/modes/predict for help."
            )
        if format == "pt":
            backend_kwargs["fuse"] = fuse
            backend_kwargs["verbose"] = verbose
        elif format in {"saved_model", "pb", "tflite", "edgetpu", "dnn"}:
            backend_kwargs["format"] = format
        self.backend = self._BACKEND_MAP[format](model, **backend_kwargs)

        self.nhwc = format in {"coreml", "saved_model", "pb", "tflite", "edgetpu", "rknn"}
        self.format = format

        # 确保后端有类名（如果元数据未设置，则回退为默认值）
        if not self.backend.names:
            self.backend.names = default_class_names(data)
        self.backend.names = check_class_names(self.backend.names)

    def __getattr__(self, name: str) -> Any:
        """将属性访问委托给后端。

        这允许 AutoBackend 无需显式复制属性即可透明地暴露后端属性。

        Args:
            name: 要查找的属性名称。

        Returns:
            后端中的属性值。

        Raises:
            AttributeError: 如果在后端中找不到该属性则抛出。
        """
        if "backend" in self.__dict__ and hasattr(self.backend, name):
            return getattr(self.backend, name)
        return super().__getattr__(name)

    def forward(
        self,
        im: torch.Tensor,
        augment: bool = False,
        visualize: bool = False,
        embed: list | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor]:
        """在 AutoBackend 模型上运行推理。

        Args:
            im (torch.Tensor): 要进行推理的图像张量。
            augment (bool): 是否在推理期间执行数据增强。
            visualize (bool): 是否可视化输出预测结果。
            embed (list, optional): 要返回嵌入的层索引列表。
            **kwargs (Any): 模型配置的附加关键字参数。

        Returns:
            (torch.Tensor | list[torch.Tensor]): 模型的原始输出张量。
        """
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW 转为 numpy BHWC 形状 (1,320,192,3)
        if self.backend.fp16 and im.dtype != torch.float16:
            im = im.half()

        # 根据后端类型构建 forward 关键字参数
        forward_kwargs = {}
        if self.format == "pt":
            forward_kwargs = {"augment": augment, "visualize": visualize, "embed": embed, **kwargs}

        y = self.backend.forward(im, **forward_kwargs)

        if isinstance(y, (list, tuple)):
            if len(self.names) == 999 and (self.task == "segment" or len(y) == 2):  # 分割掩码和类别名未定义
                nc = y[0].shape[1] - y[1].shape[1] - 4  # y = (1, 116, 8400), (1, 32, 160, 160)
                self.names = {i: f"class{i}" for i in range(nc)}
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """将 NumPy 数组转换为模型设备上的 torch 张量。

        Args:
            x (np.ndarray | torch.Tensor): 输入数组或张量。

        Returns:
            (torch.Tensor): 位于 `self.device` 上的张量。
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz: tuple[int, int, int, int] = (1, 3, 640, 640)) -> None:
        """通过使用虚拟输入运行前向传播来预热模型。

        Args:
            imgsz (tuple[int, int, int, int]): 虚拟输入形状，格式为 (batch, channels, height, width)。
        """
        from ultralytics.utils.nms import non_max_suppression

        if self.format in {"pt", "torchscript", "onnx", "engine", "saved_model", "pb", "triton"} and (
            self.device.type != "cpu" or self.format == "triton"
        ):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # 输入
            for _ in range(2 if self.format == "torchscript" else 1):
                self.forward(im)  # 预热模型
                warmup_boxes = torch.rand(1, 84, 16, device=self.device)  # 经验表明 16 个框效果最佳
                warmup_boxes[:, :4] *= imgsz[-1]
                non_max_suppression(warmup_boxes)  # 预热 NMS

    @staticmethod
    def _model_type(p: str = "path/to/model.pt", dnn: bool = False) -> str:
        """接收模型文件路径并返回模型格式字符串。

        Args:
            p (str): 模型文件路径。
            dnn (bool): 是否使用 OpenCV DNN 模块进行 ONNX 推理。

        Returns:
            (str): 模型格式字符串（如 'pt', 'onnx', 'engine', 'triton'）。

        Examples:
            >>> fmt = AutoBackend._model_type("path/to/model.onnx")
            >>> assert fmt == "onnx"
        """
        from ultralytics.engine.exporter import export_formats

        sf = export_formats()["Suffix"]
        if not is_url(p) and not isinstance(p, str):
            check_suffix(p, sf)
        name = Path(p).name
        types = [s in name for s in sf]
        types[5] |= name.endswith(".mlmodel")
        types[8] &= not types[9]
        format = next((f for i, f in enumerate(export_formats()["Argument"]) if types[i]), None)
        if format == "-":
            format = "pt"
        elif format == "onnx" and dnn:
            format = "dnn"
        elif not any(types):
            from urllib.parse import urlsplit

            url = urlsplit(p)
            if bool(url.netloc) and bool(url.path) and url.scheme in {"http", "grpc"}:
                format = "triton"
        return format

    def eval(self) -> AutoBackend:
        """如果支持，将后端模型设置为评估模式。"""
        if hasattr(self.backend, "model") and hasattr(self.backend.model, "eval"):
            self.backend.model.eval()
        return super().eval()

    def _apply(self, fn) -> AutoBackend:
        """将函数应用到 backend.model 的参数、缓冲区和张量。

        此方法扩展了父类 _apply 方法的功能，额外重置预测器并更新模型配置中的设备信息。
        通常用于将模型移动到不同的设备或更改其精度。

        Args:
            fn (Callable): 要应用到模型张量的函数。通常是 to()、cpu()、cuda()、half() 或 float() 等方法。

        Returns:
            (AutoBackend): 应用了函数并更新了属性的模型实例。
        """
        self = super()._apply(fn)
        if hasattr(self.backend, "model") and isinstance(self.backend.model, nn.Module):
            self.backend.model._apply(fn)
            self.backend.device = next(self.backend.model.parameters()).device  # 移动后更新设备
        return self
