# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics YOLO 推理后端包。

本包提供面向多种深度学习框架和硬件加速器的模块化推理后端。
每个后端均实现 `BaseBackend` 接口，可单独使用，也可通过统一的
`AutoBackend` 分发器进行格式自动检测与推理路由。
"""

from .axelera import AxeleraBackend
from .base import BaseBackend
from .coreml import CoreMLBackend
from .deepx import DeepXBackend
from .executorch import ExecuTorchBackend
from .mnn import MNNBackend
from .ncnn import NCNNBackend
from .onnx import ONNXBackend, ONNXIMXBackend
from .openvino import OpenVINOBackend
from .paddle import PaddleBackend
from .pytorch import PyTorchBackend, TorchScriptBackend
from .rknn import RKNNBackend
from .tensorflow import TensorFlowBackend
from .tensorrt import TensorRTBackend
from .triton import TritonBackend

__all__ = [
    "AxeleraBackend",
    "BaseBackend",
    "CoreMLBackend",
    "DeepXBackend",
    "ExecuTorchBackend",
    "MNNBackend",
    "NCNNBackend",
    "ONNXBackend",
    "ONNXIMXBackend",
    "OpenVINOBackend",
    "PaddleBackend",
    "PyTorchBackend",
    "RKNNBackend",
    "TensorFlowBackend",
    "TensorRTBackend",
    "TorchScriptBackend",
    "TritonBackend",
]
