# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
将 YOLO PyTorch 模型导出为其他格式。TensorFlow 导出由 https://github.com/zldrobit 编写。

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo26n.pt
TorchScript             | `torchscript`             | yolo26n.torchscript
ONNX                    | `onnx`                    | yolo26n.onnx
OpenVINO                | `openvino`                | yolo26n_openvino_model/
TensorRT                | `engine`                  | yolo26n.engine
CoreML                  | `coreml`                  | yolo26n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo26n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo26n.pb
TensorFlow Lite         | `tflite`                  | yolo26n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolo26n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolo26n_web_model/
PaddlePaddle            | `paddle`                  | yolo26n_paddle_model/
MNN                     | `mnn`                     | yolo26n.mnn
NCNN                    | `ncnn`                    | yolo26n_ncnn_model/
IMX                     | `imx`                     | yolo26n_imx_model/
RKNN                    | `rknn`                    | yolo26n_rknn_model/
ExecuTorch              | `executorch`              | yolo26n_executorch_model/
Axelera AI              | `axelera`                 | yolo26n_axelera_model/
DeepX                   | `deepx`                   | yolo26n_deepx_model/

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolo26n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolo26n.pt format=onnx

Inference:
    $ yolo predict model=yolo26n.pt                 # PyTorch
                         yolo26n.torchscript        # TorchScript
                         yolo26n.onnx               # ONNX Runtime 或带 dnn=True 的 OpenCV DNN
                         yolo26n_openvino_model     # OpenVINO
                         yolo26n.engine             # TensorRT
                         yolo26n.mlpackage          # CoreML (仅限 macOS)
                         yolo26n_saved_model        # TensorFlow SavedModel
                         yolo26n.pb                 # TensorFlow GraphDef
                         yolo26n.tflite             # TensorFlow Lite
                         yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo26n_paddle_model       # PaddlePaddle
                         yolo26n.mnn                # MNN
                         yolo26n_ncnn_model         # NCNN
                         yolo26n_imx_model          # IMX
                         yolo26n_rknn_model         # RKNN
                         yolo26n_executorch_model   # ExecuTorch
                         yolo26n_axelera_model      # Axelera AI
                         yolo26n_deepx_model        # DeepX

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolo26n_web_model public/yolo26n_web_model
    $ npm start
"""

from __future__ import annotations

import json
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ultralytics import __version__
from ultralytics.cfg import TASK2CALIBRATIONDATA, TASK2DATA, get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Classify, Detect, RTDETRDecoder, Segment26
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, WorldModel
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_DOCKER,
    LINUX,
    LOGGER,
    MACOS,
    MACOS_VERSION,
    RKNN_CHIPS,
    SETTINGS,
    TORCH_VERSION,
    WINDOWS,
    YAML,
    callbacks,
    colorstr,
    get_default_args,
    is_jetson,
)
from ultralytics.utils.checks import IS_PYTHON_MINIMUM_3_9, check_imgsz, check_requirements, check_version, is_intel
from ultralytics.utils.files import file_size
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import Profile
from ultralytics.utils.patches import arange_patch
from ultralytics.utils.torch_utils import (
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_1,
    TORCH_2_3,
    TORCH_2_8,
    TORCH_2_9,
    select_device,
)


def export_formats():
    """返回 Ultralytics YOLO 导出格式的字典。"""
    x = [
        ["PyTorch", "-", ".pt", True, True, []],
        ["TorchScript", "torchscript", ".torchscript", True, True, ["batch", "optimize", "half", "nms", "dynamic"]],
        ["ONNX", "onnx", ".onnx", True, True, ["batch", "dynamic", "half", "opset", "simplify", "nms"]],
        [
            "OpenVINO",
            "openvino",
            "_openvino_model",
            True,
            False,
            ["batch", "data", "dynamic", "half", "int8", "nms", "fraction"],
        ],
        [
            "TensorRT",
            "engine",
            ".engine",
            False,
            True,
            ["batch", "data", "dynamic", "half", "int8", "simplify", "nms", "fraction"],
        ],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "dynamic", "half", "int8", "nms"]],
        [
            "TensorFlow SavedModel",
            "saved_model",
            "_saved_model",
            True,
            True,
            ["batch", "data", "fraction", "int8", "keras", "nms"],
        ],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"]],
        ["TensorFlow Lite", "tflite", ".tflite", True, False, ["batch", "data", "half", "int8", "nms", "fraction"]],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False, ["data", "fraction", "int8"]],
        ["TensorFlow.js", "tfjs", "_web_model", True, False, ["batch", "data", "fraction", "half", "int8", "nms"]],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"]],
        ["MNN", "mnn", ".mnn", True, True, ["batch", "half", "int8"]],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "half"]],
        ["IMX", "imx", "_imx_model", True, True, ["data", "int8", "fraction", "nms"]],
        ["RKNN", "rknn", "_rknn_model", False, False, ["batch", "name"]],
        ["ExecuTorch", "executorch", "_executorch_model", True, False, ["batch"]],
        ["Axelera AI", "axelera", "_axelera_model", False, False, ["batch", "int8", "fraction", "data"]],
        ["DeepX", "deepx", "_deepx_model", False, False, ["data", "int8", "optimize"]],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))


def validate_args(format, passed_args, valid_args):
    """根据导出格式验证参数。

    Args:
        format (str): 导出格式。
        passed_args (SimpleNamespace): 导出期间使用的参数。
        valid_args (list): 该格式的有效参数列表。

    Raises:
        AssertionError: 如果使用了不支持的参数，或者该格式缺少支持的参数列表。
    """
    export_args = ["half", "int8", "dynamic", "keras", "nms", "batch", "fraction", "data"]

    assert valid_args is not None, f"ERROR ❌️ valid arguments for '{format}' not listed."
    custom = {"batch": 1, "data": None, "device": None}  # 导出器默认值
    default_args = get_cfg(DEFAULT_CFG, custom)
    for arg in export_args:
        not_default = getattr(passed_args, arg, None) != getattr(default_args, arg, None)
        if not_default:
            assert arg in valid_args, f"ERROR ❌️ argument '{arg}' is not supported for format='{format}'"


def try_export(inner_func):
    """YOLO 导出装饰器，即 @try_export。"""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """导出一个模型。"""
        prefix = inner_args["prefix"]
        dt = 0.0
        try:
            with Profile() as dt:
                f = inner_func(*args, **kwargs)  # 导出的文件/目录 或 (文件/目录, *) 元组
            path = f if isinstance(f, (str, Path)) else f[0]
            mb = file_size(path)
            assert mb > 0.0, "0.0 MB output model size"
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{path}' ({mb:.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"{prefix} export failure {dt.t:.1f}s: {e}")
            raise e

    return outer_func


class Exporter:
    """用于将 YOLO 模型导出为各种格式的类。

    此类提供将 YOLO 模型导出为不同格式的功能，包括 ONNX、TensorRT、CoreML、
    TensorFlow 等。它处理格式验证、设备选择、模型准备以及每种支持格式的实际导出过程。

    Attributes:
        args (SimpleNamespace): 导出器的配置参数。
        callbacks (dict): 不同导出事件的回调函数字典。
        im (torch.Tensor): 导出期间用于模型推理的输入张量。
        model (torch.nn.Module): 要导出的 YOLO 模型。
        file (Path): 正在导出的模型文件路径。
        output_shape (tuple): 模型输出张量的形状。
        pretty_name (str): 用于显示的格式化模型名称。
        metadata (dict): 模型元数据，包括描述、作者、版本等。
        device (torch.device): 模型加载到的设备。
        imgsz (list): 模型的输入图像尺寸。

    Methods:
        __call__: 处理导出过程的主要导出方法。
        get_int8_calibration_dataloader: 构建 INT8 校准的数据加载器。
        export_torchscript: 将模型导出为 TorchScript 格式。
        export_onnx: 将模型导出为 ONNX 格式。
        export_openvino: 将模型导出为 OpenVINO 格式。
        export_paddle: 将模型导出为 PaddlePaddle 格式。
        export_mnn: 将模型导出为 MNN 格式。
        export_ncnn: 将模型导出为 NCNN 格式。
        export_coreml: 将模型导出为 CoreML 格式。
        export_engine: 将模型导出为 TensorRT 格式。
        export_saved_model: 将模型导出为 TensorFlow SavedModel 格式。
        export_pb: 将模型导出为 TensorFlow GraphDef 格式。
        export_tflite: 将模型导出为 TensorFlow Lite 格式。
        export_edgetpu: 将模型导出为 Edge TPU 格式。
        export_tfjs: 将模型导出为 TensorFlow.js 格式。
        export_rknn: 将模型导出为 RKNN 格式。
        export_imx: 将模型导出为 IMX 格式。
        export_executorch: 将模型导出为 ExecuTorch 格式。
        export_axelera: 将模型导出为 Axelera 格式。
        export_deepx: 将模型导出为 DeepX 格式。

    Examples:
        将 YOLO26 模型导出为 ONNX 格式
        >>> from ultralytics.engine.exporter import Exporter
        >>> exporter = Exporter()
        >>> exporter(model="yolo26n.pt")  # 导出到 yolo26n.onnx

        使用特定参数导出
        >>> args = {"format": "onnx", "dynamic": True, "half": True}
        >>> exporter = Exporter(overrides=args)
        >>> exporter(model="yolo26n.pt")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """初始化 Exporter 类。

        Args:
            cfg (str | Path | dict | SimpleNamespace, optional): 配置文件路径或配置对象。
            overrides (dict, optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        """导出一个模型并返回最终的导出路径字符串。

        Returns:
            (str): 导出文件或目录的路径（最后的导出产物）。
        """
        t = time.time()
        fmt = self.args.format.lower()  # 转小写
        if fmt in {"tensorrt", "trt"}:  # 'engine' 的别名
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' 的别名
            fmt = "coreml"
        fmts_dict = export_formats()
        fmts = tuple(fmts_dict["Argument"][1:])  # 可用的导出格式
        if fmt not in fmts:
            import difflib

            # 如果格式无效，获取最接近的匹配
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 需要 60% 相似度才能匹配
            if not matches:
                msg = "Model is already in PyTorch format." if fmt == "pt" else f"Invalid export format='{fmt}'."
                raise ValueError(f"{msg} Valid formats are {fmts}")
            LOGGER.warning(f"Invalid export format='{fmt}', updating to format='{matches[0]}'")
            fmt = matches[0]
        is_tf_format = fmt in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}

        # 设备
        self.dla = None
        if fmt == "engine" and self.args.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if fmt == "engine" and "dla" in str(self.args.device):  # 先将 int/list 转换为 str
            device_str = str(self.args.device)
            self.dla = device_str.rsplit(":", 1)[-1]
            self.args.device = "0"  # 更新 device 为 "0"
            assert self.dla in {"0", "1"}, f"Expected device 'dla:0' or 'dla:1', but got {device_str}."
        if fmt == "imx" and self.args.device is None and torch.cuda.is_available():
            LOGGER.warning("Exporting on CPU while CUDA is available, setting device=0 for faster export on GPU.")
            self.args.device = "0"  # 更新 device 为 "0"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # 参数兼容性检查
        fmt_keys = dict(zip(fmts_dict["Argument"], fmts_dict["Arguments"]))[fmt]
        validate_args(fmt, self.args, fmt_keys)
        if fmt in {"deepx", "axelera", "imx", "edgetpu"} and not self.args.int8:
            LOGGER.warning(f"{fmt} export requires int8=True, setting int8=True.")
            self.args.int8 = True
        if fmt == "axelera":
            if model.task == "segment" and any(isinstance(m, Segment26) for m in model.modules()):
                raise ValueError("Axelera export does not currently support YOLO26 segmentation models.")
            if not self.args.data:
                self.args.data = TASK2CALIBRATIONDATA.get(model.task)
        if fmt == "imx":
            if not self.args.nms and model.task in {"detect", "pose", "segment"}:
                LOGGER.warning("IMX export requires nms=True, setting nms=True.")
                self.args.nms = True
            if model.task not in {"detect", "pose", "classify", "segment"}:
                raise ValueError(
                    "IMX export only supported for detection, pose estimation, classification, and segmentation models."
                )
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if fmt in {"rknn", "ncnn", "executorch", "paddle", "imx", "edgetpu"}:
                # 为某些不支持 topk 的导出格式禁用 end2end 分支
                model.end2end = False
                LOGGER.warning(f"{fmt.upper()} export does not support end2end models, disabling end2end branch.")
            if fmt == "engine" and self.args.int8:
                # JetPack 6 上的 TensorRT 10.3.0 配合 int8 存在已知的 end2end 构建问题
                # https://github.com/ultralytics/ultralytics/issues/23841
                try:
                    import tensorrt as trt

                    if check_version(trt.__version__, "==10.3.0") and is_jetson(jetpack=6):
                        model.end2end = False
                        LOGGER.warning(
                            "TensorRT 10.3.0 on JetPack 6 with int8 has known end2end build issues, disabling end2end branch. "
                            "For a fix, see https://docs.ultralytics.com/guides/nvidia-jetson/#why-does-my-tensorrt-int8-export-disable-end2end-on-jetpack-6"
                            ""
                        )
                except ImportError:
                    pass
        if self.args.half and self.args.int8:
            LOGGER.warning("half=True and int8=True are mutually exclusive, setting half=False.")
            self.args.half = False
        if self.args.half and fmt == "torchscript" and self.device.type == "cpu":
            LOGGER.warning(
                "half=True only compatible with GPU export for TorchScript, i.e. use device=0, setting half=False."
            )
            self.args.half = False
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # 检查图像尺寸
        if self.args.optimize:
            assert fmt != "ncnn", "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if fmt == "rknn":
            if not self.args.name:
                LOGGER.warning(
                    "Rockchip RKNN export requires a missing 'name' arg for processor type. "
                    "Using default name='rk3588'."
                )
                self.args.name = "rk3588"
            self.args.name = self.args.name.lower()
            assert self.args.name in RKNN_CHIPS, (
                f"Invalid processor name '{self.args.name}' for Rockchip RKNN export. Valid names are {RKNN_CHIPS}."
            )
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert fmt != "tflite" or not ARM64 or not LINUX, "TFLite export with NMS unsupported on ARM64 Linux"
            assert not is_tf_format or TORCH_1_13, "TensorFlow exports with NMS require torch>=1.13"
            assert fmt != "onnx" or TORCH_1_13, "ONNX export with NMS requires torch>=1.13"
            if getattr(model, "end2end", False) or isinstance(model.model[-1], RTDETRDecoder):
                LOGGER.warning("'nms=True' is not available for end2end models. Forcing 'nms=False'.")
                self.args.nms = False
            self.args.conf = self.args.conf or 0.25  # 为 nms 导出设置 conf 默认值
        if (fmt in {"engine", "coreml"} or self.args.nms) and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.args.nms else f'format={self.args.format}'}' requires max batch size, i.e. 'batch=16'"
            )
        if fmt == "edgetpu":
            if not LINUX or ARM64:
                raise SystemError(
                    "Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler"
                )
            elif self.args.batch != 1:  # 参见 github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("Edge TPU export requires batch size 1, setting batch=1.")
                self.args.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "YOLOWorld (original version) export is not supported to any format. "
                "YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
            model.clip_model = None  # openvino int8 导出错误: https://github.com/ultralytics/ultralytics/pull/18445
        if self.args.int8 and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # 分配默认数据
            LOGGER.warning(
                f"INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.args.data}'."
            )
        if fmt == "tfjs" and ARM64 and LINUX:
            raise SystemError("TF.js exports are not currently supported on ARM64 Linux")
        # 如果是导出且在 Intel CPU 上，推荐 OpenVINO
        if SETTINGS.get("openvino_msg"):
            if is_intel():
                LOGGER.info(
                    "💡 ProTip: Export to OpenVINO format for best performance on Intel hardware."
                    " Learn more at https://docs.ultralytics.com/integrations/openvino/"
                )
            SETTINGS["openvino_msg"] = False

        # 输入
        im = torch.zeros(self.args.batch, model.yaml.get("channels", 3), *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # 更新模型
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        if fmt == "imx":
            from ultralytics.utils.export.imx import FXModel

            model = FXModel(model, self.imgsz)
        if fmt in {"tflite", "edgetpu"}:
            from ultralytics.utils.export.tensorflow import tf_wrapper

            model = tf_wrapper(model)
        if fmt == "executorch":
            from ultralytics.utils.export.executorch import executorch_wrapper

            model = executorch_wrapper(model)
        for m in model.modules():
            if isinstance(m, Classify):
                m.export = True
            if isinstance(m, (Detect, RTDETRDecoder)):  # 包括所有 Detect 子类如 Segment, Pose, OBB
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
                # 对于小图像尺寸，将 max_det 限制为锚点数（TensorRT 兼容性所需）
                anchors = sum(int(self.imgsz[0] / s) * int(self.imgsz[1] / s) for s in model.stride.tolist())
                m.max_det = min(self.args.max_det, anchors)
                m.agnostic_nms = self.args.agnostic_nms
                m.xyxy = self.args.nms and fmt != "coreml"
                m.shape = None  # 重置缓存形状以适应新的导出输入尺寸
                if hasattr(model, "pe") and hasattr(m, "fuse") and not hasattr(m, "lrpc"):  # 用于 YOLOE 模型
                    m.fuse(model.pe.to(self.device))
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU 不支持 FlexSplitV，而 split 提供更清晰的 ONNX 图
                m.forward = m.forward_split

        y = None
        for _ in range(2):  # 干运行
            y = NMSModel(model, self.args)(im) if self.args.nms and fmt not in {"coreml", "imx"} else model(im)
        if self.args.half and fmt in {"onnx", "torchscript"} and self.device.type != "cpu":
            im, model = im.half(), model.half()  # 转为 FP16

        # 赋值
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f"Ultralytics {self.pretty_name} model {f'trained on {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(model.stride)),
            "task": model.task,
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": model.names,
            "args": {k: v for k, v in self.args if k in fmt_keys},
            "channels": model.yaml.get("channels", 3),
            "end2end": getattr(model, "end2end", False),
        }  # 模型元数据
        if self.dla is not None:
            self.metadata["dla"] = self.dla  # 确保 `AutoBackend` 使用正确的 dla 设备（如果有）
        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape
            if hasattr(model, "kpt_names"):
                self.metadata["kpt_names"] = model.kpt_names

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"
        )
        self.run_callbacks("on_export_start")

        # 导出
        if is_tf_format:
            f, keras_model = self.export_saved_model()
            if fmt in {"pb", "tfjs"}:  # pb 是 tfjs 的前提
                f = self.export_pb(keras_model=keras_model)
            if fmt == "tflite":
                f = self.export_tflite()
            if fmt == "edgetpu":
                f = self.export_edgetpu(tflite_model=Path(f) / f"{self.file.stem}_full_integer_quant.tflite")
            if fmt == "tfjs":
                f = self.export_tfjs()
        else:
            f = getattr(self, f"export_{fmt}")()

        # 完成
        if f:
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
                f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
            q = "int8" if self.args.int8 else "half" if self.args.half else ""  # 量化
            LOGGER.info(
                f"\nExport complete ({time.time() - t:.1f}s)"
                f"\nResults saved to {colorstr('bold', Path(f).resolve())}"
                f"\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q}"
                f"\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}"
                f"\nVisualize:       https://netron.app"
            )

        self.run_callbacks("on_export_end")
        return f  # 最终导出产物的路径

    def get_int8_calibration_dataloader(self, prefix=""):
        """构建并返回用于 INT8 模型校准的数据加载器。"""
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
        data = (check_cls_dataset if self.model.task == "classify" else check_det_dataset)(self.args.data)
        dataset = YOLODataset(
            data[self.args.split or "val"],
            data=data,
            fraction=self.args.fraction,
            task=self.model.task,
            imgsz=max(self.imgsz),
            augment=False,
            batch_size=self.args.batch,
        )
        if hasattr(dataset.transforms.transforms[0], "new_shape"):
            dataset.transforms.transforms[0].new_shape = self.imgsz  # 非正方形 imgsz 的 LetterBox
        n = len(dataset)
        if n < 1:
            raise ValueError(f"The calibration dataset must have at least 1 image, but found {n} images.")
        batch = min(self.args.batch, n)
        if n < self.args.batch:
            LOGGER.warning(
                f"{prefix} calibration dataset has only {n} images, reducing calibration batch size to {batch}."
            )
        if self.args.format == "axelera" and n < 100:
            LOGGER.warning(f"{prefix} >100 images required for Axelera calibration, found {n} images.")
        elif self.args.format != "axelera" and n < 300:
            LOGGER.warning(f"{prefix} >300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=batch, workers=0, drop_last=True)  # 批次加载所需

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """将 YOLO 模型导出为 TorchScript 格式。"""
        from ultralytics.utils.export.torchscript import torch2torchscript

        return torch2torchscript(
            model=NMSModel(self.model, self.args) if self.args.nms else self.model,
            im=self.im,
            output_file=self.file.with_suffix(".torchscript"),
            optimize=self.args.optimize,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """将 YOLO 模型导出为 ONNX 格式。"""
        requirements = ["onnx>=1.12.0,<2.0.0"]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.71", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)
        import onnx

        from ultralytics.utils.export.engine import best_onnx_opset, torch2onnx

        opset = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type)
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset}...")
        if self.args.nms:
            assert TORCH_1_13, f"'nms=True' ONNX export requires torch>=1.13 (found torch=={TORCH_VERSION})"

        f = str(self.file.with_suffix(".onnx"))
        output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)
            if self.args.nms:  # 使用 NMS 时仅批次大小是动态的
                dynamic["output0"].pop(2)
        if self.args.nms and self.model.task == "obb":
            self.args.opset = opset  # 用于 NMSModel
            self.args.simplify = True  # 修复与 topk 相关的 OBB 运行时错误

        with arange_patch(dynamic=bool(dynamic), half=self.args.half, fmt=self.args.format):
            torch2onnx(
                NMSModel(self.model, self.args) if self.args.nms else self.model,
                self.im,
                f,
                opset=opset,
                input_names=["images"],
                output_names=output_names,
                dynamic=dynamic or None,
            )

        # 检查
        model_onnx = onnx.load(f)  # 加载 onnx 模型

        # 简化
        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)

            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # 元数据
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        # IR 版本
        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(f"{prefix} limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility...")
            model_onnx.ir_version = 10

        # CPU 导出的 FP16 转换（GPU 导出已在跟踪期间通过 model.half() 转为 FP16）
        if self.args.half and self.args.format == "onnx" and self.device.type == "cpu":
            try:
                from onnxruntime.transformers import float16

                LOGGER.info(f"{prefix} converting to FP16...")
                model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
            except Exception as e:
                LOGGER.warning(f"{prefix} FP16 conversion failure: {e}")

        onnx.save(model_onnx, f)
        return f

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        """将 YOLO 模型导出为 OpenVINO 格式。"""
        from ultralytics.utils.export.openvino import torch2openvino

        # macOS 15.4+ 上 OpenVINO <= 2025.1.0 错误: https://github.com/openvinotoolkit/openvino/issues/30023
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"

        def serialize(ov_model, file):
            """设置 RT 信息，序列化，并保存元数据 YAML。"""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

            ov.save_model(ov_model, file, compress_to_fp16=self.args.half)
            YAML.save(Path(file).parent / "metadata.yaml", self.metadata)  # 添加 metadata.yaml

        calibration_dataset, ignored_scope = None, None
        if self.args.int8:
            check_requirements("packaging>=23.2")  # 必须首先安装以构建 nncf 轮子
            check_requirements("nncf>=2.14.0,<3.0.0" if not TORCH_2_3 else "nncf>=2.14.0")
            import nncf

            calibration_dataset = nncf.Dataset(self.get_int8_calibration_dataloader(prefix), self._transform_fn)
            if isinstance(self.model.model[-1], Detect):
                # 包括所有 Detect 子类如 Segment, Pose, OBB, WorldDetect, YOLOEDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])
                ignored_scope = nncf.IgnoredScope(  # 忽略操作
                    patterns=[
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                    ],
                    types=["Sigmoid"],
                )

        ov_model = torch2openvino(
            model=NMSModel(self.model, self.args) if self.args.nms else self.model,
            im=self.im,
            dynamic=self.args.dynamic,
            half=self.args.half,
            int8=self.args.int8,
            calibration_dataset=calibration_dataset,
            ignored_scope=ignored_scope,
            prefix=prefix,
        )

        suffix = f"_{'int8_' if self.args.int8 else ''}openvino_model{os.sep}"
        f = str(self.file).replace(self.file.suffix, suffix)
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """将 YOLO 模型导出为 PaddlePaddle 格式。"""
        from ultralytics.utils.export.paddle import torch2paddle

        return torch2paddle(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}"),
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):
        """使用 MNN https://github.com/alibaba/MNN 将 YOLO 模型导出为 MNN 格式。"""
        from ultralytics.utils.export.mnn import onnx2mnn

        return onnx2mnn(
            onnx_file=self.export_onnx(),
            output_file=self.file.with_suffix(".mnn"),
            half=self.args.half,
            int8=self.args.int8,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """使用 PNNX https://github.com/pnnx/pnnx 将 YOLO 模型导出为 NCNN 格式。"""
        from ultralytics.utils.export.ncnn import torch2ncnn

        return torch2ncnn(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, "_ncnn_model/"),
            half=self.args.half,
            metadata=self.metadata,
            device=self.device,
            prefix=prefix,
        )

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """将 YOLO 模型导出为 CoreML 格式。"""
        mlmodel = self.args.format.lower() == "mlmodel"  # 请求的是旧版 *.mlmodel 导出格式
        from ultralytics.utils.export.coreml import IOSDetectModel, pipeline_coreml, torch2coreml

        # 最新的 numpy 2.4.0rc1 会破坏 coremltools 导出
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert TORCH_1_11, "CoreML export requires torch>=1.11"
        if self.args.batch > 1:
            assert self.args.dynamic, (
                "batch sizes > 1 are not supported without 'dynamic=True' for CoreML export. Please retry at 'dynamic=True'."
            )
        if self.args.dynamic:
            assert not self.args.nms, (
                "'nms=True' cannot be used together with 'dynamic=True' for CoreML export. Please disable one of them."
            )
            assert self.model.task != "classify", "'dynamic=True' is not supported for CoreML classification models."
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        if self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im, mlprogram=not mlmodel) if self.args.nms else self.model
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} 'nms=True' is only available for Detect models like 'yolo26n.pt'.")
                # TODO CoreML 分割和姿态模型流水线
            model = self.model

        if self.args.dynamic:
            input_shape = ct.Shape(
                shape=(
                    ct.RangeDim(lower_bound=1, upper_bound=self.args.batch, default=1),
                    self.im.shape[1],
                    ct.RangeDim(lower_bound=32, upper_bound=self.imgsz[0] * 2, default=self.imgsz[0]),
                    ct.RangeDim(lower_bound=32, upper_bound=self.imgsz[1] * 2, default=self.imgsz[1]),
                )
            )
            inputs = [ct.TensorType("image", shape=input_shape)]
        else:
            inputs = [ct.ImageType("image", shape=self.im.shape, scale=1 / 255, bias=[0.0, 0.0, 0.0])]

        ct_model = torch2coreml(
            model=model,
            inputs=inputs,
            im=self.im,
            classifier_names=list(self.model.names.values()) if self.model.task == "classify" else None,
            mlmodel=mlmodel,
            half=self.args.half,
            int8=self.args.int8,
            metadata=self.metadata,
            prefix=prefix,
        )

        if self.args.nms and self.model.task == "detect":
            ct_model = pipeline_coreml(
                ct_model,
                self.output_shape,
                weights_dir=None if mlmodel else ct_model.weights_dir,
                metadata=self.metadata,
                mlmodel=mlmodel,
                iou=self.args.iou,
                conf=self.args.conf,
                agnostic_nms=self.args.agnostic_nms,
                prefix=prefix,
            )

        if self.model.task == "classify":
            ct_model.user_defined_metadata.update({"com.apple.coreml.model.preview.type": "imageClassifier"})

        try:
            ct_model.save(str(f))  # 保存 *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f

    @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):
        """将 YOLO 模型导出为 TensorRT 格式 https://developer.nvidia.com/tensorrt。"""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx = self.export_onnx()  # 在 TRT 导入前运行 https://github.com/ultralytics/ultralytics/issues/7016
        from ultralytics.utils.export.engine import onnx2engine

        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine 文件
        onnx2engine(
            f_onnx,
            f,
            self.args.workspace,
            self.args.half,
            self.args.int8,
            self.args.dynamic,
            self.im.shape,
            dla=self.dla,
            dataset=self.get_int8_calibration_dataloader(prefix) if self.args.int8 else None,
            metadata=self.metadata,
            verbose=self.args.verbose,
            prefix=prefix,
        )

        return f

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """将 YOLO 模型导出为 TensorFlow SavedModel 格式。"""
        from ultralytics.utils.export.tensorflow import onnx2saved_model

        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # 删除输出文件夹

        # 导出到 TF
        images = None
        if self.args.int8 and self.args.data:
            images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]
            images = (
                torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz)
                .permute(0, 2, 3, 1)
                .numpy()
                .astype(np.float32)
            )

        # 导出到 ONNX
        if isinstance(self.model.model[-1], RTDETRDecoder):
            self.args.opset = self.args.opset or 19
            assert 16 <= self.args.opset <= 19, "RTDETR export requires opset>=16;<=19"
        self.args.simplify = True
        f_onnx = self.export_onnx()  # 确保 ONNX 可用
        keras_model = onnx2saved_model(
            f_onnx,
            f,
            int8=self.args.int8,
            images=images,
            disable_group_convolution=self.args.format in {"tfjs", "edgetpu"},
            prefix=prefix,
        )
        YAML.save(f / "metadata.yaml", self.metadata)  # 添加 metadata.yaml
        # 添加 TFLite 元数据
        for file in f.rglob("*.tflite"):
            file.unlink() if "quant_with_int16_act.tflite" in str(file) else self._add_tflite_metadata(file)

        return str(f), keras_model  # 或者 keras_model = tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """将 YOLO 模型导出为 TensorFlow GraphDef *.pb 格式 https://github.com/leimao/Frozen-Graph-TensorFlow。"""
        from ultralytics.utils.export.tensorflow import keras2pb

        return keras2pb(keras_model, output_file=self.file.with_suffix(".pb"), prefix=prefix)

    @try_export
    def export_tflite(self, prefix=colorstr("TensorFlow Lite:")):
        """将 YOLO 模型导出为 TensorFlow Lite 格式。"""
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        import tensorflow as tf

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 输入/输出
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 输入/输出
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        return str(f)

    @try_export
    def export_axelera(self, prefix=colorstr("Axelera:")):
        """将 YOLO 模型导出为 Axelera 格式。"""
        assert LINUX and not (ARM64 and IS_DOCKER), (
            "export is only supported on Linux and is not supported on ARM64 Docker."
        )
        assert TORCH_2_8, "export requires torch>=2.8.0."

        from ultralytics.utils.export.axelera import torch2axelera

        output_dir = self.file.parent / f"{self.file.stem}_axelera_model"
        return torch2axelera(
            model=self.model,
            output_dir=output_dir,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix),
            transform_fn=self._transform_fn,
            model_name=self.file.stem,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_executorch(self, prefix=colorstr("ExecuTorch:")):
        """将 YOLO 模型导出为 ExecuTorch *.pte 格式。"""
        assert TORCH_2_9, f"ExecuTorch requires torch>=2.9.0 but torch=={TORCH_VERSION} is installed"
        from ultralytics.utils.export.executorch import torch2executorch

        return torch2executorch(
            model=self.model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, "_executorch_model/"),
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """将 YOLO 模型导出为 Edge TPU 格式 https://coral.ai/docs/edgetpu/models-intro/。"""
        from ultralytics.utils.export.tensorflow import tflite2edgetpu

        output_file = tflite2edgetpu(tflite_file=tflite_model, output_dir=tflite_model.parent, prefix=prefix)
        self._add_tflite_metadata(output_file)
        return output_file

    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """将 YOLO 模型导出为 TensorFlow.js 格式。"""
        from ultralytics.utils.export.tensorflow import pb2tfjs

        output_dir = pb2tfjs(
            pb_file=str(self.file.with_suffix(".pb")),
            output_dir=str(self.file).replace(self.file.suffix, "_web_model/"),
            half=self.args.half,
            int8=self.args.int8,
            prefix=prefix,
        )
        YAML.save(Path(output_dir) / "metadata.yaml", self.metadata)
        return output_dir

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):
        """将 YOLO 模型导出为 RKNN 格式。"""
        from ultralytics.utils.export.rknn import onnx2rknn

        self.args.opset = min(self.args.opset or 19, 19)  # rknn-toolkit 期望 opset<=19
        f_onnx = self.export_onnx()
        return onnx2rknn(
            onnx_file=f_onnx,
            output_dir=str(self.file).replace(self.file.suffix, f"_rknn_model{os.sep}"),
            name=self.args.name,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_imx(self, prefix=colorstr("IMX:")):
        """将 YOLO 模型导出为 IMX 格式。"""
        assert LINUX, (
            "Export only supported on Linux."
            "See https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter?version=3.17.3&progLang="
        )
        assert IS_PYTHON_MINIMUM_3_9, "IMX export is only supported on Python 3.9 or above."

        if getattr(self.model, "end2end", False):
            raise ValueError("IMX export is not supported for end2end models.")
        from ultralytics.utils.export.imx import torch2imx

        return torch2imx(
            model=self.model,
            output_dir=str(self.file).replace(self.file.suffix, "_imx_model/"),
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            metadata=self.metadata,
            dataset=partial(self.get_int8_calibration_dataloader, prefix),
            prefix=prefix,
        )

    @try_export
    def export_deepx(self, prefix=colorstr("DeepX:")):
        """将 YOLO 模型导出为 DeepX 格式。"""
        assert LINUX and not ARM64, "DeepX export only supported on non-aarch64 Linux"
        from ultralytics.utils.export.deepx import onnx2deepx

        f = self.export_onnx()
        return onnx2deepx(
            onnx_file=f,
            imgsz=self.imgsz,
            dataset=self.get_int8_calibration_dataloader(prefix),
            metadata=self.metadata,
            optimize=self.args.optimize,
            prefix=prefix,
        )

    def _add_tflite_metadata(self, file):
        """根据 https://ai.google.dev/edge/litert/models/metadata 向 *.tflite 模型添加元数据。"""
        import zipfile

        with zipfile.ZipFile(file, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(self.metadata, indent=2))

    @staticmethod
    def _transform_fn(data_item) -> np.ndarray:
        """Axelera/OpenVINO 量化预处理的变换函数。"""
        data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
        assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
        im = data_item.numpy().astype(np.float32) / 255.0  # uint8 转 fp16/32，0-255 转 0.0-1.0
        return im[None] if im.ndim == 3 else im

    def add_callback(self, event: str, callback):
        """将给定的回调追加到指定事件。"""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """执行为给定事件注册的所有回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)


class NMSModel(torch.nn.Module):
    """嵌入 NMS 的模型包装器，用于 Detect、Segment、Pose 和 OBB。"""

    def __init__(self, model, args):
        """初始化 NMSModel。

        Args:
            model (torch.nn.Module): 要用 NMS 后处理包装的模型。
            args (SimpleNamespace): 导出参数。
        """
        super().__init__()
        self.model = model
        self.args = args
        self.obb = model.task == "obb"
        self.is_tf = self.args.format in frozenset({"saved_model", "tflite", "tfjs"})

    def forward(self, x):
        """执行带 NMS 后处理的推理。支持 Detect、Segment、OBB 和 Pose。

        Args:
            x (torch.Tensor): 形状为 (B, C, H, W) 的预处理张量。

        Returns:
            (torch.Tensor | tuple): 形状为 (B, max_det, 4 + 2 + extra_shape) 的张量，其中 B 是批次大小，
                或用于分割模型的 (检测, proto) 元组。
        """
        from torchvision.ops import nms

        preds = self.model(x)
        pred = preds[0] if isinstance(preds, tuple) else preds
        kwargs = dict(device=pred.device, dtype=pred.dtype)
        bs = pred.shape[0]
        pred = pred.transpose(-1, -2)  # shape(1,84,6300) 转 shape(1,6300,84)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))  # 来自 Segment、OBB、Pose 的额外部分
        if self.args.dynamic and self.args.batch > 1:  # 由于循环展开，批次大小需要始终相同
            pad = torch.zeros(torch.max(torch.tensor(self.args.batch - bs), torch.tensor(0)), *pred.shape[1:], **kwargs)
            pred = torch.cat((pred, pad))
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(dim=-1)
        self.args.max_det = min(pred.shape[1], self.args.max_det)  # 以防 num_anchors < max_det
        # (N, max_det, 4 坐标 + 1 类别分数 + 1 类别标签 + extra_shape)。
        out = torch.zeros(pred.shape[0], self.args.max_det, boxes.shape[-1] + 2 + extra_shape, **kwargs)
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.args.conf
            if self.is_tf or (self.args.format == "onnx" and self.obb):
                # 如果 mask 为空则 TFLite GatherND 错误
                score *= mask
                # 显式长度否则 reshape 错误，硬编码为 `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # `8` 是实验得出的 obb 正确 NMS 结果的最小值
            multiplier = 8 if self.obb else 1 / max(len(self.model.names), 1)
            # 为 NMS 归一化边界框，因为类别偏移的较大值会导致 int8 量化问题
            if self.args.format == "tflite":  # TFLite 已归一化
                nmsbox *= multiplier
            else:
                nmsbox = multiplier * (nmsbox / torch.tensor(x.shape[2:], **kwargs).max())
            if not self.args.agnostic_nms:  # 按类别 NMS
                end = 2 if self.obb else 4
                # 完全显式展开否则 reshape 错误
                cls_offset = cls.view(cls.shape[0], 1).expand(cls.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)
            nms_fn = (
                partial(
                    TorchNMS.fast_nms,
                    use_triu=not (
                        self.is_tf
                        or (self.args.opset or 14) < 14
                        or (self.args.format == "openvino" and self.args.int8)  # OpenVINO int8 与 triu 一起出错
                    ),
                    iou_func=batch_probiou,
                    exit_early=False,
                )
                if self.obb
                else nms
            )
            keep = nms_fn(
                torch.cat([nmsbox, extra], dim=-1) if self.obb else nmsbox,
                score,
                self.args.iou,
            )[: self.args.max_det]
            dets = torch.cat(
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            # 零填充到 max_det 大小以避免 reshape 错误
            pad = (0, 0, 0, self.args.max_det - dets.shape[0])
            out[i] = torch.nn.functional.pad(dets, pad)
        return (out[:bs], preds[1]) if self.model.task == "segment" else out[:bs]
