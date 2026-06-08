# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER

from .base import BaseBackend


class TensorFlowBackend(BaseBackend):
    """谷歌 TensorFlow 推理后端，支持多种序列化格式。

    加载并运行谷歌 TensorFlow 模型，支持 SavedModel、GraphDef（.pb）、
    TFLite（.tflite）和 Edge TPU 格式。处理量化模型的反量化和任务特定输出格式化。
    """

    def __init__(self, weight: str | Path, device: torch.device, fp16: bool = False, format: str = "saved_model"):
        """初始化谷歌 TensorFlow 后端。

        Args:
            weight (str | Path): SavedModel 目录、.pb 文件或 .tflite 文件路径。
            device (torch.device): 执行推理的设备。
            fp16 (bool): 是否使用 FP16 半精度推理。
            format (str): 模型格式，可选 "saved_model"、"pb"、"tflite" 或 "edgetpu"。
        """
        assert format in {"saved_model", "pb", "tflite", "edgetpu"}, f"Unsupported TensorFlow format: {format}."
        self.format = format
        super().__init__(weight, device, fp16)

    def load_model(self, weight: str | Path) -> None:
        """加载 SavedModel、GraphDef、TFLite 或 Edge TPU 格式的谷歌 TensorFlow 模型。

        Args:
            weight (str | Path): 模型文件或目录路径。
        """
        if self.format in {"saved_model", "pb"}:
            import tensorflow as tf

        if self.format == "saved_model":
            LOGGER.info(f"Loading {weight} for TensorFlow SavedModel inference...")
            self.model = tf.saved_model.load(weight)
            # 加载 SavedModel 目录下的元数据
            metadata_file = Path(weight) / "metadata.yaml"
            if metadata_file.exists():
                from ultralytics.utils import YAML

                self.apply_metadata(YAML.load(metadata_file))
        elif self.format == "pb":
            LOGGER.info(f"Loading {weight} for TensorFlow GraphDef inference...")
            from ultralytics.utils.export.tensorflow import gd_outputs

            def wrap_frozen_graph(gd, inputs, outputs):
                """通过剪枝将 TensorFlow 冻结图包装为可推理的函数，指定输入/输出节点。"""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            gd = tf.Graph().as_graph_def()
            with open(weight, "rb") as f:
                gd.ParseFromString(f.read())
            self.frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))

            # 尝试在相邻目录中查找 SavedModel 格式的元数据文件
            try:
                metadata_file = next(
                    Path(weight).resolve().parent.rglob(f"{Path(weight).stem}_saved_model*/metadata.yaml")
                )
                from ultralytics.utils import YAML

                self.apply_metadata(YAML.load(metadata_file))
            except StopIteration:
                pass
        else:
            # TFLite 和 Edge TPU 格式
            try:
                # 优先使用轻量级 tflite_runtime（不依赖完整 TF）
                from tflite_runtime.interpreter import Interpreter, load_delegate

                self.tf = None
            except ImportError:
                # 回退到完整 TensorFlow 的 Lite 解释器
                import tensorflow as tf

                self.tf = tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate

            if self.format == "edgetpu":
                # Edge TPU 需要加载平台特定的委托库（.so/.dylib/.dll）
                device = self.device[3:] if str(self.device).startswith("tpu") else ":0"
                LOGGER.info(f"Loading {weight} on device {device[1:]} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                self.interpreter = Interpreter(
                    model_path=str(weight),
                    experimental_delegates=[load_delegate(delegate, options={"device": device})],
                )
                # Edge TPU 从 PyTorch 视角来看运行在 CPU 上
                self.device = torch.device("cpu")
            else:
                LOGGER.info(f"Loading {weight} for TensorFlow Lite inference...")
                self.interpreter = Interpreter(model_path=weight)

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # 从 TFLite 文件内嵌的 ZIP 压缩数据中读取元数据
            try:
                with zipfile.ZipFile(weight, "r") as zf:
                    name = zf.namelist()[0]
                    contents = zf.read(name).decode("utf-8")
                    if name == "metadata.json":
                        self.apply_metadata(json.loads(contents))
                    else:
                        self.apply_metadata(ast.literal_eval(contents))
            except (zipfile.BadZipFile, SyntaxError, ValueError, json.JSONDecodeError):
                pass

    def forward(self, im: torch.Tensor) -> list[np.ndarray]:
        """执行谷歌 TensorFlow 推理，根据格式选择执行路径并进行输出后处理。

        Args:
            im (torch.Tensor): 输入图像张量，格式为 BHWC（由 AutoBackend 从 BCHW 转换而来）。

        Returns:
            (list[np.ndarray]): 模型预测结果列表，每个元素为 numpy 数组。
        """
        im = im.cpu().numpy()
        if self.format == "saved_model":
            # SavedModel：直接调用 serving_default 签名
            y = self.model.serving_default(im)
            if not isinstance(y, list):
                y = [y]
        elif self.format == "pb":
            import tensorflow as tf

            # GraphDef：通过冻结函数执行
            y = self.frozen_func(x=tf.constant(im))
        else:
            # TFLite / Edge TPU：通过解释器逐张量执行
            h, w = im.shape[1:3]

            details = self.input_details[0]
            is_int = details["dtype"] in {np.int8, np.int16}

            if is_int:
                # 量化模型：将 float 输入量化为 int8/int16
                scale, zero_point = details["quantization"]
                im = (im / scale + zero_point).astype(details["dtype"])

            self.interpreter.set_tensor(details["index"], im)
            self.interpreter.invoke()

            y = []
            for output in self.output_details:
                x = self.interpreter.get_tensor(output["index"])
                if is_int:
                    # 量化模型：将 int8/int16 输出反量化为 float
                    scale, zero_point = output["quantization"]
                    x = (x.astype(np.float32) - zero_point) * scale
                if x.ndim == 3:
                    # 对 xywh 坐标按图像尺寸反归一化
                    if x.shape[-1] == 6 or self.end2end:
                        x[:, :, [0, 2]] *= w
                        x[:, :, [1, 3]] *= h
                        if self.task == "pose":
                            x[:, :, 6::3] *= w
                            x[:, :, 7::3] *= h
                    else:
                        x[:, [0, 2]] *= w
                        x[:, [1, 3]] *= h
                        if self.task == "pose":
                            x[:, 5::3] *= w
                            x[:, 6::3] *= h
                y.append(x)

        if self.task == "segment":
            # 分割任务：修正 (det, proto) 输出顺序
            if len(y[1].shape) != 4:
                y = list(reversed(y))  # 期望：(1, 116, 8400), (1, 160, 160, 32)
            if y[1].shape[-1] == 6:
                # 端到端模型只保留检测输出
                y = [y[1]]
            else:
                y[1] = np.transpose(y[1], (0, 3, 1, 2))  # BHWC → BCHW：(1, 32, 160, 160)
        return [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
