# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import IS_JETSON, LOGGER, PYTHON_VERSION
from ultralytics.utils.checks import check_requirements, check_tensorrt, check_version

from .base import BaseBackend


class TensorRTBackend(BaseBackend):
    """NVIDIA TensorRT 推理后端，用于 GPU 加速部署。

    加载并运行 NVIDIA TensorRT 序列化引擎（.engine 文件）。
    同时支持 TensorRT 7-9 及 TensorRT 10+ API，支持动态输入形状、FP16 精度和 DLA 核心卸载。
    """

    def load_model(self, weight: str | Path) -> None:
        """从序列化的 .engine 文件中加载 NVIDIA TensorRT 引擎。

        Args:
            weight (str | Path): 包含可选内嵌元数据的 .engine 文件路径。
        """
        LOGGER.info(f"Loading {weight} for TensorRT inference...")

        # Jetson + Python <=3.8.10 环境下需锁定 numpy 版本
        if IS_JETSON and check_version(PYTHON_VERSION, "<=3.8.10"):
            check_requirements("numpy==1.23.5")

        try:
            import tensorrt as trt
        except ImportError:
            check_tensorrt()
            import tensorrt as trt

        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.2.0", msg="https://github.com/ultralytics/ultralytics/pull/24367")

        # TensorRT 只支持 CUDA 设备
        if self.device.type == "cpu":
            self.device = torch.device("cuda:0")

        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)

        # 读取引擎文件：先尝试解析文件头部的 JSON 元数据，再反序列化引擎
        with open(weight, "rb") as f, trt.Runtime(logger) as runtime:
            try:
                # 文件头 4 字节为元数据长度（小端序），后跟 JSON 字符串
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata = json.loads(f.read(meta_len).decode("utf-8"))
                dla = metadata.get("dla", None)
                if dla is not None:
                    # 配置 DLA 核心（深度学习加速器，仅 Jetson AGX 等平台支持）
                    runtime.DLA_core = int(dla)
            except UnicodeDecodeError:
                # 旧版引擎文件无元数据头，从头重新读取
                f.seek(0)
                metadata = None
            engine = runtime.deserialize_cuda_engine(f.read())
            self.apply_metadata(metadata)
        try:
            self.context = engine.create_execution_context()
        except Exception as e:
            LOGGER.error("TensorRT model exported with a different version than expected\n")
            raise e

        # 初始化绑定表：记录每个 IO 张量的名称、类型、形状和 GPU 指针
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        self.dynamic = False
        # TRT10+ 使用 num_io_tensors 接口；旧版使用 num_bindings 接口
        self.is_trt10 = not hasattr(engine, "num_bindings")
        num = range(engine.num_io_tensors) if self.is_trt10 else range(engine.num_bindings)

        for i in num:
            if self.is_trt10:
                name = engine.get_tensor_name(i)
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(engine.get_tensor_shape(name))
                # 取 profile 的最大形状作为动态形状的预分配尺寸
                profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None
            else:
                name = engine.get_binding_name(i)
                dtype = trt.nptype(engine.get_binding_dtype(i))
                is_input = engine.binding_is_input(i)
                shape = tuple(engine.get_binding_shape(i))
                profile_shape = tuple(engine.get_profile_shape(0, i)[1]) if is_input else None

            if is_input:
                if -1 in shape:
                    # 包含 -1 的形状表示动态输入，先用 profile 形状初始化上下文
                    self.dynamic = True
                    if self.is_trt10:
                        self.context.set_input_shape(name, profile_shape)
                    else:
                        self.context.set_binding_shape(i, profile_shape)
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            # 获取上下文推断后的实际形状并分配 GPU 内存
            shape = (
                tuple(self.context.get_tensor_shape(name))
                if self.is_trt10
                else tuple(self.context.get_binding_shape(i))
            )
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        # 地址表：用于 execute_v2 的指针列表
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.model = engine

    def forward(self, im: torch.Tensor) -> list[torch.Tensor]:
        """执行 NVIDIA TensorRT 推理，支持动态形状处理。

        Args:
            im (torch.Tensor): 位于 CUDA 设备上的输入图像张量，格式为 BCHW。

        Returns:
            (list[torch.Tensor]): 位于 CUDA 设备上的模型预测结果张量列表。
        """
        if self.dynamic and im.shape != self.bindings["images"].shape:
            # 动态形状：输入尺寸变化时，重新绑定输入并调整输出缓冲区大小
            if self.is_trt10:
                self.context.set_input_shape("images", im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
            else:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

        s = self.bindings["images"].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"

        # 更新输入 GPU 指针并执行推理
        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        # 按输出名称排序返回结果，保证顺序一致
        return [self.bindings[x].data for x in sorted(self.output_names)]
