# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import functools
import gc
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import __version__
from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    NUM_THREADS,
    PYTHON_VERSION,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    WINDOWS,
    colorstr,
)
from ultralytics.utils.checks import check_version
from ultralytics.utils.cpu import CPUInfo
from ultralytics.utils.patches import torch_load

# Version checks (all default to version>=min_version)
TORCH_1_9 = check_version(TORCH_VERSION, "1.9.0")
TORCH_1_10 = check_version(TORCH_VERSION, "1.10.0")
TORCH_1_11 = check_version(TORCH_VERSION, "1.11.0")
TORCH_1_13 = check_version(TORCH_VERSION, "1.13.0")
TORCH_2_0 = check_version(TORCH_VERSION, "2.0.0")
TORCH_2_1 = check_version(TORCH_VERSION, "2.1.0")
TORCH_2_3 = check_version(TORCH_VERSION, "2.3.0")
TORCH_2_4 = check_version(TORCH_VERSION, "2.4.0")
TORCH_2_8 = check_version(TORCH_VERSION, "2.8.0")
TORCH_2_9 = check_version(TORCH_VERSION, "2.9.0")
TORCH_2_10 = check_version(TORCH_VERSION, "2.10.0")
TORCH_2_12 = check_version(TORCH_VERSION, "2.12.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")
if WINDOWS and check_version(TORCH_VERSION, "==2.4.0"):  # 在 Windows 上拒绝 2.4.0 版本
    LOGGER.warning(
        "Known issue with torch==2.4.0 on Windows with CPU, recommend upgrading to torch>=2.4.1 to resolve "
        "https://github.com/ultralytics/ultralytics/issues/15049"
    )


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """确保分布式训练中的所有进程等待本地主进程（rank 0）先完成任务。"""
    initialized = dist.is_available() and dist.is_initialized()
    use_ids = initialized and dist.get_backend() == "nccl"

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()


def smart_inference_mode():
    """若 torch>=1.10.0 则应用 torch.inference_mode() 装饰器，否则应用 torch.no_grad() 装饰器。"""

    def decorate(fn):
        """根据 torch 版本应用适当的推理模式装饰器。"""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # 已在 inference_mode 中，直接透传
        else:
            return (torch.inference_mode if TORCH_1_10 else torch.no_grad)()(fn)

    return decorate


def autocast(enabled: bool, device: str = "cuda"):
    """根据 PyTorch 版本和 AMP 设置获取适当的 autocast 上下文管理器。

    此函数返回一个兼容新旧 PyTorch 版本的自动混合精度（AMP）训练上下文管理器，处理不同 PyTorch 版本间 autocast API 的差异。

    Args:
        enabled (bool): 是否启用自动混合精度。
        device (str, optional): 用于 autocast 的设备。

    Returns:
        (torch.amp.autocast): 适当的 autocast 上下文管理器。

    Examples:
        >>> with autocast(enabled=True):
        ...     # 此处为混合精度操作
        ...     pass

    Notes:
        - 对于 PyTorch 1.13 及更新版本，使用 `torch.amp.autocast`。
        - 对于旧版本，使用 `torch.cuda.amp.autocast`。
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)


@functools.lru_cache
def get_cpu_info():
    """返回系统 CPU 信息字符串，如 'Apple M2'。"""
    from ultralytics.utils import PERSISTENT_CACHE  # 避免循环导入错误

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            PERSISTENT_CACHE["cpu_info"] = CPUInfo.name()
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


@functools.lru_cache
def get_gpu_info(index):
    """返回系统 GPU 信息字符串，如 'Tesla T4, 15102MiB'。"""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def select_device(device="", newline=False, verbose=True):
    """根据提供的参数选择适当的 PyTorch 设备。

    此函数接收指定设备的字符串或 torch.device 对象，返回表示所选设备的 torch.device 对象。函数还会验证可用设备数量，
    如果请求的设备不可用则抛出异常。

    Args:
        device (str | torch.device, optional): 设备字符串或 torch.device 对象。选项包括 'cpu'、'cuda'、'0'、
            '0,1,2,3'、'mps'、'npu'、'npu:0' 或 '-1' 自动选择。默认自动选择第一个可用 GPU，若无 GPU 则选择 CPU。
        newline (bool, optional): 若为 True，在日志字符串末尾添加换行。
        verbose (bool, optional): 若为 True，记录设备信息。

    Returns:
        (torch.device): Selected device.

    Examples:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    Notes:
        设置 'CUDA_VISIBLE_DEVICES' 环境变量以指定使用的 GPU。
    """
    if isinstance(device, torch.device) or str(device).startswith(("tpu", "intel", "vulkan")):
        return device

    s = f"Ultralytics {__version__} 🚀 Python-{PYTHON_VERSION} torch-{TORCH_VERSION} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # 转为字符串，'cuda:0' -> '0'，'(0, 1)' -> '0,1'

    # 华为昇腾 NPU
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa
        except ImportError:
            raise ValueError(f"Invalid NPU 'device={device}'. Install 'torch_npu' at https://github.com/Ascend/pytorch")

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise ValueError(f"Invalid NPU 'device={device}' requested. Ascend NPU is not available.")

        # 解析 'npu' 或 'npu:N'（尚不支持多 NPU）
        suffix = device[3:]
        if suffix == "":
            idx = 0
        elif suffix.startswith(":") and suffix[1:].isdigit():
            idx = int(suffix[1:])
        else:
            raise ValueError(f"Invalid NPU 'device={device}' format. Use 'npu' or 'npu:0'.")

        n = torch.npu.device_count()
        if idx >= n:
            raise ValueError(f"Invalid NPU 'device={device}' requested. Only {n} NPU(s) available.")

        torch.npu.set_device(idx)
        if verbose:
            LOGGER.info(f"{s}NPU:{idx} ({torch.npu.get_device_name(idx)})\n")
        return torch.device(f"npu:{idx}")

    # 自动选择 GPU
    if "-1" in device:
        from ultralytics.utils.autodevice import GPUInfo

        # 将每个 -1 替换为选定的 GPU 或移除
        parts = device.split(",")
        selected = GPUInfo().select_idle_gpu(count=parts.count("-1"), min_memory_fraction=0.2)
        for i in range(len(parts)):
            if parts[i] == "-1":
                parts[i] = str(selected.pop(0)) if selected else ""
        device = ",".join(p for p in parts if p)

    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 torch.cuda.is_available() = False
    elif device:  # 请求了非 CPU 设备
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])  # 移除连续逗号，如 "0,,1" -> "0,1"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # 设置环境变量 - 必须在 assert is_available() 之前
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # 优先使用 GPU
        devices = device.split(",") if device else "0"  # 如 "0,1" -> ["0", "1"]
        space = " " * len(s)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(i)})\n"  # 字节转 MB
        arg = "cuda:0"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # 优先使用 MPS（若可用）
        s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:  # 回退到 CPU
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # 重置 OMP_NUM_THREADS 用于 CPU 训练
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def time_sync():
    """返回 PyTorch 精确时间。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):
    """融合 Conv2d 和 BatchNorm2d 层以优化推理。

    Args:
        conv (nn.Conv2d): 要融合的卷积层。
        bn (nn.BatchNorm2d): 要融合的批归一化层。

    Returns:
        (nn.Conv2d): 禁用梯度的融合卷积层。

    Examples:
        >>> conv = nn.Conv2d(3, 16, 3)
        >>> bn = nn.BatchNorm2d(16)
        >>> fused_conv = fuse_conv_and_bn(conv, bn)
    """
    # 计算融合权重
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    conv.weight.data = torch.mm(w_bn, w_conv).view(conv.weight.shape)

    # 计算融合偏置
    b_conv = (
        torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    if conv.bias is None:
        conv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        conv.bias.data = fused_bias

    return conv.requires_grad_(False)


def fuse_deconv_and_bn(deconv, bn):
    """融合 ConvTranspose2d 和 BatchNorm2d 层以优化推理。

    Args:
        deconv (nn.ConvTranspose2d): 要融合的转置卷积层。
        bn (nn.BatchNorm2d): 要融合的批归一化层。

    Returns:
        (nn.ConvTranspose2d): 禁用梯度的融合转置卷积层。

    Examples:
        >>> deconv = nn.ConvTranspose2d(16, 3, 3)
        >>> bn = nn.BatchNorm2d(3)
        >>> fused_deconv = fuse_deconv_and_bn(deconv, bn)
    """
    # 计算融合权重
    w_deconv = deconv.weight.view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    deconv.weight.data = torch.mm(w_bn, w_deconv).view(deconv.weight.shape)

    # 计算融合偏置
    b_conv = (
        torch.zeros(deconv.out_channels, device=deconv.weight.device, dtype=deconv.weight.dtype)
        if deconv.bias is None
        else deconv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    if deconv.bias is None:
        deconv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        deconv.bias.data = fused_bias

    return deconv.requires_grad_(False)


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """逐层打印并返回详细的模型信息。

    Args:
        model (nn.Module): 要分析的模型。
        detailed (bool, optional): 是否打印详细的层信息。
        verbose (bool, optional): 是否打印模型信息。
        imgsz (int | list, optional): 输入图像尺寸。

    Returns:
        (tuple): Tuple containing:
            - n_l (int): 层数。
            - n_p (int): 参数数量。
            - n_g (int): 梯度数量。
            - flops (float): GFLOPs。
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # 参数数量
    n_g = get_num_gradients(model)  # 梯度数量
    layers = __import__("collections").OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)  # 层数
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        LOGGER.info(h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{list(p.shape)!s:>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('torch.', ''):>15}"
                    )
            else:  # 无可学习参数的层
                LOGGER.info(f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{[]!s:>20}{'-':>10}{'-':>10}{'-':>15}")

    flops = get_flops(model, imgsz)  # imgsz 可以是 int 或 list，如 imgsz=640 或 imgsz=[640, 320]
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """返回 YOLO 模型的参数总数。"""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """返回 YOLO 模型中有梯度的参数总数。"""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def model_info_for_loggers(trainer):
    """返回包含有用模型信息的字典。

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含模型和验证数据的训练器对象。

    Returns:
        (dict): 包含模型参数、GFLOPs 和推理速度的字典。

    Examples:
        YOLOv8n info for loggers
        >>> results = {
        ...    "model/parameters": 3151904,
        ...    "model/GFLOPs": 8.746,
        ...    "model/speed_ONNX(ms)": 41.244,
        ...    "model/speed_TensorRT(ms)": 3.211,
        ...    "model/speed_PyTorch(ms)": 18.755,
        ...}
    """
    if trainer.args.profile:  # 分析 ONNX 和 TensorRT 耗时
        from ultralytics.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device).run()[0]
        results.pop("model/name")
    else:  # 仅从最近验证返回 PyTorch 耗时
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3),
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results


def get_flops(model, imgsz=640):
    """计算模型的 FLOPs（浮点运算次数），单位 GFLOPs。

    尝试两种计算方法：首先使用基于步幅的张量以提高效率，如需要则回退到完整图像尺寸
    （如 RTDETR 模型）。如果 thop 库不可用或计算失败则返回 0.0。

    Args:
        model (nn.Module): 要计算 FLOPs 的模型。
        imgsz (int | list, optional): 输入图像尺寸。

    Returns:
        (float): 模型的 GFLOPs（十亿浮点运算次数）。
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda 环境未安装 'ultralytics-thop'

    if not thop:
        return 0.0  # 若未安装则返回 0.0 GFLOPs

    try:
        model = unwrap_model(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # 若为 int/float 则扩展
        try:
            # 方法1：使用基于步幅的输入张量
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # 最大步幅
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # 输入图像，BCHW 格式
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # 步幅 GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # 图像尺寸 GFLOPs
        except Exception:
            # 方法2：使用实际图像尺寸（RTDETR 模型需要）
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # 输入图像，BCHW 格式
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # 图像尺寸 GFLOPs
    except Exception:
        return 0.0


def get_flops_with_torch_profiler(model, imgsz=640):
    """使用 torch profiler 计算模型 FLOPs（thop 包的替代方案，但慢 2-10 倍）。

    Args:
        model (nn.Module): 要计算 FLOPs 的模型。
        imgsz (int | list, optional): 输入图像尺寸。

    Returns:
        (float): 模型的 GFLOPs（十亿浮点运算次数）。
    """
    if not TORCH_2_0:  # torch profiler 在 torch>=2.0 中实现
        return 0.0
    model = unwrap_model(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # 若为 int/float 则扩展
    try:
        # 使用步幅尺寸作为输入张量
        stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2  # 最大步幅
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # 输入图像，BCHW 格式
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
        flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception:
        # 使用实际图像尺寸作为输入张量（如 RTDETR 模型需要）
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # 输入图像，BCHW 格式
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
    return flops


def initialize_weights(model):
    """将模型权重、偏置和模块设置初始化为默认值。"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """缩放和填充图像张量，可选择保持纵横比并填充到 gs 的倍数。

    Args:
        img (torch.Tensor): 输入图像张量。
        ratio (float, optional): 缩放比例。
        same_shape (bool, optional): 是否保持相同形状。
        gs (int, optional): 填充的网格大小。

    Returns:
        (torch.Tensor): 缩放和填充后的图像张量。
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # 新尺寸
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # 调整大小
    if not same_shape:  # 填充/裁剪图像
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # 值 = ImageNet 均值


def copy_attr(a, b, include=(), exclude=()):
    """将对象 'b' 的属性复制到对象 'a'，可选择包含/排除某些属性。

    Args:
        a (Any): 复制属性的目标对象。
        b (Any): 复制属性的源对象。
        include (tuple, optional): 要包含的属性。若为空，则包含所有属性。
        exclude (tuple, optional): 要排除的属性。
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def intersect_dicts(da, db, exclude=()):
    """返回具有匹配形状的交集键的字典，排除 'exclude' 键，使用 da 的值。

    Args:
        da (dict): 第一个字典。
        db (dict): 第二个字典。
        exclude (tuple, optional): 要排除的键。

    Returns:
        (dict): 具有匹配形状的交集键字典。
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def is_parallel(model):
    """如果模型为 DP 或 DDP 类型则返回 True。

    Args:
        model (nn.Module): 要检查的模型。

    Returns:
        (bool): 若模型为 DataParallel 或 DistributedDataParallel 则为 True。
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def unwrap_model(m: nn.Module) -> nn.Module:
    """解包编译和并行模型以获取基础模型。

    Args:
        m (nn.Module): 可能被 torch.compile (._orig_mod) 或并行包装器（如
            DataParallel/DistributedDataParallel (.module)）包装的模型。

    Returns:
        (nn.Module): 无编译或并行包装器的基础模型。
    """
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, nn.Module):
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(m.module, nn.Module):
            m = m.module
        else:
            return m


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """返回一个正弦斜坡的 lambda 函数，从 y1 到 y2 https://arxiv.org/pdf/1812.01187.pdf。

    Args:
        y1 (float, optional): 初始值。
        y2 (float, optional): 最终值。
        steps (int, optional): 步数。

    Returns:
        (function): 计算正弦斜坡的 lambda 函数。
    """
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def init_seeds(seed=0, deterministic=False):
    """初始化随机数生成器（RNG）种子 https://pytorch.org/docs/stable/notes/randomness.html。

    Args:
        seed (int, optional): 随机种子。
        deterministic (bool, optional): 是否设置确定性算法。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU，异常安全
    # torch.backends.cudnn.benchmark = True  # AutoBatch 问题 https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)  # 如果无法确定性则警告
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            LOGGER.warning("Upgrade to torch>=2.0.0 for deterministic training.")
    else:
        unset_deterministic()


def unset_deterministic():
    """取消所有用于确定性训练的配置。"""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop("PYTHONHASHSEED", None)


class ModelEMA:
    """更新的指数移动平均（EMA）实现。

    保持模型 state_dict（参数和缓冲区）中所有内容的移动平均。EMA 详见参考文献。

    要禁用 EMA，将 `enabled` 属性设为 `False`。

    Attributes:
        ema (nn.Module): 评估模式下模型的副本。
        updates (int): EMA 更新次数。
        decay (function): 决定 EMA 权重的衰减函数。
        enabled (bool): 是否启用 EMA。

    References:
        - https://github.com/rwightman/pytorch-image-models
        - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """使用给定参数为 'model' 初始化 EMA。

        Args:
            model (nn.Module): 要创建 EMA 的模型。
            decay (float, optional): 最大 EMA 衰减率。
            tau (int, optional): EMA 衰减时间常数。
            updates (int, optional): 初始更新次数。
        """
        self.ema = deepcopy(unwrap_model(model)).eval()  # FP32 EMA
        self.updates = updates  # EMA 更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # 指数衰减斜坡（帮助早期训练）
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """更新 EMA 参数。

        Args:
            model (nn.Module): 更新 EMA 的模型。
        """
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = unwrap_model(model).state_dict()  # 模型 state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # 对 FP16 和 FP32 为真
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """将模型属性复制到 EMA，可选择包含/排除某些属性。

        Args:
            model (nn.Module): 复制属性的源模型。
            include (tuple, optional): 要包含的属性。
            exclude (tuple, optional): 要排除的属性。
        """
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def strip_optimizer(f: str | Path = "best.pt", s: str = "", updates: dict[str, Any] | None = None) -> dict[str, Any]:
    """从 'f' 中去除优化器以完成训练，可选保存为 's'。

    Args:
        f (str | Path): 要去除优化器的模型文件路径。
        s (str, optional): 保存去除优化器后模型的文件路径。若未提供，将覆盖 'f'。
        updates (dict, optional): 保存前覆盖到检查点的更新字典。

    Returns:
        (dict): 合并后的检查点字典。

    Examples:
        >>> from pathlib import Path
        >>> from ultralytics.utils.torch_utils import strip_optimizer
        >>> for f in Path("path/to/model/checkpoints").rglob("*.pt"):
        ...     strip_optimizer(f)
    """
    try:
        x = torch_load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
        return {}

    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # 更新模型
    if x.get("ema"):
        x["model"] = x["ema"]  # 用 EMA 替换模型
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # 从 IterableSimpleNamespace 转为 dict
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # 去除损失准则
    x["model"].half()  # 转为 FP16
    for p in x["model"].parameters():
        p.requires_grad = False

    # 更新其他键
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # 合并参数
    for k in "optimizer", "best_fitness", "ema", "updates", "scaler":  # 键
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # 去除非默认键
    # x['model'].args = x['train_args']

    # 保存
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)  # 合并字典（右侧优先）
    mb = os.path.getsize(s or f) / 1e6  # 文件大小
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined


def convert_optimizer_state_dict_to_fp16(state_dict):
    """将给定优化器的 state_dict 转换为 FP16，专注于 'state' 键的张量转换。

    Args:
        state_dict (dict): 优化器状态字典。

    Returns:
        (dict): 转换后的 FP16 张量优化器状态字典。
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k not in {"step", "exp_avg_sq"} and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict


@contextmanager
def cuda_memory_usage(device=None):
    """监控和管理 CUDA 内存使用。

    此函数检查 CUDA 是否可用，若可用则清空 CUDA 缓存以释放未使用的内存。然后
    生成一个包含内存使用信息的字典，调用者可更新。最后，用指定设备上 CUDA 保留的
    内存量更新字典。

    Args:
        device (torch.device, optional): 要查询内存使用情况的 CUDA 设备。

    Yields:
        (dict): 包含键 'memory'（初始化为 0）的字典，将用保留内存更新。
    """
    cuda_info = dict(memory=0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            yield cuda_info
        finally:
            cuda_info["memory"] = torch.cuda.memory_reserved(device)
    else:
        yield cuda_info


def profile_ops(input, ops, n=10, device=None, max_num_obj=0):
    """Ultralytics 速度、内存和 FLOPs 分析器。

    Args:
        input (torch.Tensor | list): 要分析的输入张量。
        ops (nn.Module | list): 要分析的模型或操作列表。
        n (int, optional): 平均迭代次数。
        device (str | torch.device, optional): 分析所用的设备。
        max_num_obj (int, optional): 模拟的最大目标数。

    Returns:
        (list): 每个操作的分析结果。

    Examples:
        >>> from ultralytics.utils.torch_utils import profile_ops
        >>> input = torch.randn(16, 3, 640, 640)
        >>> m1 = lambda x: x * torch.sigmoid(x)
        >>> m2 = nn.SiLU()
        >>> profile_ops(input, [m1, m2], n=100)  # profile over 100 iterations
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda 环境未安装 'ultralytics-thop'

    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )
    gc.collect()  # 尝试释放未使用的内存
    torch.cuda.empty_cache()
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # 设备
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # 前向、反向耗时
            try:
                flops = thop.profile(deepcopy(m), inputs=[x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
            except Exception:
                flops = 0

            try:
                mem = 0
                for _ in range(n):
                    with cuda_memory_usage(device) as cuda_info:
                        t[0] = time_sync()
                        y = m(x)
                        t[1] = time_sync()
                        try:
                            (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                            t[2] = time_sync()
                        except Exception:  # 无反向方法
                            # print(e)  # 调试用
                            t[2] = float("nan")
                    mem += cuda_info["memory"] / 1e9  # (GB)
                    tf += (t[1] - t[0]) * 1000 / n  # 每次操作前向耗时（ms）
                    tb += (t[2] - t[1]) * 1000 / n  # 每次操作反向耗时（ms）
                    if max_num_obj:  # 模拟每图像网格预测的训练（用于 AutoBatch）
                        with cuda_memory_usage(device) as cuda_info:
                            torch.randn(
                                x.shape[0],
                                max_num_obj,
                                int(sum((x.shape[-1] / s) * (x.shape[-2] / s) for s in m.stride.tolist())),
                                device=device,
                                dtype=torch.float32,
                            )
                        mem += cuda_info["memory"] / 1e9  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # 形状
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # 参数
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{s_in!s:>24s}{s_out!s:>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            finally:
                gc.collect()  # 尝试释放未使用的内存
                torch.cuda.empty_cache()
    return results


class EarlyStopping:
    """早停类，当指定轮数无改进时停止训练。

    Attributes:
        best_fitness (float): 观察到的最佳适应度值。
        best_epoch (int): 观察到最佳适应度的轮次。
        patience (int): 适应度停止改进后等待停止的轮数。
        possible_stop (bool): 指示下一轮是否可能停止的标志。
    """

    def __init__(self, patience=50):
        """初始化早停对象。

        Args:
            patience (int, optional): 适应度停止改进后等待停止的轮数。
        """
        self.best_fitness = 0.0  # 即 mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # 适应度停止改进后等待停止的轮数
        self.possible_stop = False  # 下一轮可能停止

    def __call__(self, epoch, fitness):
        """检查是否应停止训练。

        Args:
            epoch (int): 当前训练轮次。
            fitness (float): 当前轮次的适应度值。

        Returns:
            (bool): 若应停止训练则为 True，否则为 False。
        """
        if fitness is None:  # 检查 fitness=None（val=False 时发生）
            return False

        if fitness > self.best_fitness or self.best_fitness == 0:  # 允许早期零适应度训练阶段
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # 无改进的轮数
        self.possible_stop = delta >= (self.patience - 1)  # 下一轮可能停止
        stop = delta >= self.patience  # 若超过耐心值则停止训练
        if stop:
            prefix = colorstr("EarlyStopping: ")
            LOGGER.info(
                f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
            )
        return stop


def attempt_compile(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int = 640,
    use_autocast: bool = False,
    warmup: bool = False,
    mode: bool | str = "default",
) -> torch.nn.Module:
    """使用 torch.compile 编译模型，可选预热图以减少首次迭代延迟。

    此工具尝试使用 inductor 后端编译提供的模型。如果编译不可用或失败，返回原始模型不变。
    可选预热在虚拟输入上执行单次前向传播，以初始化编译图并测量编译/预热时间。

    Args:
        model (torch.nn.Module): 要编译的模型。
        device (torch.device): 用于预热和 autocast 决策的推理设备。
        imgsz (int, optional): 创建形状 (1, 3, imgsz, imgsz) 虚拟张量的方形输入尺寸，用于预热。
        use_autocast (bool, optional): 是否在 CUDA 或 MPS 设备上以 autocast 运行预热。
        warmup (bool, optional): 是否执行单次虚拟前向传播以预热编译模型。
        mode (bool | str, optional): torch.compile 模式。True → "default"，False → 不编译，或字符串如
            "default"、"reduce-overhead"、"max-autotune-no-cudagraphs"。

    Returns:
        (torch.nn.Module): 若编译成功则返回编译后模型，否则返回原始未修改模型。

    Examples:
        >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        >>> # Try to compile and warm up a model with a 640x640 input
        >>> model = attempt_compile(model, device=device, imgsz=640, use_autocast=True, warmup=True)

    Notes:
        - 如果当前 PyTorch 版本不提供 torch.compile，函数立即返回输入模型。
        - 预热在 torch.inference_mode 下运行，可在 CUDA/MPS 上使用 torch.autocast 以对齐计算精度。
        - 预热后同步 CUDA 设备以处理异步内核执行。
    """
    if not hasattr(torch, "compile") or not mode:
        return model

    if mode is True:
        mode = "default"
    prefix = colorstr("compile:")
    LOGGER.info(f"{prefix} starting torch.compile with '{mode}' mode...")
    if mode == "max-autotune":
        LOGGER.warning(f"{prefix} mode='{mode}' not recommended, using mode='max-autotune-no-cudagraphs' instead")
        mode = "max-autotune-no-cudagraphs"
    t0 = time.perf_counter()
    try:
        model = torch.compile(model, mode=mode, backend="inductor")
    except Exception as e:
        LOGGER.warning(f"{prefix} torch.compile failed, continuing uncompiled: {e}")
        return model
    t_compile = time.perf_counter() - t0

    t_warm = 0.0
    if warmup:
        # 使用单个虚拟张量构建图形状状态，减少首次迭代延迟
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        if use_autocast and device.type == "cuda":
            dummy = dummy.half()
        t1 = time.perf_counter()
        with torch.inference_mode():
            if use_autocast and device.type in {"cuda", "mps"}:
                with torch.autocast(device.type):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_warm = time.perf_counter() - t1

    total = t_compile + t_warm
    if warmup:
        LOGGER.info(f"{prefix} complete in {total:.1f}s (compile {t_compile:.1f}s + warmup {t_warm:.1f}s)")
    else:
        LOGGER.info(f"{prefix} compile complete in {t_compile:.1f}s (no warmup)")
    return model
