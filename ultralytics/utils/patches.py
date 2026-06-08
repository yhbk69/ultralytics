# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""用于更新/扩展现有函数功能的猴子补丁。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

# OpenCV 多语言友好函数 --------------------------------------------------------------------------------------------
_imshow = cv2.imshow  # 复制以避免递归错误


def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读取图像文件，支持多语言文件名。

    参数:
        filename (str): 要读取的文件路径。
        flags (int, 可选): 可取 cv2.IMREAD_* 值的标志，控制图像读取方式。

    返回:
        (np.ndarray | None): 读取的图像数组，如果读取失败则返回 None。

    示例:
        >>> img = imread("path/to/image.jpg")
        >>> img = imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)
    """
    try:
        file_bytes = np.fromfile(filename, np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            # 处理多帧 TIFF 和彩色图像
            return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
        return None
    else:
        im = cv2.imdecode(file_bytes, flags)
        # 回退处理 OpenCV imdecode 可能不支持的格式（AVIF, HEIC）
        if im is None and filename.lower().endswith((".avif", ".heic")):
            im = _imread_pil(filename, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im  # 始终确保 3 维


# PIL 补丁 ---------------------------------------------------------------------------------------------------------
_image_open = Image.open  # 复制以避免递归错误
_pil_plugins_registered = False


def image_open(filename, *args, **kwargs):
    """使用 PIL 打开图像，在首次失败时延迟注册 HEIF 插件。

    此猴子补丁为 PIL.Image.open 添加了通过 pi-heif（轻量级，仅解码）的 HEIC/HEIF 支持，
    避免了导入该包约 800ms 的启动开销，除非确实需要。
    AVIF 由 Pillow 12+ 原生支持，不需要插件。

    参数:
        filename (str): 图像文件路径。
        *args (Any): 传递给 PIL.Image.open 的额外位置参数。
        **kwargs (Any): 传递给 PIL.Image.open 的额外关键字参数。

    返回:
        (PIL.Image.Image): 打开的 PIL 图像。
    """
    global _pil_plugins_registered
    if _pil_plugins_registered:
        return _image_open(filename, *args, **kwargs)
    try:
        return _image_open(filename, *args, **kwargs)
    except Exception:
        from ultralytics.utils.checks import check_requirements

        check_requirements("pi-heif")
        from pi_heif import register_heif_opener

        register_heif_opener()
        _pil_plugins_registered = True
        return _image_open(filename, *args, **kwargs)


Image.open = image_open  # 应用补丁


def _imread_pil(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """使用 PIL 读取图像，作为不支持格式的回退。

    参数:
        filename (str): 要读取的文件路径。
        flags (int, 可选): OpenCV imread 标志（用于确定灰度转换）。

    返回:
        (np.ndarray | None): 以 BGR 格式读取的图像数组，如果读取失败则返回 None。
    """
    try:
        with Image.open(filename) as img:
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.asarray(img.convert("L"))
            return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def imwrite(filename: str, img: np.ndarray, params: list[int] | None = None) -> bool:
    """将图像写入文件，支持多语言文件名。

    参数:
        filename (str): 要写入的文件路径。
        img (np.ndarray): 要写入的图像。
        params (list[int], 可选): 图像编码的额外参数。

    返回:
        (bool): 如果文件成功写入则返回 True，否则返回 False。

    示例:
        >>> import numpy as np
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)  # 创建黑色图像
        >>> success = imwrite("output.jpg", img)  # 将图像写入文件
        >>> print(success)
        True
    """
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False


def imshow(winname: str, mat: np.ndarray) -> None:
    """在指定窗口中显示图像，支持多语言窗口名。

    该函数是 OpenCV imshow 函数的包装器，在命名窗口中显示图像。
    它通过正确编码窗口名来处理多语言窗口名，以确保 OpenCV 兼容性。

    参数:
        winname (str): 显示图像的窗口名称。如果已存在同名窗口，则在该窗口中显示。
        mat (np.ndarray): 要显示的图像。应为有效的 numpy 图像数组。

    示例:
        >>> import numpy as np
        >>> img = np.zeros((300, 300, 3), dtype=np.uint8)  # 创建黑色图像
        >>> img[:100, :100] = [255, 0, 0]  # 添加蓝色方块
        >>> imshow("Example Window", img)  # 显示图像
    """
    _imshow(winname.encode("unicode_escape").decode(), mat)


# PyTorch 函数 ----------------------------------------------------------------------------------------------------
_torch_save = torch.save


def torch_load(*args, **kwargs):
    """加载 PyTorch 模型，更新参数以避免警告。

    该函数包装 torch.load，并为 PyTorch 1.13.0+ 添加 'weights_only' 参数以防止警告。

    参数:
        *args (Any): 传递给 torch.load 的可变长度参数列表。
        **kwargs (Any): 传递给 torch.load 的任意关键字参数。

    返回:
        (Any): 加载的 PyTorch 对象。

    注意:
        对于 PyTorch 1.13 及以上版本，如果未提供该参数，此函数会自动设置 `weights_only=False`，
        以避免弃用警告。
    """
    from ultralytics.utils.torch_utils import TORCH_1_13

    if TORCH_1_13 and "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return torch.load(*args, **kwargs)


def torch_save(*args, **kwargs):
    """保存 PyTorch 对象，具有重试机制以确保健壮性。

    该函数包装 torch.save，在保存失败时进行 3 次重试和指数退避，
    失败可能是由于设备刷新延迟或杀毒软件扫描造成的。

    参数:
        *args (Any): 传递给 torch.save 的位置参数。
        **kwargs (Any): 传递给 torch.save 的关键字参数。

    示例:
        >>> model = torch.nn.Linear(10, 1)
        >>> torch_save(model.state_dict(), "model.pt")
    """
    for i in range(4):  # 3 次重试
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:  # 无法保存，可能等待设备刷新或杀毒软件扫描
            if i == 3:
                raise e
            time.sleep((2**i) / 2)  # 指数退避: 0.5s, 1.0s, 2.0s


@contextmanager
def arange_patch(dynamic: bool = False, half: bool = False, fmt: str = ""):
    """ONNX 中 torch.arange 与 FP16 不兼容的变通方案。

    https://github.com/pytorch/pytorch/issues/148041。
    """
    if dynamic and half and fmt == "onnx":
        func = torch.arange

        def arange(*args, dtype=None, **kwargs):
            """包装 torch.arange，在创建后转换 dtype 而不是直接传递。"""
            return func(*args, **kwargs).to(dtype)  # 转换为 dtype 而不是传递 dtype

        torch.arange = arange  # 打补丁
        yield
        torch.arange = func  # 取消补丁
    else:
        yield


@contextmanager
def onnx_export_patch():
    """PyTorch 2.9+ 启用 Dynamo 时 ONNX 导出问题的变通方案。"""
    from ultralytics.utils.torch_utils import TORCH_2_9

    if TORCH_2_9:
        func = torch.onnx.export

        def torch_export(*args, **kwargs):
            """禁用 Dynamo 导出模型为 ONNX 格式以确保兼容性。"""
            return func(*args, **kwargs, dynamo=False)

        torch.onnx.export = torch_export  # 打补丁
        yield
        torch.onnx.export = func  # 取消补丁
    else:
        yield


@contextmanager
def override_configs(args, overrides: dict[str, Any] | None = None):
    """临时覆盖 args 中配置的上下文管理器。

    参数:
        args (IterableSimpleNamespace): 原始配置参数。
        overrides (dict[str, Any] | None): 要应用的覆盖字典。

    生成:
        (IterableSimpleNamespace): 应用了覆盖的配置参数。
    """
    if overrides:
        original_args = copy(args)
        for key, value in overrides.items():
            setattr(args, key, value)
        try:
            yield args
        finally:
            args.__dict__.update(original_args.__dict__)
    else:
        yield args
