# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import NOT_MACOS14


class Profile(contextlib.ContextDecorator):
    """Ultralytics Profile 类，用于代码执行计时。

    用作装饰器 @Profile() 或上下文管理器 'with Profile():'。提供精确的计时测量，
    支持用于 GPU 操作的 CUDA 同步。

    属性:
        t (float): 累计时间（秒）。
        device (torch.device): 模型推理使用的设备。
        cuda (bool): 是否使用 CUDA 进行计时同步。

    Examples:
        Use as a context manager to time code execution
        >>> with Profile(device=device) as dt:
        ...     pass  # 此处为慢操作
        >>> print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"

        Use as a decorator to time function execution
        >>> @Profile()
        ... def slow_function():
        ...     time.sleep(0.1)
    """

    def __init__(self, t: float = 0.0, device: torch.device | None = None):
        """初始化 Profile 类。

        Args:
            t (float): 初始累计时间（秒）。
            device (torch.device, optional): 模型推理使用的设备，以启用 CUDA 同步。
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """开始计时。"""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """停止计时。"""
        self.dt = self.time() - self.start  # 增量时间
        self.t += self.dt  # 累加 dt

    def __str__(self):
        """返回表示累计耗时的人类可读字符串。"""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """如适用，获取经 CUDA 同步的当前时间。"""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()


def segment2box(segment: np.ndarray, width: int = 640, height: int = 640) -> np.ndarray:
    """将分割坐标转换为边界框坐标。

    通过找到 x 和 y 坐标的最小值和最大值，将单个分割标签转换为边界框标签。必要时应用图像内约束并裁剪坐标。

    Args:
        segment (np.ndarray): 分割坐标，格式 (N, 2)，N 为点数。
        width (int): 图像宽度（像素）。
        height (int): 图像高度（像素）。

    Returns:
        (np.ndarray): xyxy 格式的边界框坐标 [x1, y1, x2, y2]。
    """
    x, y = segment.T  # 分割 xy
    # 如果 4 条边中有 3 条超出图像则裁剪坐标
    if np.array([x.min() < 0, y.min() < 0, x.max() > width, y.max() > height]).sum() >= 3:
        x = x.clip(0, width)
        y = y.clip(0, height)
    inside = (x > 0) & (y > 0) & (x < width) & (y < height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: torch.Tensor | np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple | None = None,
    padding: bool = True,
    xywh: bool = False,
) -> torch.Tensor | np.ndarray:
    """将边界框从一个图像形状缩放到另一个图像形状。

    将边界框从 img1_shape 缩放到 img0_shape，考虑填充和宽高比变化。支持 xyxy 和 xywh 两种框格式。

    Args:
        img1_shape (tuple[int, int]): 源图像形状（高度，宽度）。
        boxes (torch.Tensor | np.ndarray): 要缩放的边界框，格式 (N, 4)。
        img0_shape (tuple[int, int]): 目标图像形状（高度，宽度）。
        ratio_pad (tuple, optional): 用于缩放的 (ratio, pad) 元组。若为 None，则从图像形状计算。
        padding (bool): 边界框是否基于带填充的 YOLO 风格增强图像。
        xywh (bool): 框格式是否为 xywh（True）或 xyxy（False）。

    Returns:
        (torch.Tensor | np.ndarray): 与输入格式相同的缩放后边界框。
    """
    if ratio_pad is None:  # 从 img0_shape 计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 增益 = 旧 / 新
        pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
        pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x  # x 填充
        boxes[..., 1] -= pad_y  # y 填充
        if not xywh:
            boxes[..., 2] -= pad_x  # x 填充
            boxes[..., 3] -= pad_y  # y 填充
    boxes[..., :4] /= gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def make_divisible(x: int, divisor):
    """返回最接近的可被给定除数整除的数。

    Args:
        x (int): 要使其可整除的数。
        divisor (int | torch.Tensor): 除数。

    Returns:
        (int): 最接近的可被除数整除的数。
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # 转为 int
    return math.ceil(x / divisor) * divisor


def clip_boxes(boxes, shape):
    """将边界框裁剪到图像边界内。

    Args:
        boxes (torch.Tensor | np.ndarray): 要裁剪的边界框。
        shape (tuple): 图像形状，HWC 或 HW 格式（两者均支持）。

    Returns:
        (torch.Tensor | np.ndarray): 裁剪后的边界框。
    """
    h, w = shape[:2]  # 支持 HWC 或 HW 形状
    if isinstance(boxes, torch.Tensor):  # 逐个操作更快
        if NOT_MACOS14:
            boxes[..., 0].clamp_(0, w)  # x1
            boxes[..., 1].clamp_(0, h)  # y1
            boxes[..., 2].clamp_(0, w)  # x2
            boxes[..., 3].clamp_(0, h)  # y2
        else:  # Apple macOS14 MPS 错误 https://github.com/ultralytics/ultralytics/pull/21878
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(0, w)
            boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:  # np.array（分组操作更快）
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """将线坐标裁剪到图像边界内。

    Args:
        coords (torch.Tensor | np.ndarray): 要裁剪的线坐标。
        shape (tuple): 图像形状，HWC 或 HW 格式（两者均支持）。

    Returns:
        (torch.Tensor | np.ndarray): 裁剪后的坐标。
    """
    h, w = shape[:2]  # 支持 HWC 或 HW 形状
    if isinstance(coords, torch.Tensor):
        if NOT_MACOS14:
            coords[..., 0].clamp_(0, w)  # x
            coords[..., 1].clamp_(0, h)  # y
        else:  # Apple macOS14 MPS 错误 https://github.com/ultralytics/ultralytics/pull/21878
            coords[..., 0] = coords[..., 0].clamp(0, w)
            coords[..., 1] = coords[..., 1].clamp(0, h)
    else:  # np.array
        coords[..., 0] = coords[..., 0].clip(0, w)  # x
        coords[..., 1] = coords[..., 1].clip(0, h)  # y
    return coords


def xyxy2xywh(x):
    """将边界框坐标从 (x1, y1, x2, y2) 格式转换为 (x, y, width, height) 格式，其中 (x1, y1) 为左上角，
    (x2, y2) 为右下角。

    Args:
        x (np.ndarray | torch.Tensor): (x1, y1, x2, y2) 格式的输入边界框坐标。

    Returns:
        (np.ndarray | torch.Tensor): (x, y, width, height) 格式的边界框坐标。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 比 clone/copy 更快
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x 中心
    y[..., 1] = (y1 + y2) / 2  # y 中心
    y[..., 2] = x2 - x1  # 宽度
    y[..., 3] = y2 - y1  # 高度
    return y


def xywh2xyxy(x):
    """将边界框坐标从 (x, y, width, height) 格式转换为 (x1, y1, x2, y2) 格式，其中 (x1, y1) 为左上角，
    (x2, y2) 为右下角。注意：每 2 通道操作比逐通道更快。

    Args:
        x (np.ndarray | torch.Tensor): (x, y, width, height) 格式的输入边界框坐标。

    Returns:
        (np.ndarray | torch.Tensor): (x1, y1, x2, y2) 格式的边界框坐标。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 比 clone/copy 更快
    xy = x[..., :2]  # 中心
    wh = x[..., 2:] / 2  # 半宽高
    y[..., :2] = xy - wh  # 左上 xy
    y[..., 2:] = xy + wh  # 右下 xy
    return y


def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0):
    """将归一化的边界框坐标转换为像素坐标。

    Args:
        x (np.ndarray | torch.Tensor): (x, y, w, h) 格式的归一化边界框坐标。
        w (int): 图像宽度（像素）。
        h (int): 图像高度（像素）。
        padw (int): 填充宽度（像素）。
        padh (int): 填充高度（像素）。

    Returns:
        (np.ndarray | torch.Tensor): (x1, y1, x2, y2) 格式的边界框坐标。
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 比 clone/copy 更快
    xc, yc, xw, xh = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    half_w, half_h = xw / 2, xh / 2
    y[..., 0] = w * (xc - half_w) + padw  # 左上 x
    y[..., 1] = h * (yc - half_h) + padh  # 左上 y
    y[..., 2] = w * (xc + half_w) + padw  # 右下 x
    y[..., 3] = h * (yc + half_h) + padh  # 右下 y
    return y


def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    """将边界框坐标从 (x1, y1, x2, y2) 格式转换为归一化的 (x, y, width, height) 格式。x、y、
    width 和 height 均归一化到图像尺寸。

    Args:
        x (np.ndarray | torch.Tensor): (x1, y1, x2, y2) 格式的输入边界框坐标。
        w (int): 图像宽度（像素）。
        h (int): 图像高度（像素）。
        clip (bool): 是否将框裁剪到图像边界内。
        eps (float): 框宽度和高度的最小值。

    Returns:
        (np.ndarray | torch.Tensor): (x, y, width, height) 格式的归一化边界框坐标。
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # 比 clone/copy 更快
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = ((x1 + x2) / 2) / w  # x 中心
    y[..., 1] = ((y1 + y2) / 2) / h  # y 中心
    y[..., 2] = (x2 - x1) / w  # 宽度
    y[..., 3] = (y2 - y1) / h  # 高度
    return y


def xywh2ltwh(x):
    """将边界框格式从 [x, y, w, h] 转换为 [x1, y1, w, h]，其中 x1、y1 为左上角坐标。

    Args:
        x (np.ndarray | torch.Tensor): xywh 格式的输入边界框坐标。

    Returns:
        (np.ndarray | torch.Tensor): ltwh 格式的边界框坐标。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # 左上 x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # 左上 y
    return y


def xyxy2ltwh(x):
    """将边界框从 [x1, y1, x2, y2] 转换为 [x1, y1, w, h] 格式。

    Args:
        x (np.ndarray | torch.Tensor): xyxy 格式的输入边界框坐标。

    Returns:
        (np.ndarray | torch.Tensor): ltwh 格式的边界框坐标。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # 宽度
    y[..., 3] = x[..., 3] - x[..., 1]  # 高度
    return y


def ltwh2xywh(x):
    """将边界框从 [x1, y1, w, h] 转换为 [x, y, w, h]，其中 xy1=左上角，xy=中心。

    Args:
        x (np.ndarray | torch.Tensor): 输入边界框坐标。

    Returns:
        (np.ndarray | torch.Tensor): xywh 格式的边界框坐标。
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # 中心 x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # 中心 y
    return y


def xyxyxyxy2xywhr(x):
    """将批量定向边界框（OBB）从 [xy1, xy2, xy3, xy4] 转换为 [xywh, rotation] 格式。

    Args:
        x (np.ndarray | torch.Tensor): [xy1, xy2, xy3, xy4] 格式的输入框角点，形状 (N, 8)。

    Returns:
        (np.ndarray | torch.Tensor): 转换后的 [cx, cy, w, h, rotation] 格式数据，形状 (N, 5)。旋转值以弧度表示，范围 [-pi/4, 3pi/4)。
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # 注意：使用 cv2.minAreaRect 获取精确的 xywhr，
        # 尤其是某些目标被数据加载器中的增强操作截断。
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        # 将角度转换为弧度并归一化到 [-pi/4, 3pi/4)
        theta = angle / 180 * np.pi
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rboxes.append([cx, cy, w, h, theta])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """将批量定向边界框（OBB）从 [xywh, rotation] 转换为 [xy1, xy2, xy3, xy4] 格式。

    Args:
        x (np.ndarray | torch.Tensor): [cx, cy, w, h, rotation] 格式的框，形状 (N, 5) 或 (B, N, 5)。旋转值应以弧度表示，范围 [-pi/4, 3pi/4)。

    Returns:
        (np.ndarray | torch.Tensor): 转换后的角点，形状 (N, 4, 2) 或 (B, N, 4, 2)。
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """将边界框从 [x1, y1, w, h] 转换为 [x1, y1, x2, y2]，其中 xy1=左上角，xy2=右下角。

    Args:
        x (np.ndarray | torch.Tensor): 输入边界框坐标。

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyxy format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # x2
    y[..., 3] = x[..., 3] + x[..., 1]  # y2
    return y


def segments2boxes(segments):
    """将分割坐标转换为 xywh 格式的边界框标签。

    Args:
        segments (list): 分割列表，每个分割为点列表，每个点为 [x, y] 坐标。

    Returns:
        (np.ndarray): xywh 格式的边界框坐标。
    """
    boxes = []
    for s in segments:
        x, y = s.T  # 分割 xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # 类别、xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n: int = 1000):
    """使用线性插值将分割重采样为 n 个点。

    Args:
        segments (list): (N, 2) 数组列表，N 为每个分割中的点数。
        n (int): 每个分割重采样的点数。

    Returns:
        (list): 每个分割 n 个点的重采样分割。
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n - len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # 分割 xy
    return segments


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """将掩码裁剪到边界框区域。

    Args:
        masks (torch.Tensor): 形状 (N, H, W) 的掩码。
        boxes (torch.Tensor): xyxy 像素格式的边界框坐标，形状 (N, 4)。

    Returns:
        (torch.Tensor): 裁剪后的掩码。
    """
    if boxes.device != masks.device:
        boxes = boxes.to(masks.device)
    n, h, w = masks.shape
    if n < 50 and not masks.is_cuda:  # 掩码较少时更快（预测）
        for i, (x1, y1, x2, y2) in enumerate(boxes.clamp(min=0).round().int()):
            masks[i, :y1] = 0
            masks[i, y2:] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks
    else:  # 掩码较多时更快（验证）
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 形状(n,1,1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # 行 shape(1,1,w)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # 列 shape(1,h,1)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """使用掩码头输出将掩码应用到边界框。

    Args:
        protos (torch.Tensor): 掩码原型，形状 (mask_dim, mask_h, mask_w)。
        masks_in (torch.Tensor): 掩码系数，形状 (N, mask_dim)，N 为 NMS 后的掩码数量。
        bboxes (torch.Tensor): 边界框，形状 (N, 4)，N 为 NMS 后的掩码数量。
        shape (tuple): 输入图像尺寸 (height, width)。
        upsample (bool): 是否将掩码上采样到原始图像尺寸。

    Returns:
        (torch.Tensor): 形状 [n, h, w] 的二值掩码张量，n 为 NMS 后的掩码数量，h 和 w 为输入图像的高宽。掩码已应用到边界框。
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # NHW

    width_ratio = mw / shape[1]
    height_ratio = mh / shape[0]
    ratios = torch.tensor([[width_ratio, height_ratio, width_ratio, height_ratio]], device=bboxes.device)

    masks = crop_mask(masks, boxes=bboxes * ratios)  # NHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear")[0]  # NHW
    return masks.gt_(0.0).byte()


def process_mask_native(protos, masks_in, bboxes, shape):
    """使用掩码头输出和原生上采样将掩码应用到边界框。

    Args:
        protos (torch.Tensor): 掩码原型，形状 (mask_dim, mask_h, mask_w)。
        masks_in (torch.Tensor): 掩码系数，形状 (N, mask_dim)，N 为 NMS 后的掩码数量。
        bboxes (torch.Tensor): 边界框，形状 (N, 4)，N 为 NMS 后的掩码数量。
        shape (tuple): 输入图像尺寸 (height, width)。

    Returns:
        (torch.Tensor): 形状 (N, H, W) 的二值掩码张量。
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # NHW
    masks = crop_mask(masks, bboxes)  # NHW
    return masks.gt_(0.0).byte()


def scale_masks(
    masks: torch.Tensor,
    shape: tuple[int, int],
    ratio_pad: tuple[tuple[int, int], tuple[int, int]] | None = None,
    padding: bool = True,
) -> torch.Tensor:
    """将分割掩码缩放到目标形状。

    Args:
        masks (torch.Tensor): 形状 (N, C, H, W) 的掩码。
        shape (tuple[int, int]): 目标高度和宽度 (height, width)。
        ratio_pad (tuple, optional): 比率和填充值，格式 ((ratio_h, ratio_w), (pad_w, pad_h))。
        padding (bool): 掩码是否基于带填充的 YOLO 风格增强图像。

    Returns:
        (torch.Tensor): 缩放后的掩码。
    """
    im1_h, im1_w = masks.shape[2:]
    im0_h, im0_w = shape[:2]
    if im1_h == im0_h and im1_w == im0_w:
        return masks

    if ratio_pad is None:  # 从 im0_shape 计算
        gain = min(im1_h / im0_h, im1_w / im0_w)  # 增益 = 旧 / 新
        pad_w, pad_h = (im1_w - round(im0_w * gain)), (im1_h - round(im0_h * gain))  # 宽高填充
        if padding:
            pad_w /= 2
            pad_h /= 2
    else:
        pad_w, pad_h = ratio_pad[1]
    top, left = (round(pad_h - 0.1), round(pad_w - 0.1)) if padding else (0, 0)
    bottom = im1_h - round(pad_h + 0.1)
    right = im1_w - round(pad_w + 0.1)
    return F.interpolate(masks[..., top:bottom, left:right].float(), shape, mode="bilinear")  # NCHW 掩码


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize: bool = False, padding: bool = True):
    """将分割坐标从 img1_shape 缩放到 img0_shape。

    Args:
        img1_shape (tuple): 源图像形状，HWC 或 HW 格式（两者均支持）。
        coords (torch.Tensor): 要缩放的坐标，形状 (N, 2)。
        img0_shape (tuple): 图像 0 的形状，HWC 或 HW 格式（两者均支持）。
        ratio_pad (tuple, optional): 比率和填充值，格式 ((ratio_h, ratio_w), (pad_w, pad_h))。
        normalize (bool): 是否将坐标归一化到 [0, 1] 范围。
        padding (bool): 坐标是否基于带填充的 YOLO 风格增强图像。

    Returns:
        (torch.Tensor): 缩放后的坐标。
    """
    img0_h, img0_w = img0_shape[:2]  # 支持 HWC 或 HW 形状
    if ratio_pad is None:  # 从 img0_shape 计算
        img1_h, img1_w = img1_shape[:2]  # 支持 HWC 或 HW 形状
        gain = min(img1_h / img0_h, img1_w / img0_w)  # 增益 = 旧 / 新
        pad = (img1_w - round(img0_w * gain)) / 2, (img1_h - round(img0_h * gain)) / 2  # 宽高填充
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x 填充
        coords[..., 1] -= pad[1]  # y 填充
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_w  # 宽度
        coords[..., 1] /= img0_h  # 高度
    return coords


def regularize_rboxes(rboxes):
    """将旋转边界框正则化到 [0, pi/2) 范围。

    Args:
        rboxes (torch.Tensor): 输入旋转框，形状 (N, 5)，xywhr 格式。

    Returns:
        (torch.Tensor): 正则化后的旋转框。
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # 如果 t >= pi/2 且非对称则交换边
    swap = t % math.pi >= math.pi / 2
    w_ = torch.where(swap, h, w)
    h_ = torch.where(swap, w, h)
    t = t % (math.pi / 2)
    return torch.stack([x, y, w_, h_, t], dim=-1)  # 正则化后的框


def masks2segments(masks: np.ndarray | torch.Tensor, strategy: str = "all") -> list[np.ndarray]:
    """使用轮廓检测将掩码转换为分割。

    Args:
        masks (np.ndarray | torch.Tensor): 形状 (N, H, W) 的二值掩码。
        strategy (str): 分割策略，'all' 或 'largest'。

    Returns:
        (list): float32 数组形式的分割掩码列表。
    """
    from ultralytics.data.converter import merge_multi_segment

    masks = masks.astype("uint8") if isinstance(masks, np.ndarray) else masks.byte().cpu().numpy()
    segments = []
    for x in np.ascontiguousarray(masks):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # 合并并连接所有分割
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # 选择最大分割
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # 未找到分割
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """将一批 FP32 torch 张量转换为 NumPy uint8 数组，从 BCHW 布局改为 BHWC 布局。

    Args:
        batch (torch.Tensor): 输入张量批次，形状 (Batch, Channels, Height, Width)，dtype torch.float32。

    Returns:
        (np.ndarray): 输出 NumPy 数组批次，形状 (Batch, Height, Width, Channels)，dtype uint8。
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).byte().cpu().numpy()


def clean_str(s):
    """通过将特殊字符替换为 '_' 来清理字符串。

    Args:
        s (str): 需要替换特殊字符的字符串。

    Returns:
        (str): 特殊字符替换为下划线 _ 的字符串。
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨`><+]", repl="_", string=s)


def empty_like(x):
    """创建与输入具有相同形状和 dtype 的空 torch.Tensor 或 np.ndarray。"""
    return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)
