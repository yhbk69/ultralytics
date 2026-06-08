# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""用于估算最佳 YOLO 批量大小的函数，以使用可用 CUDA 内存的一部分。"""

from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import torch

from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import autocast, profile_ops


def check_train_batch_size(
    model: torch.nn.Module,
    imgsz: int = 640,
    amp: bool = True,
    batch: int | float = -1,
    max_num_obj: int = 1,
    dataset_size: int = 0,
) -> int:
    """使用 autobatch() 函数计算最佳的 YOLO 训练批量大小。

    参数:
        model (torch.nn.Module): 要检查批量大小的 YOLO 模型。
        imgsz (int, 可选): 训练使用的图像大小。
        amp (bool, 可选): 如果为 True，使用自动混合精度。
        batch (int | float, 可选): 要使用的 GPU 内存比例。如果为 -1，使用默认值。
        max_num_obj (int, 可选): 数据集中的最大目标数。
        dataset_size (int, 可选): 训练图像总数。如果 > 0，批量大小不会超过此值。

    返回:
        (int): 使用 autobatch() 函数计算的最佳批量大小。

    注意:
        如果 0.0 < batch < 1.0，它将作为使用的 GPU 内存比例。
        否则，使用默认比例 0.6。
    """
    with autocast(enabled=amp):
        return autobatch(
            deepcopy(model).train(),
            imgsz,
            fraction=batch if 0.0 < batch < 1.0 else 0.6,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )


def autobatch(
    model: torch.nn.Module,
    imgsz: int = 640,
    fraction: float = 0.60,
    batch_size: int = DEFAULT_CFG.batch,
    max_num_obj: int = 1,
    dataset_size: int = 0,
) -> int:
    """自动估算最佳 YOLO 批量大小，以使用可用 CUDA 内存的一部分。

    参数:
        model (torch.nn.Module): 要计算批量大小的 YOLO 模型。
        imgsz (int, 可选): YOLO 模型的输入图像大小。
        fraction (float, 可选): 要使用的可用 CUDA 内存比例。
        batch_size (int, 可选): 检测到错误时使用的默认批量大小。
        max_num_obj (int, 可选): 数据集中的最大目标数。
        dataset_size (int, 可选): 训练图像总数。如果 > 0，批量大小不会超过此值。

    返回:
        (int): 最佳批量大小。
    """
    # 检查设备
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% CUDA memory utilization.")
    device = next(model.parameters()).device  # 获取模型设备
    if device.type in {"cpu", "mps"}:
        LOGGER.warning(f"{prefix}intended for CUDA devices, using default batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.warning(f"{prefix}Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    # 检查 CUDA 内存
    gb = 1 << 30  # 字节转换为 GiB (1024 ** 3)
    d = f"CUDA:{os.getenv('CUDA_VISIBLE_DEVICES', '0').strip()[0]}"  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # 设备属性
    t = properties.total_memory / gb  # GiB 总量
    r = torch.cuda.memory_reserved(device) / gb  # GiB 已保留
    a = torch.cuda.memory_allocated(device) / gb  # GiB 已分配
    f = t - (r + a)  # GiB 空闲
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    # 分析不同批量大小
    batch_sizes = [1, 2, 4, 8, 16] if t < 16 else [1, 2, 4, 8, 16, 32, 64]
    if dataset_size > 0:
        batch_sizes = [b for b in batch_sizes if b <= dataset_size]
    ch = model.yaml.get("channels", 3)
    try:
        img = [torch.empty(b, ch, imgsz, imgsz) for b in batch_sizes]
        results = profile_ops(img, model, n=1, device=device, max_num_obj=max_num_obj)

        # 拟合解
        xy = [
            [x, y[2]]
            for i, (x, y) in enumerate(zip(batch_sizes, results))
            if y  # 有效结果
            and isinstance(y[2], (int, float))  # 是数值
            and 0 < y[2] < t  # 在 0 和 GPU 限制之间
            and (i == 0 or not results[i - 1] or y[2] > results[i - 1][2])  # 第一项或内存递增
        ]
        fit_x, fit_y = zip(*xy) if xy else ([], [])
        p = np.polyfit(fit_x, fit_y, deg=1)  # 一阶多项式拟合
        b = int((round(f * fraction) - p[1]) / p[0])  # y 截距（最佳批量大小）
        if None in results:  # 某些大小失败
            i = results.index(None)  # 第一个失败的索引
            if b >= batch_sizes[i]:  # y 截距超过失败点
                b = batch_sizes[max(i - 1, 0)]  # 选择之前的安全点
        if b < 1 or b > 1024:  # b 超出安全范围
            LOGGER.warning(f"{prefix}batch={b} outside safe range, using default batch-size {batch_size}.")
            b = batch_size
        if dataset_size > 0:
            b = min(b, dataset_size)

        fraction = (np.polyval(p, b) + r + a) / t  # 预测比例
        LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        return b
    except Exception as e:
        LOGGER.warning(f"{prefix}error detected: {e},  using default batch-size {batch_size}.")
        return batch_size
    finally:
        torch.cuda.empty_cache()
