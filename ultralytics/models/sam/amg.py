# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from collections.abc import Generator
from itertools import product
from typing import Any

import numpy as np
import torch


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: list[int], orig_box: list[int], atol: float = 20.0
) -> torch.Tensor:
    """判断边界框是否接近裁剪图像区域的边缘，使用指定的容差。

    Args:
        boxes (torch.Tensor): XYXY格式的边界框。
        crop_box (list[int]): 裁剪框坐标，格式为[x0, y0, x1, y1]。
        orig_box (list[int]): 原始图像框坐标，格式为[x0, y0, x1, y1]。
        atol (float, optional): 边缘接近检测的绝对容差。

    Returns:
        (torch.Tensor): 布尔张量，指示哪些框接近裁剪边缘。

    Examples:
        >>> boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]])
        >>> crop_box = [0, 0, 200, 200]
        >>> orig_box = [0, 0, 300, 300]
        >>> near_edge = is_box_near_crop_edge(boxes, crop_box, orig_box, atol=20.0)
    """
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def batch_iterator(batch_size: int, *args) -> Generator[list[Any]]:
    """将输入参数的数据按指定批次大小分批生成，以便高效处理。

    此函数接受一个批次大小和任意数量的可迭代对象，然后从这些可迭代对象中生成元素的批次。
    所有输入可迭代对象必须具有相同长度。

    Args:
        batch_size (int): 每个批次的大小。
        *args (Any): 可变长度的输入可迭代对象。所有可迭代对象必须具有相同长度。

    Yields:
        (list[Any]): 每个输入可迭代对象中批量元素的列表。

    Examples:
        >>> data = [1, 2, 3, 4, 5]
        >>> labels = ["a", "b", "c", "d", "e"]
        >>> for batch in batch_iterator(2, data, labels):
        ...     print(batch)
        [[1, 2], ['a', 'b']]
        [[3, 4], ['c', 'd']]
        [[5], ['e']]
    """
    assert args and all(len(a) == len(args[0]) for a in args), "Batched iteration must have same-size inputs."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def calculate_stability_score(masks: torch.Tensor, mask_threshold: float, threshold_offset: float) -> torch.Tensor:
    """计算一批掩码的稳定性分数。

    稳定性分数是通过在高值和低值处对预测掩码logits进行阈值处理获得的二进制掩码之间的IoU。

    Args:
        masks (torch.Tensor): 一批预测掩码logits。
        mask_threshold (float): 创建二进制掩码的阈值。
        threshold_offset (float): 应用于阈值的偏移量，用于创建高值和低值的二进制掩码。

    Returns:
        (torch.Tensor): 批次中每个掩码的稳定性分数。

    Examples:
        >>> masks = torch.rand(10, 256, 256)  # 包含10个掩码的批次
        >>> mask_threshold = 0.5
        >>> threshold_offset = 0.1
        >>> stability_scores = calculate_stability_score(masks, mask_threshold, threshold_offset)

    Notes:
        - 一个掩码始终包含在另一个掩码内部。
        - 通过避免不必要的torch.int64类型转换来节省内存。
    """
    intersections = (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """在[0,1]x[0,1]范围内生成均匀分布的2D点网格，用于图像分割任务。"""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)


def build_all_layer_point_grids(n_per_side: int, n_layers: int, scale_per_layer: int) -> list[np.ndarray]:
    """为多个裁剪层生成不同缩放比例和密度的点网格。"""
    return [build_point_grid(int(n_per_side / (scale_per_layer**i))) for i in range(n_layers + 1)]


def generate_crop_boxes(
    im_size: tuple[int, ...], n_layers: int, overlap_ratio: float
) -> tuple[list[list[int]], list[int]]:
    """生成不同大小的裁剪框，用于多尺度图像处理，具有分层的重叠区域。

    Args:
        im_size (tuple[int, ...]): 输入图像的高度和宽度。
        n_layers (int): 生成裁剪框的层数。
        overlap_ratio (float): 相邻裁剪框之间的重叠比例。

    Returns:
        crop_boxes (list[list[int]]): 裁剪框列表，格式为[x0, y0, x1, y1]。
        layer_idxs (list[int]): 每个裁剪框对应的层索引列表。

    Examples:
        >>> im_size = (800, 1200)  # 高度, 宽度
        >>> n_layers = 3
        >>> overlap_ratio = 0.25
        >>> crop_boxes, layer_idxs = generate_crop_boxes(im_size, n_layers, overlap_ratio)
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # 原始图像
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        """根据原始长度、裁剪数量和重叠计算每个裁剪的长度。"""
        return math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops)

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # 以XYWH格式的裁剪
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: list[int]) -> torch.Tensor:
    """通过将裁剪框偏移量加到边界框坐标上来恢复原始坐标。"""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # 检查boxes是否有通道维度
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def uncrop_points(points: torch.Tensor, crop_box: list[int]) -> torch.Tensor:
    """通过将裁剪框偏移量加到点坐标上来恢复原始坐标。"""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    # 检查points是否有通道维度
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def uncrop_masks(masks: torch.Tensor, crop_box: list[int], orig_h: int, orig_w: int) -> torch.Tensor:
    """通过将掩码填充到原始图像尺寸来恢复掩码，处理坐标变换。"""
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # 坐标变换掩码
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)


def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> tuple[np.ndarray, bool]:
    """根据面积阈值和模式移除掩码中较小的不连通区域或孔洞。

    Args:
        mask (np.ndarray): 要处理的二进制掩码。
        area_thresh (float): 面积阈值，小于此值的区域将被移除。
        mode (str): 处理模式，'holes'表示填充小孔洞，'islands'表示移除小孤立区域。

    Returns:
        processed_mask (np.ndarray): 移除小区域后的二进制掩码。
        modified (bool): 是否有任何区域被修改。

    Examples:
        >>> mask = np.zeros((100, 100), dtype=np.bool_)
        >>> mask[40:60, 40:60] = True  # 创建一个正方形
        >>> mask[45:55, 45:55] = False  # 创建一个孔洞
        >>> processed_mask, modified = remove_small_regions(mask, 50, "holes")
    """
    import cv2  # type: ignore

    assert mode in {"holes", "islands"}, f"Provided mode {mode} is invalid"
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # 第0行是背景标签
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if not small_regions:
        return mask, False
    fill_labels = [0, *small_regions]
    if not correct_holes:
        # 如果每个区域都低于阈值，保留最大的
        fill_labels = [i for i in range(n_labels) if i not in fill_labels] or [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """计算二进制掩码周围的XYXY格式边界框。

    Args:
        masks (torch.Tensor): 二进制掩码，形状为(B, H, W)或(B, C, H, W)。

    Returns:
        (torch.Tensor): XYXY格式的边界框，形状为(B, 4)或(B, C, 4)。

    Notes:
        - 对空掩码返回零框。
        - 在输出中保留输入张量的维度。
    """
    # 下面的torch.max在空输入上会报错，直接跳过这种情况
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # 将形状规范化为CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    masks = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)
    # 获取顶部和底部边缘
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # 获取左侧和右侧边缘
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # 如果掩码为空，右边缘会在左边缘的左侧。
    # 将这些框替换为[0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # 恢复为原始形状
    return out.reshape(*shape[:-2], 4) if len(shape) > 2 else out[0]
