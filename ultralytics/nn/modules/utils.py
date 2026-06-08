# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

__all__ = "inverse_sigmoid", "multi_scale_deformable_attn_pytorch"


def _get_clones(module, n):
    """从给定模块创建 n 个深拷贝，返回 ModuleList。

    Args:
        module (nn.Module): 待克隆的模块。
        n (int): 克隆数量。

    Returns:
        (nn.ModuleList): 包含 n 个输入模块深拷贝的 ModuleList。

    Examples:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """根据给定的先验概率值初始化卷积/全连接层的偏置。

    通过逆 Sigmoid（logit）函数将先验概率转换为偏置初始值，
    常用于目标检测模型的分类层初始化，使初始正样本预测概率等于 prior_prob。

    Args:
        prior_prob (float, optional): 用于偏置初始化的先验概率。

    Returns:
        (float): 由先验概率计算得出的偏置初始值。

    Examples:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Bias initialization value: {bias:.4f}")
        Bias initialization value: -4.5951
    """
    return float(-np.log((1 - prior_prob) / prior_prob))  # 返回偏置初始值


def linear_init(module):
    """初始化线性模块的权重和偏置。

    使用由输出维度计算的范围内的均匀分布初始化权重，
    若模块有偏置项也一并初始化。

    Args:
        module (nn.Module): 待初始化的线性模块。

    Examples:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """计算张量的逆 Sigmoid 函数。

    对张量应用 Sigmoid 的逆运算，在注意力机制和坐标变换等神经网络操作中十分有用。

    Args:
        x (torch.Tensor): 值域为 [0, 1] 的输入张量。
        eps (float, optional): 防止数值不稳定的小 epsilon 值。

    Returns:
        (torch.Tensor): 应用逆 Sigmoid 后的输出张量。

    Examples:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """用 PyTorch 实现多尺度可变形注意力。

    在多个特征图尺度上执行可变形注意力，允许模型通过学习到的偏移量关注不同空间位置。

    Args:
        value (torch.Tensor): 值张量，形状为 (bs, num_keys, num_heads, embed_dims)。
        value_spatial_shapes (torch.Tensor): 各尺度特征图的空间尺寸，形状为 (num_levels, 2)。
        sampling_locations (torch.Tensor): 采样位置，形状为
            (bs, num_queries, num_heads, num_levels, num_points, 2)。
        attention_weights (torch.Tensor): 注意力权重，形状为
            (bs, num_queries, num_heads, num_levels, num_points)。

    Returns:
        (torch.Tensor): 输出张量，形状为 (bs, num_queries, num_heads * embed_dims)。

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()
