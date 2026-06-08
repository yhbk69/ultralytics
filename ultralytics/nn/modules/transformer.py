# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer 模块。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.torch_utils import TORCH_1_11

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "AIFI",
    "MLP",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "LayerNorm2d",
    "MLPBlock",
    "MSDeformAttn",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
)


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器的单层。

    该类实现了一个标准的 Transformer 编码器层，包含多头注意力机制和前馈网络，支持前置归一化和后置归一化两种配置。

    Attributes:
        ma (nn.MultiheadAttention): 多头注意力模块。
        fc1 (nn.Linear): 前馈网络中的第一个线性层。
        fc2 (nn.Linear): 前馈网络中的第二个线性层。
        norm1 (nn.LayerNorm): 注意力层后的层归一化。
        norm2 (nn.LayerNorm): 前馈网络后的层归一化。
        dropout (nn.Dropout): 前馈网络的 Dropout 层。
        dropout1 (nn.Dropout): 注意力层后的 Dropout 层。
        dropout2 (nn.Dropout): 前馈网络后的 Dropout 层。
        act (nn.Module): 激活函数。
        normalize_before (bool): 是否在注意力和前馈网络之前应用归一化。
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """初始化 TransformerEncoderLayer，使用指定的参数。

        Args:
            c1 (int): 输入维度。
            cm (int): 前馈网络中的隐藏维度。
            num_heads (int): 注意力头数。
            dropout (float): Dropout 概率。
            act (nn.Module): 激活函数。
            normalize_before (bool): 是否在注意力和前馈网络之前应用归一化。
        """
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # 前馈网络模型的实现
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """如果提供了位置嵌入，则将其添加到输入张量上。"""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行后置归一化的前向传播。

        Args:
            src (torch.Tensor): 输入张量。
            src_mask (torch.Tensor, optional): 源序列的掩码。
            src_key_padding_mask (torch.Tensor, optional): 按批次对源键进行填充的掩码。
            pos (torch.Tensor, optional): 位置编码。

        Returns:
            (torch.Tensor): 经过注意力和前馈网络后的输出张量。
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行前置归一化的前向传播。

        Args:
            src (torch.Tensor): 输入张量。
            src_mask (torch.Tensor, optional): 源序列的掩码。
            src_key_padding_mask (torch.Tensor, optional): 按批次对源键进行填充的掩码。
            pos (torch.Tensor, optional): 位置编码。

        Returns:
            (torch.Tensor): 经过注意力和前馈网络后的输出张量。
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将输入通过编码器模块进行前向传播。

        Args:
            src (torch.Tensor): 输入张量。
            src_mask (torch.Tensor, optional): 源序列的掩码。
            src_key_padding_mask (torch.Tensor, optional): 按批次对源键进行填充的掩码。
            pos (torch.Tensor, optional): 位置编码。

        Returns:
            (torch.Tensor): Transformer 编码器层处理后的输出张量。
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """AIFI Transformer 层，用于带有位置嵌入的 2D 数据。

    该类继承自 TransformerEncoderLayer，通过添加 2D 正弦-余弦位置嵌入并适当处理空间维度，
    使其能够处理 2D 特征图。
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """使用指定的参数初始化 AIFI 实例。

        Args:
            c1 (int): 输入维度。
            cm (int): 前馈网络中的隐藏维度。
            num_heads (int): 注意力头数。
            dropout (float): Dropout 概率。
            act (nn.Module): 激活函数。
            normalize_before (bool): 是否在注意力和前馈网络之前应用归一化。
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """AIFI Transformer 层的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]。

        Returns:
            (torch.Tensor): 输出张量，形状为 [B, C, H, W]。
        """
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # 将 [B, C, H, W] 展平为 [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """构建 2D 正弦-余弦位置嵌入。

        Args:
            w (int): 特征图的宽度。
            h (int): 特征图的高度。
            embed_dim (int): 嵌入维度。
            temperature (float): 正弦/余弦函数的温度参数。

        Returns:
            (torch.Tensor): 位置嵌入，形状为 [1, h*w, embed_dim]。
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij") if TORCH_1_11 else torch.meshgrid(grid_w, grid_h)
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer 层 https://arxiv.org/abs/2010.11929（为了更好的性能移除了 LayerNorm 层）。"""

    def __init__(self, c: int, num_heads: int):
        """使用线性变换和多头注意力初始化自注意力机制。

        Args:
            c (int): 输入和输出通道维度。
            num_heads (int): 注意力头数。
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入 x 应用 Transformer 块并返回输出。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            (torch.Tensor): Transformer 层处理后的输出张量。
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """基于 https://arxiv.org/abs/2010.11929 的 Vision Transformer 块。

    该类实现了一个完整的 Transformer 块，包含可选的卷积层用于通道调整、可学习的位置嵌入
    以及多个 Transformer 层。

    Attributes:
        conv (Conv, optional): 输入和输出通道不同时使用的卷积层。
        linear (nn.Linear): 可学习的位置嵌入。
        tr (nn.Sequential): Transformer 层的顺序容器。
        c2 (int): 输出通道维度。
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """初始化 Transformer 模块，包含位置嵌入和指定数量的注意力头与层数。

        Args:
            c1 (int): 输入通道维度。
            c2 (int): 输出通道维度。
            num_heads (int): 注意力头数。
            num_layers (int): Transformer 层数。
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # 可学习的位置嵌入
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将输入通过 Transformer 块进行前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 [b, c1, h, w]。

        Returns:
            (torch.Tensor): 输出张量，形状为 [b, c2, h, w]。
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, h, w = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, h, w)


class MLPBlock(nn.Module):
    """多层感知机的单个模块。"""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """使用指定的嵌入维度、MLP 维度和激活函数初始化 MLPBlock。

        Args:
            embedding_dim (int): 输入和输出维度。
            mlp_dim (int): 隐藏维度。
            act (type): 激活函数类。
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLPBlock 的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            (torch.Tensor): MLP 块处理后的输出张量。
        """
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """一个简单的多层感知机（也称为 FFN）。

    该类实现了一个可配置的 MLP，包含多个线性层、激活函数以及可选的 sigmoid 输出激活。

    Attributes:
        num_layers (int): MLP 中的层数。
        layers (nn.ModuleList): 线性层列表。
        sigmoid (bool): 是否对输出应用 sigmoid。
        act (nn.Module): 激活函数。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act=nn.ReLU,
        sigmoid: bool = False,
        residual: bool = False,
        out_norm: nn.Module = None,
    ):
        """使用指定的输入、隐藏、输出维度和层数初始化 MLP。

        Args:
            input_dim (int): 输入维度。
            hidden_dim (int): 隐藏维度。
            output_dim (int): 输出维度。
            num_layers (int): 层数。
            act (type): 激活函数类。
            sigmoid (bool): 是否对输出应用 sigmoid。
            residual (bool): 是否使用残差连接。
            out_norm (nn.Module, optional): 输出的归一化层。
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.sigmoid = sigmoid
        self.act = act()
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # 是否对输出应用归一化层
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """整个 MLP 的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            (torch.Tensor): MLP 处理后的输出张量。
        """
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        if getattr(self, "residual", False):
            x = x + orig_x
        x = getattr(self, "out_norm", nn.Identity())(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """受 Detectron2 和 ConvNeXt 实现启发的 2D 层归一化模块。

    该类实现了用于 2D 特征图的层归一化，在保留空间维度的同时跨通道维度进行归一化。

    Attributes:
        weight (nn.Parameter): 可学习的缩放参数。
        bias (nn.Parameter): 可学习的偏置参数。
        eps (float): 用于数值稳定性的小常数。

    References:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """使用给定的参数初始化 LayerNorm2d。

        Args:
            num_channels (int): 输入中的通道数。
            eps (float): 用于数值稳定性的小常数。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 2D 层归一化的前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            (torch.Tensor): 归一化后的输出张量。
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """基于 Deformable-DETR 和 PaddleDetection 实现的多尺度可变形注意力模块。

    该模块实现了多尺度可变形注意力，可以在多个尺度上关注特征，具有可学习的采样位置和注意力权重。

    Attributes:
        im2col_step (int): im2col 操作的步长。
        d_model (int): 模型维度。
        n_levels (int): 特征层级数。
        n_heads (int): 注意力头数。
        n_points (int): 每个注意力头在每个特征层级上的采样点数。
        sampling_offsets (nn.Linear): 用于生成采样偏移量的线性层。
        attention_weights (nn.Linear): 用于生成注意力权重的线性层。
        value_proj (nn.Linear): 用于投影值的线性层。
        output_proj (nn.Linear): 用于投影输出的线性层。

    References:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """使用给定的参数初始化 MSDeformAttn。

        Args:
            d_model (int): 模型维度。
            n_levels (int): 特征层级数。
            n_heads (int): 注意力头数。
            n_points (int): 每个注意力头在每个特征层级上的采样点数。
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # 最好将 _d_per_head 设置为 2 的幂，这在 CUDA 实现中更高效
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """重置模块参数。"""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行多尺度可变形注意力的前向传播。

        Args:
            query (torch.Tensor): 查询张量，形状为 [bs, query_length, C]。
            refer_bbox (torch.Tensor): 参考边界框，形状为 [bs, query_length, n_levels, 2 或 4]，
                范围在 [0, 1] 内，左上角为 (0,0)，右下角为 (1, 1)，包含填充区域。
            value (torch.Tensor): 值张量，形状为 [bs, value_length, C]。
            value_shapes (list): 形状为 [n_levels, 2] 的列表，[(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]。
            value_mask (torch.Tensor, optional): 掩码张量，形状为 [bs, value_length]，True 表示填充元素，
                False 表示非填充元素。

        Returns:
            (torch.Tensor): 输出张量，形状为 [bs, Length_{query}, C]。

        References:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """受 PaddleDetection 和 Deformable-DETR 实现启发的可变形 Transformer 解码器层。

    该类实现了一个单独的解码器层，包含自注意力、使用多尺度可变形注意力的交叉注意力以及前馈网络。

    Attributes:
        self_attn (nn.MultiheadAttention): 自注意力模块。
        dropout1 (nn.Dropout): 自注意力后的 Dropout。
        norm1 (nn.LayerNorm): 自注意力后的层归一化。
        cross_attn (MSDeformAttn): 交叉注意力模块。
        dropout2 (nn.Dropout): 交叉注意力后的 Dropout。
        norm2 (nn.LayerNorm): 交叉注意力后的层归一化。
        linear1 (nn.Linear): 前馈网络中的第一个线性层。
        act (nn.Module): 激活函数。
        dropout3 (nn.Dropout): 前馈网络中的 Dropout。
        linear2 (nn.Linear): 前馈网络中的第二个线性层。
        dropout4 (nn.Dropout): 前馈网络后的 Dropout。
        norm3 (nn.LayerNorm): 前馈网络后的层归一化。

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """使用给定的参数初始化 DeformableTransformerDecoderLayer。

        Args:
            d_model (int): 模型维度。
            n_heads (int): 注意力头数。
            d_ffn (int): 前馈网络的维度。
            dropout (float): Dropout 概率。
            act (nn.Module): 激活函数。
            n_levels (int): 特征层级数。
            n_points (int): 采样点数。
        """
        super().__init__()

        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 交叉注意力
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """将位置嵌入添加到输入张量上（如果提供了的话）。"""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """执行层中前馈网络部分的前向传播。

        Args:
            tgt (torch.Tensor): 输入张量。

        Returns:
            (torch.Tensor): FFN 处理后的输出张量。
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行整个解码器层的前向传播。

        Args:
            embed (torch.Tensor): 输入嵌入。
            refer_bbox (torch.Tensor): 参考边界框。
            feats (torch.Tensor): 特征图。
            shapes (list): 特征形状。
            padding_mask (torch.Tensor, optional): 填充掩码。
            attn_mask (torch.Tensor, optional): 注意力掩码。
            query_pos (torch.Tensor, optional): 查询位置嵌入。

        Returns:
            (torch.Tensor): 解码器层处理后的输出张量。
        """
        # 自注意力
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # 交叉注意力
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """基于 PaddleDetection 实现的可变形 Transformer 解码器。

    该类实现了一个完整的可变形 Transformer 解码器，包含多个解码器层以及用于边界框回归和分类的预测头。

    Attributes:
        layers (nn.ModuleList): 解码器层列表。
        num_layers (int): 解码器层数。
        hidden_dim (int): 隐藏维度。
        eval_idx (int): 评估期间使用的层的索引。

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """使用给定的参数初始化 DeformableTransformerDecoder。

        Args:
            hidden_dim (int): 隐藏维度。
            decoder_layer (nn.Module): 解码器层模块。
            num_layers (int): 解码器层数。
            eval_idx (int): 评估期间使用的层的索引。
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # 解码器嵌入
        refer_bbox: torch.Tensor,  # 锚点
        feats: torch.Tensor,  # 图像特征
        shapes: list,  # 特征形状
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """执行整个解码器的前向传播。

        Args:
            embed (torch.Tensor): 解码器嵌入。
            refer_bbox (torch.Tensor): 参考边界框。
            feats (torch.Tensor): 图像特征。
            shapes (list): 特征形状。
            bbox_head (nn.Module): 边界框预测头。
            score_head (nn.Module): 分数预测头。
            pos_mlp (nn.Module): 位置 MLP。
            attn_mask (torch.Tensor, optional): 注意力掩码。
            padding_mask (torch.Tensor, optional): 填充掩码。

        Returns:
            dec_bboxes (torch.Tensor): 解码后的边界框。
            dec_cls (torch.Tensor): 解码后的分类分数。
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
