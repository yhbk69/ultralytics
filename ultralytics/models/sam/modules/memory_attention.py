# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import copy

import torch
from torch import nn

from .blocks import RoPEAttention


class MemoryAttentionLayer(nn.Module):
    """实现具有自注意力和交叉注意力机制的内存注意力层，用于神经网络。

    此类结合了自注意力、交叉注意力和前馈组件来处理输入张量并生成基于内存的注意力输出。

    Attributes:
        d_model (int): 模型的维度。
        dim_feedforward (int): 前馈网络的维度。
        dropout_value (float): 正则化的dropout率。
        self_attn (RoPEAttention): 使用RoPE（旋转位置嵌入）的自注意力机制。
        cross_attn_image (RoPEAttention): 用于图像处理的交叉注意力机制。
        linear1 (nn.Linear): 前馈网络的第一全连接层。
        linear2 (nn.Linear): 前馈网络的第二全连接层。
        norm1 (nn.LayerNorm): 自注意力输出的层归一化。
        norm2 (nn.LayerNorm): 交叉注意力输出的层归一化。
        norm3 (nn.LayerNorm): 前馈网络输出的层归一化。
        dropout1 (nn.Dropout): 自注意力后的dropout层。
        dropout2 (nn.Dropout): 交叉注意力后的dropout层。
        dropout3 (nn.Dropout): 前馈网络后的dropout层。
        activation (nn.ReLU): 前馈网络的激活函数。
        pos_enc_at_attn (bool): 是否在注意力处添加位置编码的标志。
        pos_enc_at_cross_attn_queries (bool): 是否在交叉注意力查询处添加位置编码的标志。
        pos_enc_at_cross_attn_keys (bool): 是否在交叉注意力键处添加位置编码的标志。

    Methods:
        forward: 对输入张量执行完整的内存注意力操作。
        _forward_sa: 对输入张量执行自注意力。
        _forward_ca: 在目标张量和内存张量之间执行交叉注意力。

    Examples:
        >>> layer = MemoryAttentionLayer(d_model=256, dim_feedforward=2048, dropout=0.1)
        >>> tgt = torch.randn(1, 100, 256)
        >>> memory = torch.randn(1, 100, 64)
        >>> pos = torch.randn(1, 100, 256)
        >>> query_pos = torch.randn(1, 100, 256)
        >>> output = layer(tgt, memory, pos, query_pos)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_enc_at_attn: bool = False,
        pos_enc_at_cross_attn_keys: bool = True,
        pos_enc_at_cross_attn_queries: bool = False,
        self_attn: nn.Module | None = None,
        cross_attn: nn.Module | None = None,
    ):
        """初始化具有自注意力、交叉注意力和前馈组件的内存注意力层。

        Args:
            d_model (int): 模型的维度。
            dim_feedforward (int): 前馈网络的维度。
            dropout (float): 正则化的dropout率。
            pos_enc_at_attn (bool): 是否在注意力处添加位置编码。
            pos_enc_at_cross_attn_keys (bool): 是否在交叉注意力键处添加位置编码。
            pos_enc_at_cross_attn_queries (bool): 是否在交叉注意力查询处添加位置编码。
            self_attn (nn.Module | None): 自定义自注意力模块。如果为None，则使用默认的RoPEAttention。
            cross_attn (nn.Module | None): 自定义交叉注意力模块。如果为None，则使用默认的RoPEAttention。
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attn or RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1)
        self.cross_attn_image = cross_attn or RoPEAttention(
            rope_k_repeat=True,
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=64,
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor | None) -> torch.Tensor:
        """使用位置编码和RoPE注意力机制对输入张量执行自注意力。"""
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None,
        pos: torch.Tensor | None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """使用RoPEAttention机制在目标张量和内存张量之间执行交叉注意力。"""
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """通过自注意力、交叉注意力和前馈网络层处理输入张量。

        Args:
            tgt (torch.Tensor): 自注意力的目标张量，形状为(N, L, D)。
            memory (torch.Tensor): 交叉注意力的内存张量，形状为(N, S, D)。
            pos (torch.Tensor | None): 内存张量的位置编码。
            query_pos (torch.Tensor | None): 目标张量的位置编码。
            num_k_exclude_rope (int): 从旋转位置嵌入中排除的键数量。

        Returns:
            (torch.Tensor): 经过注意力和前馈层处理后形状为(N, L, D)的张量。
        """
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    """用于处理具有自注意力和交叉注意力机制的序列数据的内存注意力模块。

    此类实现了结合自注意力和交叉注意力的多层注意力机制，用于处理序列数据，
    在transformer类架构中特别有用。

    Attributes:
        d_model (int): 模型隐藏状态的维度。
        layers (nn.ModuleList): MemoryAttentionLayer模块的列表。
        num_layers (int): 注意力层的数量。
        norm (nn.LayerNorm): 应用于输出的层归一化。
        pos_enc_at_input (bool): 是否在输入处应用位置编码。
        batch_first (bool): 输入张量是否为batch first格式。

    Methods:
        forward: 通过注意力层处理输入张量。

    Examples:
        >>> d_model = 256
        >>> layer = MemoryAttentionLayer(d_model)
        >>> attention = MemoryAttention(d_model, pos_enc_at_input=True, layer=layer, num_layers=3)
        >>> curr = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
        >>> memory = torch.randn(20, 32, d_model)  # (mem_len, batch_size, d_model)
        >>> curr_pos = torch.randn(10, 32, d_model)
        >>> memory_pos = torch.randn(20, 32, d_model)
        >>> output = attention(curr, memory, curr_pos, memory_pos)
        >>> print(output.shape)
        torch.Size([10, 32, 256])
    """

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        """初始化MemoryAttention，使用指定的层和归一化处理序列数据。

        此类实现了结合自注意力和交叉注意力的多层注意力机制，用于处理序列数据，
        在transformer类架构中特别有用。

        Args:
            d_model (int): 模型隐藏状态的维度。
            pos_enc_at_input (bool): 是否在输入处应用位置编码。
            layer (nn.Module): 模块中使用的注意力层。
            num_layers (int): 注意力层的数量。
            batch_first (bool): 输入张量是否为batch first格式。
        """
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: torch.Tensor | None = None,  # pos_enc for self-attention inputs
        memory_pos: torch.Tensor | None = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ) -> torch.Tensor:
        """通过注意力层处理输入，应用自注意力和交叉注意力以及位置编码。

        Args:
            curr (torch.Tensor): 自注意力输入张量，表示当前状态。
            memory (torch.Tensor): 交叉注意力输入张量，表示内存信息。
            curr_pos (torch.Tensor | None): 自注意力输入的位置编码。
            memory_pos (torch.Tensor | None): 交叉注意力输入的位置编码。
            num_obj_ptr_tokens (int): 从旋转位置嵌入中排除的对象指针令牌数量。

        Returns:
            (torch.Tensor): 经过注意力层和归一化处理后的输出张量。

        Examples:
            >>> d_model = 256
            >>> layer = MemoryAttentionLayer(d_model)
            >>> attention = MemoryAttention(d_model, pos_enc_at_input=True, layer=layer, num_layers=3)
            >>> curr = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
            >>> memory = torch.randn(20, 32, d_model)  # (mem_len, batch_size, d_model)
            >>> curr_pos = torch.randn(10, 32, d_model)
            >>> memory_pos = torch.randn(20, 32, d_model)
            >>> output = attention(curr, memory, curr_pos, memory_pos)
            >>> print(output.shape)
            torch.Size([10, 32, 256])
        """
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        assert curr.shape[1] == memory.shape[1], "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
