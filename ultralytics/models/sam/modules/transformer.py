# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import MLPBlock


class TwoWayTransformer(nn.Module):
    """一个同时对图像和查询点进行注意力的双向Transformer模块。

    此类实现了一个专用的transformer解码器，使用带有位置嵌入的查询来关注输入图像。它适用于目标检测、
    图像分割和点云处理等任务。

    Attributes:
        depth (int): transformer中的层数。
        embedding_dim (int): 输入嵌入的通道维度。
        num_heads (int): 多头注意力的头数。
        mlp_dim (int): MLP块的内部通道维度。
        layers (nn.ModuleList): 组成transformer的TwoWayAttentionBlock层列表。
        final_attn_token_to_image (Attention): 从查询到图像的最终注意力层。
        norm_final_attn (nn.LayerNorm): 应用于最终查询的层归一化。

    Methods:
        forward: 通过transformer处理图像和点嵌入。

    Examples:
        >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 32, 32)
        >>> image_pe = torch.randn(1, 256, 32, 32)
        >>> point_embedding = torch.randn(1, 100, 256)
        >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
        >>> print(output_queries.shape, output_image.shape)
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """初始化一个同时对图像和查询点进行注意力的双向Transformer。

        Args:
            depth (int): transformer中的层数。
            embedding_dim (int): 输入嵌入的通道维度。
            num_heads (int): 多头注意力的头数。必须能整除embedding_dim。
            mlp_dim (int): MLP块的内部通道维度。
            activation (type[nn.Module], optional): MLP块中使用的激活函数。
            attention_downsample_rate (int, optional): 注意力机制的下采样率。
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """通过双向Transformer处理图像和点嵌入。

        Args:
            image_embedding (torch.Tensor): 要关注的图像，形状为(B, embedding_dim, H, W)。
            image_pe (torch.Tensor): 添加到图像的位置编码，形状与image_embedding相同。
            point_embedding (torch.Tensor): 添加到查询点的嵌入，形状为(B, N_points, embedding_dim)。

        Returns:
            queries (torch.Tensor): 处理后的点嵌入，形状为(B, N_points, embedding_dim)。
            keys (torch.Tensor): 处理后的图像嵌入，形状为(B, H*W, embedding_dim)。
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # 准备查询
        queries = point_embedding
        keys = image_embedding

        # 应用transformer块和最终层归一化
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # 应用从点到图像的最终注意力层
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """一个同时对图像和查询点进行注意力的双向注意力块。

    此类实现了一个专用的transformer块，包含四个主要层：稀疏输入上的自注意力、
    从稀疏输入到密集输入的交叉注意力、稀疏输入上的MLP块，以及从密集输入到
    稀疏输入的交叉注意力。

    Attributes:
        self_attn (Attention): 查询的自注意力层。
        norm1 (nn.LayerNorm): 自注意力后的层归一化。
        cross_attn_token_to_image (Attention): 从查询到键的交叉注意力层。
        norm2 (nn.LayerNorm): 第二次注意力块后的层归一化。
        mlp (MLPBlock): 用于转换查询嵌入的MLP块。
        norm3 (nn.LayerNorm): MLP块后的层归一化。
        norm4 (nn.LayerNorm): 第三次注意力块后的层归一化。
        cross_attn_image_to_token (Attention): 从键到查询的交叉注意力层。
        skip_first_layer_pe (bool): 在第一层中跳过位置编码的标志。

    Methods:
        forward: 将自注意力和交叉注意力应用于查询和键。

    Examples:
        >>> embedding_dim, num_heads = 256, 8
        >>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
        >>> queries = torch.randn(1, 100, embedding_dim)
        >>> keys = torch.randn(1, 1000, embedding_dim)
        >>> query_pe = torch.randn(1, 100, embedding_dim)
        >>> key_pe = torch.randn(1, 1000, embedding_dim)
        >>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """初始化TwoWayAttentionBlock，用于同时对图像和查询点进行注意力计算。

            此块实现了四个主要组件：稀疏输入上的自注意力、
            从稀疏输入到密集输入的交叉注意力、稀疏输入上的MLP块，以及
            从密集输入到稀疏输入的交叉注意力。

            Args:
                embedding_dim (int): 嵌入的通道维度。
                num_heads (int): 注意力层中的注意力头数量。
                mlp_dim (int, optional): MLP块的内部通道维度。
                activation (type[nn.Module], optional): MLP块中使用的激活函数。
                attention_downsample_rate (int, optional): 注意力机制的下采样率。
                skip_first_layer_pe (bool, optional): 是否在第一层中跳过位置编码。
            """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """应用双向注意力来处理查询和键嵌入。

            Args:
                queries (torch.Tensor): 查询嵌入，形状为(B, N_queries, embedding_dim)。
                keys (torch.Tensor): 键嵌入，形状为(B, N_keys, embedding_dim)。
                query_pe (torch.Tensor): 查询的位置编码，形状与查询相同。
                key_pe (torch.Tensor): 键的位置编码，形状与键相同。

            Returns:
                queries (torch.Tensor): 处理后的查询嵌入，形状为(B, N_queries, embedding_dim)。
                keys (torch.Tensor): 处理后的键嵌入，形状为(B, N_keys, embedding_dim)。
            """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """一个具有下采样能力的注意力层，用于投影后的嵌入大小。

    This class implements a multi-head attention mechanism with the option to downsample the internal dimension of
    queries, keys, and values.

    Attributes:
        embedding_dim (int): Dimensionality of input embeddings.
        kv_in_dim (int): Dimensionality of key and value inputs.
        internal_dim (int): Internal dimension after downsampling.
        num_heads (int): Number of attention heads.
        q_proj (nn.Linear): Linear projection for queries.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear projection for output.

    Methods:
        _separate_heads: Separate input tensor into attention heads.
        _recombine_heads: Recombine separated attention heads.
        forward: Compute attention output for given query, key, and value tensors.

    Examples:
        >>> attn = Attention(embedding_dim=256, num_heads=8, downsample_rate=2)
        >>> q = torch.randn(1, 100, 256)
        >>> k = v = torch.randn(1, 50, 256)
        >>> output = attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int | None = None,
    ) -> None:
        """Initialize the Attention module with specified dimensions and settings.

        Args:
            embedding_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            downsample_rate (int, optional): Factor by which internal dimensions are downsampled.
            kv_in_dim (int | None, optional): Dimensionality of key and value inputs. If None, uses embedding_dim.

        Raises:
            AssertionError: If num_heads does not evenly divide the internal dim (embedding_dim / downsample_rate).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    @staticmethod
    def _separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Separate the input tensor into the specified number of attention heads."""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    @staticmethod
    def _recombine_heads(x: Tensor) -> Tensor:
        """Recombine separated attention heads into a single tensor."""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention to query, key, and value tensors with optional downsampling.

        Args:
            q (torch.Tensor): Query tensor with shape (B, N_q, embedding_dim).
            k (torch.Tensor): Key tensor with shape (B, N_k, kv_in_dim).
            v (torch.Tensor): Value tensor with shape (B, N_k, kv_in_dim).

        Returns:
            (torch.Tensor): Output tensor after attention with shape (B, N_q, embedding_dim).
        """
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)
