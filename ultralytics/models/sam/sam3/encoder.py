# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

# 版权所有 (c) Meta Platforms, Inc. 及其附属公司。保留所有权利。
# 基于 https://github.com/IDEA-Research/GroundingDINO
from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules.utils import _get_clones

from .model_misc import get_valid_ratio


class TransformerEncoderLayer(nn.Module):
    """执行自注意力后接交叉注意力的 Transformer 编码器层。

    该层之前被称为 TransformerDecoderLayer，但被重命名以更好地反映其在架构中的角色。
    它通过自注意力处理输入序列，然后与另一个输入（通常是图像特征）进行交叉注意力。

    该层支持 pre-norm 和 post-norm 配置，以及在注意力机制不同阶段的位置编码。
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module = None,
        cross_attention: nn.Module = None,
    ):
        """初始化 Transformer 编码器层。

        Args:
            d_model: 模型维度/隐藏层大小
            dim_feedforward: 前馈网络维度
            dropout: Dropout 概率
            pos_enc_at_attn: 是否在自注意力中添加位置编码
            pos_enc_at_cross_attn_keys: 是否在交叉注意力的键中添加位置编码
            pos_enc_at_cross_attn_queries: 是否在交叉注意力的查询中添加位置编码
            pre_norm: 是否使用 pre-norm（True）或 post-norm（False）架构
            self_attention: 自注意力模块
            cross_attention: 用于关注图像特征的交叉注意力模块
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention or nn.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256)
        self.cross_attn_image = cross_attention or nn.MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256)

        # 前馈模型的实现
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
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.layer_idx = None

    def forward_post(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        pos: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """post-norm 架构的前向传播。

        在 post-norm 架构中，归一化在注意力和前馈操作之后应用。

        Args:
            tgt (torch.Tensor): 待处理的输入张量。
            memory (torch.Tensor): 用于交叉注意力的记忆张量。
            tgt_mask (torch.Tensor): 自注意力的掩码。
            memory_mask (torch.Tensor): 交叉注意力的掩码。
            tgt_key_padding_mask (torch.Tensor): 自注意力的键填充掩码。
            memory_key_padding_mask (torch.Tensor): 交叉注意力的键填充掩码。
            pos (torch.Tensor): 记忆的位置编码。
            query_pos (torch.Tensor): 查询的位置编码。
            **kwargs (Any): 额外的关键字参数。

        Returns:
            处理后的张量
        """
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        # 自注意力
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 对图像的交叉注意力
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        dac: bool = False,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        pos: torch.Tensor = None,
        query_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """pre-norm 架构的前向传播。

        在 pre-norm 架构中，归一化在注意力和前馈操作之前应用。

        Args:
            tgt: 待处理的输入张量
            memory: 用于交叉注意力的记忆张量
            dac: 是否使用分而治之注意力
            tgt_mask: 自注意力的掩码
            memory_mask: 交叉注意力的掩码
            tgt_key_padding_mask: 自注意力的键填充掩码
            memory_key_padding_mask: 交叉注意力的键填充掩码
            pos: 记忆的位置编码
            query_pos: 查询的位置编码

        Returns:
            处理后的张量
        """
        if dac:
            # 我们仅对前半部分查询应用自注意力
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt).contiguous()
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            # 重新合并
            tgt = torch.cat((tgt, other_tgt), dim=0)
        tgt2 = self.norm2(tgt)
        memory = memory.to(tgt2.dtype).contiguous()
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        dac: bool = False,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        pos: torch.Tensor = None,
        query_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """Transformer 编码器层的前向传播。

        Args:
            tgt: 待处理的输入张量
            memory: 用于交叉注意力的记忆张量（如图像特征）
            dac: 是否使用分而治之注意力（仅对前半部分查询应用自注意力）
            tgt_mask: 自注意力的掩码
            memory_mask: 交叉注意力的掩码
            tgt_key_padding_mask: 自注意力的键填充掩码
            memory_key_padding_mask: 交叉注意力的键填充掩码
            pos: 记忆的位置编码
            query_pos: 查询的位置编码

        Returns:
            经过自注意力、交叉注意力和前馈网络处理后的张量
        """
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            # attn_bias=attn_bias,
            # **kwds,
        )


class TransformerEncoder(nn.Module):
    """处理多级特征的 Transformer 编码器。

    该编码器接收多级特征（例如来自骨干网络），并通过堆叠的 Transformer 编码器层进行处理。
    它支持来自多个级别（例如不同分辨率）的特征，并可在训练时应用激活检查点以节省内存。

    Args:
        layer: 被堆叠多次的编码器层
        num_layers: 要堆叠的编码器层数
        d_model: 模型维度/隐藏层大小
        num_feature_levels: 要处理的特征级别数
        frozen: 是否冻结该模块的参数
        use_act_checkpoint: 训练时是否使用激活检查点
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        frozen: bool = False,
        use_act_checkpoint: bool = False,
    ):
        """初始化 Transformer 编码器。"""
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers

        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.use_act_checkpoint = use_act_checkpoint

        # 为每层分配层索引，以便某些层可以根据自身层索引
        # 决定行为（例如，仅在选定的层中对 memory bank 执行交叉注意力）
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        """为 Transformer 编码器准备多级特征。"""
        assert len(srcs) == self.num_feature_levels, "mismatch between expected and received # of feature levels"

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        has_mask = masks is not None and masks[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            _, _, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        if has_mask:
            valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
        else:
            valid_ratios = torch.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2),
                device=src_flatten.device,
                dtype=src_flatten.dtype,
            )

        return (
            src_flatten,
            mask_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        )

    def forward(
        self,
        src: list[torch.Tensor],
        src_key_padding_masks: list[torch.Tensor] | None = None,
        pos: list[torch.Tensor] | None = None,
        prompt: torch.Tensor = None,
        prompt_key_padding_mask: torch.Tensor = None,
        encoder_extra_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """通过 Transformer 编码器处理多级特征。

        Args:
            src: 多级特征列表，每个形状为 (batch_size, channels, height, width)
            src_key_padding_masks: 每个特征级别的填充掩码列表，每个形状为 (batch_size, height, width)
            pos: 每个特征级别的位置嵌入列表，每个形状为 (batch_size, channels, height, width)
            prompt: 可选的文本/提示特征，形状为 (seq_len, batch_size, d_model)
            prompt_key_padding_mask: 提示的可选填充掩码，形状为 (batch_size, seq_len)
            encoder_extra_kwargs: 传递给每个编码器层的可选额外参数

        Returns:
            包含以下内容的元组：
            - output: 处理后的特征，形状为 (seq_len, batch_size, d_model)
            - key_padding_masks_flatten: 展平的填充掩码
            - lvl_pos_embed_flatten: 展平的位置嵌入
            - level_start_index: 每个特征级别的起始索引
            - spatial_shapes: 每个特征级别的空间维度
            - valid_ratios: 每个特征级别的有效比率
        """
        assert len(src) == self.num_feature_levels, "must be equal to num_feature_levels"
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels
        # 展平多级特征并添加级别位置嵌入
        (
            src_flatten,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)

        output = src_flatten
        for layer in self.layers:
            layer_kwargs = {}

            assert isinstance(layer, TransformerEncoderLayer)
            layer_kwargs["memory"] = prompt
            layer_kwargs["memory_key_padding_mask"] = prompt_key_padding_mask
            layer_kwargs["query_pos"] = lvl_pos_embed_flatten
            layer_kwargs["tgt"] = output
            layer_kwargs["tgt_key_padding_mask"] = key_padding_masks_flatten

            if self.training:
                assert self.use_act_checkpoint, "activation ckpt not enabled in encoder"
            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = layer(**layer_kwargs)
        # 以序列优先顺序返回
        return (
            output.transpose(0, 1),
            (key_padding_masks_flatten.transpose(0, 1) if key_padding_masks_flatten is not None else None),
            lvl_pos_embed_flatten.transpose(0, 1),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )


class TransformerEncoderFusion(TransformerEncoder):
    """融合文本和图像特征的 Transformer 编码器。

    该编码器扩展了 TransformerEncoder 以处理文本和图像特征，能够将池化的文本特征
    添加到图像特征中以实现更好的跨模态融合。它支持 torch.compile 进行性能优化。

    Args:
        layer (nn.Module): 被堆叠多次的编码器层。
        num_layers (int): 要堆叠的编码器层数。
        d_model (int): 模型维度/隐藏层大小。
        num_feature_levels (int): 要处理的特征级别数。
        add_pooled_text_to_img_feat (bool): 是否将池化的文本特征添加到图像特征。
        pool_text_with_mask (bool): 池化文本特征时是否使用掩码。
        compile_mode (str | None): torch.compile 的模式，None 则禁用编译。
        **kwargs (Any): 传递给父类的额外参数。
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        add_pooled_text_to_img_feat: bool = True,
        pool_text_with_mask: bool = False,
        compile_mode: str | None = None,
        **kwargs,
    ):
        """初始化具有文本-图像融合的 Transformer 编码器。"""
        super().__init__(
            layer,
            num_layers,
            d_model,
            num_feature_levels,
            **kwargs,
        )
        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = nn.Linear(d_model, d_model)
        self.pool_text_with_mask = pool_text_with_mask
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)

    def forward(
        self,
        src: list[torch.Tensor],
        prompt: torch.Tensor,
        src_key_padding_mask: list[torch.Tensor] | None = None,
        src_pos: list[torch.Tensor] | None = None,
        prompt_key_padding_mask: torch.Tensor = None,
        feat_sizes: list[int] | None = None,
        encoder_extra_kwargs: dict | None = None,
    ):
        """具有文本-图像融合的 Transformer 编码器的前向传播。"""
        # 恢复视觉的空间形状
        bs = src[0].shape[1]  # seq first
        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            if src_key_padding_mask is None:
                src_key_padding_mask = [None] * len(src)
            for i, (h, w) in enumerate(feat_sizes):
                src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_key_padding_mask[i] = (
                    src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1)
                    if src_key_padding_mask[i] is not None
                    else None
                )
        else:
            assert all(x.dim == 4 for x in src), "expected list of (bs, c, h, w) tensors"

        if self.add_pooled_text_to_img_feat:
            # 融合：将均值池化的文本特征添加到图像特征
            pooled_text = pool_text_feat(prompt, prompt_key_padding_mask, self.pool_text_with_mask)
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]  # prompt 为序列优先
            src = [x.add_(pooled_text) for x in src]

        (
            out,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
        ) = super().forward(
            src,
            src_key_padding_masks=src_key_padding_mask,
            pos=src_pos,
            prompt=prompt.transpose(0, 1),
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        return {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }


def pool_text_feat(prompt, prompt_mask, pool_with_mask):
    """仅对有效 token 进行均值池化提示嵌入。"""
    # prompt 的形状为 (seq, bs, dim)
    if not pool_with_mask:
        return prompt.mean(dim=0)

    # prompt_mask 的形状为 (bs, seq)，False 表示有效，True 表示填充
    assert prompt_mask.dim() == 2
    # is_valid 的形状为 (seq, bs, 1)，1 表示有效，0 表示填充
    is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
    # num_valid 的形状为 (bs, 1)
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)

    # 对所有有效 token 进行均值池化
    pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
    return pooled_text
