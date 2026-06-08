# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import LayerNorm2d

from .blocks import (
    Block,
    CXBlock,
    Fuser,
    MaskDownSampler,
    MultiScaleBlock,
    PatchEmbed,
    PositionEmbeddingRandom,
    PositionEmbeddingSine,
)


class ImageEncoderViT(nn.Module):
    """使用Vision Transformer（ViT）架构将图像编码到紧凑潜在空间的图像编码器。

    此类通过将图像分割为图块、应用transformer块并通过neck模块生成最终编码表示来处理图像。

    Attributes:
        img_size (int): 输入图像的尺寸，假设为正方形。
        patch_embed (PatchEmbed): 图块嵌入模块。
        pos_embed (nn.Parameter | None): 图块的绝对位置嵌入。
        blocks (nn.ModuleList): 用于处理图块嵌入的transformer块列表。
        neck (nn.Sequential): 进一步处理输出的neck模块。

    Methods:
        forward: 通过图块嵌入、位置嵌入、blocks和neck处理输入。

    Examples:
        >>> import torch
        >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
        >>> input_image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(input_image)
        >>> print(output.shape)
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        """初始化ImageEncoderViT实例，使用Vision Transformer架构进行图像编码。

        Args:
            img_size (int): 输入图像尺寸，假设为正方形。
            patch_size (int): 图像图块的大小。
            in_chans (int): 输入图像的通道数。
            embed_dim (int): 图块嵌入的维度。
            depth (int): transformer块的数量。
            num_heads (int): 每个块中注意力头的数量。
            mlp_ratio (float): MLP隐藏维度与嵌入维度的比率。
            out_chans (int): neck模块的输出通道数。
            qkv_bias (bool): 如果为True，为query、key、value投影添加可学习的偏置。
            norm_layer (type[nn.Module]): 要使用的归一化层类型。
            act_layer (type[nn.Module]): 要使用的激活层类型。
            use_abs_pos (bool): 如果为True，使用绝对位置嵌入。
            use_rel_pos (bool): 如果为True，为注意力图添加相对位置嵌入。
            rel_pos_zero_init (bool): 如果为True，将相对位置参数初始化为零。
            window_size (int): 窗口注意力块的注意力窗口大小。
            global_attn_indexes (tuple[int, ...]): 使用全局注意力的块索引。
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: nn.Parameter | None = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过图块嵌入、位置嵌入、transformer块和neck模块处理输入。"""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = (
                F.interpolate(self.pos_embed.permute(0, 3, 1, 2), scale_factor=self.img_size / 1024).permute(0, 2, 3, 1)
                if self.img_size != 1024
                else self.pos_embed
            )
            x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


class PromptEncoder(nn.Module):
    """为SAM的掩码解码器编码不同类型的提示，生成稀疏和密集嵌入。

    Attributes:
        embed_dim (int): 嵌入的维度。
        input_image_size (tuple[int, int]): 输入图像的尺寸，格式为(H, W)。
        image_embedding_size (tuple[int, int]): 图像嵌入的空间尺寸，格式为(H, W)。
        pe_layer (PositionEmbeddingRandom): 随机位置嵌入模块。
        num_point_embeddings (int): 不同类型点的点嵌入数量。
        point_embeddings (nn.ModuleList): 点嵌入的列表。
        not_a_point_embed (nn.Embedding): 不属于任何标签的点的嵌入。
        mask_input_size (tuple[int, int]): 输入掩码的尺寸。
        mask_downscaling (nn.Sequential): 用于下采样掩码的神经网络。
        no_mask_embed (nn.Embedding): 未提供掩码时的嵌入。

    Methods:
        get_dense_pe: 返回用于编码点提示的位置编码。
        forward: 嵌入不同类型的提示，返回稀疏和密集嵌入。

    Examples:
        >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
        >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
        >>> boxes = torch.rand(1, 2, 2)
        >>> masks = torch.rand(1, 1, 256, 256)
        >>> sparse_embeddings, dense_embeddings = prompt_encoder(points, boxes, masks)
        >>> print(sparse_embeddings.shape, dense_embeddings.shape)
        torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        """初始化PromptEncoder模块，用于编码各种类型的提示。

        Args:
            embed_dim (int): 嵌入的维度。
            image_embedding_size (tuple[int, int]): 图像嵌入的空间尺寸，格式为(H, W)。
            input_image_size (tuple[int, int]): 输入图像的填充尺寸，格式为(H, W)。
            mask_in_chans (int): 用于编码输入掩码的隐藏通道数。
            activation (type[nn.Module]): 编码输入掩码时使用的激活函数。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """返回用于编码点提示的密集位置编码。

        生成与图像编码形状匹配的密集点集位置编码。该编码用于在处理点提示时为模型提供空间信息。

        Returns:
            (torch.Tensor): 形状为(1, embed_dim, H, W)的位置编码张量，其中H和W分别为图像嵌入大小的高度和宽度。

        Examples:
            >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> dense_pe = prompt_encoder.get_dense_pe()
            >>> print(dense_pe.shape)
            torch.Size([1, 256, 64, 64])
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """通过应用位置编码和标签特定嵌入来嵌入点提示。"""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), dtype=points.dtype, device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), dtype=labels.dtype, device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """通过应用位置编码并添加角点嵌入来嵌入框提示。"""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """通过下采样和卷积层处理来嵌入掩码输入。"""
        return self.mask_downscaling(masks)

    @staticmethod
    def _get_batch_size(
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> int:
        """根据输入提示的批量大小获取输出的批量大小。"""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None,
        boxes: torch.Tensor | None,
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """嵌入不同类型的提示，返回稀疏和密集嵌入。

        Args:
            points (tuple[torch.Tensor, torch.Tensor] | None): 要嵌入的点坐标和标签。第一个张量包含形状为(B, N, 2)的坐标，
                第二个张量包含形状为(B, N)的标签。
            boxes (torch.Tensor | None): 要嵌入的框，形状为(B, M, 2, 2)，其中M是框的数量。
            masks (torch.Tensor | None): 要嵌入的掩码，形状为(B, 1, H, W)。

        Returns:
            sparse_embeddings (torch.Tensor): 点和框的稀疏嵌入，形状为(B, N, embed_dim)。
            dense_embeddings (torch.Tensor): 掩码的密集嵌入，形状为(B, embed_dim, embed_H, embed_W)。

        Examples:
            >>> encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
            >>> boxes = torch.rand(1, 2, 2, 2)
            >>> masks = torch.rand(1, 1, 256, 256)
            >>> sparse_emb, dense_emb = encoder(points, boxes, masks)
            >>> print(sparse_emb.shape, dense_emb.shape)
            torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim),
            dtype=self.point_embeddings[0].weight.dtype,
            device=self.point_embeddings[0].weight.device,
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class MemoryEncoder(nn.Module):
    """将像素特征和掩码编码为内存表示，用于高效图像分割。

    此类处理像素级特征和掩码，融合它们以生成适合SAM（Segment Anything Model）等图像分割模型下游任务
    的编码内存表示。

    Attributes:
        mask_downsampler (MaskDownSampler): 用于下采样输入掩码的模块。
        pix_feat_proj (nn.Conv2d): 用于投影像素特征的卷积层。
        fuser (Fuser): 用于融合像素特征和掩码的模块。
        position_encoding (PositionEmbeddingSine): 用于为特征添加位置编码的模块。
        out_proj (nn.Module): 输出投影层，可以是nn.Identity或nn.Conv2d。

    Methods:
        forward: 处理输入像素特征和掩码以生成编码的内存表示。

    Examples:
        >>> import torch
        >>> encoder = MemoryEncoder(out_dim=256, in_dim=256)
        >>> pix_feat = torch.randn(1, 256, 64, 64)
        >>> masks = torch.randn(1, 1, 64, 64)
        >>> encoded_feat, pos = encoder(pix_feat, masks)
        >>> print(encoded_feat.shape, pos.shape)
        torch.Size([1, 256, 64, 64]) torch.Size([1, 128, 64, 64])
    """

    def __init__(
        self,
        out_dim,
        in_dim=256,  # in_dim of pix_feats
        interpol_size: tuple[int, int] | None = None,
    ):
        """初始化MemoryEncoder，用于将像素特征和掩码编码为内存表示。

        此编码器处理像素级特征和掩码，融合它们以生成适合SAM（Segment Anything Model）等图像分割模型
        下游任务的编码内存表示。

        Args:
            out_dim (int): 编码特征的输出维度。
            in_dim (int): 像素特征的输入维度。
            interpol_size (tuple[int, int] | None): 将掩码插值到的尺寸。如果为None，则使用像素特征的尺寸。
        """
        super().__init__()

        self.mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1, interpol_size=interpol_size)

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(CXBlock(dim=256), num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=64)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> dict:
        """处理像素特征和掩码以生成用于分割的编码内存表示。"""
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        # Fuse pix_feats and downsampled masks, in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}


class ImageEncoder(nn.Module):
    """使用trunk-neck架构编码图像，生成多尺度特征和位置编码。

    此类将用于特征提取的trunk网络与用于特征细化和位置编码生成的neck网络相结合。
    它可以选择性地丢弃最低分辨率的特征。

    Attributes:
        trunk (nn.Module): 用于初始特征提取的trunk网络。
        neck (nn.Module): 用于特征细化和位置编码生成的neck网络。
        scalp (int): 要丢弃的最低分辨率特征级别数量。

    Methods:
        forward: 通过trunk和neck网络处理输入图像。

    Examples:
        >>> trunk = SomeTrunkNetwork()
        >>> neck = SomeNeckNetwork()
        >>> encoder = ImageEncoder(trunk, neck, scalp=1)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(image)
        >>> print(output.keys())
        dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
    """

    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        """初始化ImageEncoder，使用trunk和neck网络进行特征提取和细化。

        此编码器将用于特征提取的trunk网络与用于特征细化和位置编码生成的neck网络相结合。
        它可以选择性地丢弃最低分辨率的特征。

        Args:
            trunk (nn.Module): 用于初始特征提取的trunk网络。
            neck (nn.Module): 用于特征细化和位置编码生成的neck网络。
            scalp (int): 要丢弃的最低分辨率特征级别数量。
        """
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel dims of trunk {self.trunk.channel_list} and neck {self.neck.backbone_channel_list} do not match."
        )

    def forward(self, sample: torch.Tensor):
        """通过trunk和neck网络编码输入，返回多尺度特征和位置编码。"""
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        return {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }


class FpnNeck(nn.Module):
    """一种用于目标检测模型中多尺度特征融合的Feature Pyramid Network（FPN）neck变体。

    此FPN变体移除了输出卷积，并使用双三次插值进行特征尺寸调整，类似于ViT位置嵌入插值。

    Attributes:
        position_encoding (PositionEmbeddingSine): 正弦位置编码模块。
        convs (nn.ModuleList): 每个backbone级别的卷积层列表。
        backbone_channel_list (list[int]): backbone的通道维度列表。
        fpn_interp_model (str): FPN特征尺寸调整的插值模式。
        fuse_type (str): 特征融合类型，'sum'或'avg'。
        fpn_top_down_levels (list[int]): 输出中具有自上而下特征的级别。

    Methods:
        forward: 执行FPN neck的前向传播。

    Examples:
        >>> backbone_channels = [64, 128, 256, 512]
        >>> fpn_neck = FpnNeck(256, backbone_channels)
        >>> inputs = [torch.rand(1, c, 32, 32) for c in backbone_channels]
        >>> outputs, positions = fpn_neck(inputs)
        >>> print(len(outputs), len(positions))
        4 4
    """

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: list[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: list[int] | None = None,
    ):
        """初始化修改后的Feature Pyramid Network（FPN）neck。

        此FPN变体移除了输出卷积，并使用双三次插值进行特征尺寸调整，类似于ViT位置嵌入插值。

        Args:
            d_model (int): 模型的维度。
            backbone_channel_list (list[int]): backbone的通道维度列表。
            kernel_size (int): 卷积层的卷积核大小。
            stride (int): 卷积层的步幅。
            padding (int): 卷积层的填充。
            fpn_interp_model (str): FPN特征尺寸调整的插值模式。
            fuse_type (str): 特征融合类型，'sum'或'avg'。
            fpn_top_down_levels (list[int] | None): 输出中具有自上而下特征的级别。
        """
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in {"sum", "avg"}
        self.fuse_type = fuse_type

        # Levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level
        if fpn_top_down_levels is None:
            # Default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: list[torch.Tensor]):
        """执行Feature Pyramid Network（FPN）neck的前向传播。

        此方法通过FPN处理来自backbone的输入张量列表，应用横向连接和自上而下的特征融合。
        它生成输出特征图和相应的位置编码。

        Args:
            xs (list[torch.Tensor]): 来自backbone的输入张量列表，每个张量形状为(B, C, H, W)。

        Returns:
            out (list[torch.Tensor]): FPN处理后输出特征图的列表，每个形状为(B, d_model, H, W)。
            pos (list[torch.Tensor]): 对应每个输出特征图的位置编码列表。

        Examples:
            >>> fpn_neck = FpnNeck(d_model=256, backbone_channel_list=[64, 128, 256, 512])
            >>> inputs = [torch.rand(1, c, 32, 32) for c in [64, 128, 256, 512]]
            >>> outputs, positions = fpn_neck(inputs)
            >>> print(len(outputs), len(positions))
            4 4
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # FPN forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # Forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=x.dtype),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


class Hiera(nn.Module):
    """层次化视觉transformer，用于图像处理任务中的高效多尺度特征提取。

    此类实现了Hiera模型，这是一种设计用于高效多尺度特征提取的层次化视觉transformer架构。
    它使用按阶段组织的一系列transformer块，具有可选的下采样和全局注意力机制。

    Attributes:
        window_spec (tuple[int, ...]): 每个阶段的窗口大小。
        q_stride (tuple[int, int]): 阶段之间的下采样步幅。
        stage_ends (list[int]): 每个阶段最后一个块的索引。
        q_pool_blocks (list[int]): 应用下采样的块索引。
        return_interm_layers (bool): 是否返回中间层输出。
        patch_embed (PatchEmbed): 图块嵌入模块。
        global_att_blocks (tuple[int, ...]): 使用全局注意力的块索引。
        window_pos_embed_bkg_spatial_size (tuple[int, int]): 窗口位置嵌入背景的空间尺寸。
        pos_embed (nn.Parameter): 背景的位置嵌入。
        pos_embed_window (nn.Parameter): 窗口的位置嵌入。
        blocks (nn.ModuleList): MultiScaleBlock模块的列表。
        channel_list (list[int]): 每个阶段输出通道维度的列表。

    Methods:
        _get_pos_embed: 通过插值和组合窗口及背景嵌入来生成位置嵌入。
        forward: 执行Hiera模型的前向传播。

    Examples:
        >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output_features = model(input_tensor)
        >>> for feat in output_features:
        ...     print(feat.shape)
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        """初始化Hiera模型，一种用于高效多尺度特征提取的层次化视觉transformer。

        Hiera是一种设计用于图像处理任务中高效多尺度特征提取的层次化视觉transformer架构。
        它使用按阶段组织的一系列transformer块，具有可选的下采样和全局注意力机制。

        Args:
            embed_dim (int): 模型的初始嵌入维度。
            num_heads (int): 初始注意力头数。
            drop_path_rate (float): 随机深度率。
            q_pool (int): 查询下采样阶段的数量。
            q_stride (tuple[int, int]): 阶段之间的下采样步幅。
            stages (tuple[int, ...]): 每个阶段的块数量。
            dim_mul (float): 阶段转换时的维度倍增因子。
            head_mul (float): 阶段转换时的头数倍增因子。
            window_pos_embed_bkg_spatial_size (tuple[int, int]): 窗口位置嵌入背景的空间尺寸。
            window_spec (tuple[int, ...]): 不使用全局注意力时每个阶段的窗口大小。
            global_att_blocks (tuple[int, ...]): 使用全局注意力的块索引。
            return_interm_layers (bool): 是否返回中间层输出。
        """
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
        )
        # Which blocks have global attention?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # Lags by a block, so first block of next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: tuple[int, int]) -> torch.Tensor:
        """通过插值和组合窗口及背景嵌入来生成位置嵌入。"""
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """执行Hiera模型的前向传播，从输入图像中提取多尺度特征。

        Args:
            x (torch.Tensor): 表示一批图像的输入张量，形状为(B, C, H, W)。

        Returns:
            (list[torch.Tensor]): 不同尺度的特征图列表，每个形状为(B, C_i, H_i, W_i)，其中C_i是通道维度，
                H_i、W_i是尺度i的空间维度。如果return_interm_layers为True，列表从最高分辨率（精细特征）
                到最低分辨率（粗略特征）排序，否则仅包含最终输出。

        Examples:
            >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
            >>> input_tensor = torch.randn(1, 3, 224, 224)
            >>> output_features = model(input_tensor)
            >>> for feat in output_features:
            ...     print(feat.shape)
        """
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add positional embedding
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
