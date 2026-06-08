# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from ultralytics.nn.modules import MLP
from ultralytics.utils import LOGGER

from .blocks import SAM2TwoWayTransformer, TwoWayTransformer
from .decoders import MaskDecoder, SAM2MaskDecoder
from .encoders import ImageEncoderViT, PromptEncoder
from .utils import get_1d_sine_pe, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAMModel(nn.Module):
    """用于目标分割任务的Segment Anything Model (SAM)。

    此类结合了图像编码器、提示编码器和掩码解码器，用于从图像和输入
    提示中预测目标掩码。

    Attributes:
        mask_threshold (float): 掩码预测的阈值。
        image_encoder (ImageEncoderViT): 将图像编码为嵌入的主干网络。
        prompt_encoder (PromptEncoder): 用于各种类型输入提示的编码器。
        mask_decoder (MaskDecoder): 从图像和提示嵌入中预测目标掩码。
        pixel_mean (torch.Tensor): 输入图像中像素归一化的均值。
        pixel_std (torch.Tensor): 输入图像中像素归一化的标准差。

    Methods:
        set_imgsz: 设置图像大小以使模型兼容不同的图像尺寸。

    Examples:
        >>> image_encoder = ImageEncoderViT(...)
        >>> prompt_encoder = PromptEncoder(...)
        >>> mask_decoder = MaskDecoder(...)
        >>> sam_model = SAMModel(image_encoder, prompt_encoder, mask_decoder)
        >>> # 后续使用取决于SAMPredictor类

    Notes:
        所有forward()操作都在SAMPredictor类中实现。
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: list[float] = (123.675, 116.28, 103.53),
        pixel_std: list[float] = (58.395, 57.12, 57.375),
    ) -> None:
        """初始化SAMModel类，从图像和输入提示中预测目标掩码。

        Args:
            image_encoder (ImageEncoderViT): 用于将图像编码为图像嵌入的主干网络。
            prompt_encoder (PromptEncoder): 编码各种类型的输入提示。
            mask_decoder (MaskDecoder): 从图像嵌入和编码后的提示中预测掩码。
            pixel_mean (list[float]): 输入图像中像素归一化的均值。
            pixel_std (list[float]): 输入图像中像素归一化的标准差。

        Notes:
            所有forward()操作已移至SAMPredictor。
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def set_imgsz(self, imgsz):
        """Set image size to make model compatible with different image sizes."""
        if hasattr(self.image_encoder, "set_imgsz"):
            self.image_encoder.set_imgsz(imgsz)
        self.prompt_encoder.input_image_size = imgsz
        self.prompt_encoder.image_embedding_size = [x // 16 for x in imgsz]  # 16 is fixed as patch size of ViT model
        self.image_encoder.img_size = imgsz[0]


class SAM2Model(torch.nn.Module):
    """具有基于内存的视频目标分割能力的SAM2Model类。

    此类扩展了SAM的功能以处理视频序列，结合了内存机制来确保时间
    一致性以及跨帧的高效目标跟踪。

    Attributes:
        mask_threshold (float): 掩码预测的阈值。
        image_encoder (ImageEncoderViT): 用于提取图像特征的视觉编码器。
        memory_attention (nn.Module): 用于关注内存特征的模块。
        memory_encoder (nn.Module): 用于生成内存表示的编码器。
        num_maskmem (int): 可访问的内存帧数量。
        image_size (int): 输入图像的大小。
        backbone_stride (int): 主干网络输出的步幅。
        sam_prompt_embed_dim (int): SAM提示嵌入的维度。
        sam_image_embedding_size (int): SAM图像嵌入的大小。
        sam_prompt_encoder (PromptEncoder): 用于处理输入提示的编码器。
        sam_mask_decoder (SAM2MaskDecoder): 用于生成目标掩码的解码器。
        obj_ptr_proj (nn.Module): 目标指针的投影层。
        obj_ptr_tpos_proj (nn.Module): 目标指针中时间位置编码的投影。
        hidden_dim (int): 模型的隐藏维度。
        mem_dim (int): 用于编码特征的内存维度。
        use_high_res_features_in_sam (bool): 是否在SAM掩码解码器中使用高分辨率特征图。
        use_obj_ptrs_in_encoder (bool): 是否在编码器中交叉关注来自其他帧的目标指针。
        max_obj_ptrs_in_encoder (int): 编码器中来自其他帧的最大目标指针数量。
        add_tpos_enc_to_obj_ptrs (bool): 是否向目标指针添加时间位置编码。
        proj_tpos_enc_in_obj_ptrs (bool): 是否为目标指针中的时间位置编码添加额外的线性投影层。
        use_signed_tpos_enc_to_obj_ptrs (bool): 是否在时间位置编码中使用有符号距离。
        only_obj_ptrs_in_the_past_for_eval (bool): 是否在评估期间仅关注过去的目标指针。
        pred_obj_scores (bool): 是否预测帧中是否存在目标。
        pred_obj_scores_mlp (bool): 是否使用MLP预测目标分数。
        fixed_no_obj_ptr (bool): 当没有目标存在时是否使用固定的无目标指针。
        soft_no_obj_ptr (bool): 是否柔和地混合无目标指针以便于恢复和错误缓解。
        use_mlp_for_obj_ptr_proj (bool): 是否使用MLP进行目标指针投影。
        no_obj_embed_spatial (torch.Tensor | None): 空间帧的无目标嵌入。
        max_cond_frames_in_attn (int): 参与内存注意力的最大调节帧数。
        directly_add_no_mem_embed (bool): 是否在第一帧上直接将无内存嵌入添加到图像特征。
        multimask_output_in_sam (bool): 是否在初始调节帧的第一次点击时输出多个掩码。
        multimask_min_pt_num (int): 在SAM中使用多掩码输出的最小点击次数。
        multimask_max_pt_num (int): 在SAM中使用多掩码输出的最大点击次数。
        multimask_output_for_tracking (bool): 是否使用多掩码输出进行跟踪。
        use_multimask_token_for_obj_ptr (bool): 是否使用多掩码令牌作为目标指针。
        iou_prediction_use_sigmoid (bool): 是否使用sigmoid将IoU预测限制在[0-1]范围内。
        memory_temporal_stride_for_eval (int): 评估期间内存 bank 的时间步幅。
        non_overlap_masks_for_mem_enc (bool): 是否在评估期间对内存编码器中的目标掩码应用非重叠约束。
        sigmoid_scale_for_mem_enc (float): 掩码sigmoid概率的缩放因子。
        sigmoid_bias_for_mem_enc (float): 掩码sigmoid概率的偏置因子。
        binarize_mask_from_pts_for_mem_enc (bool): 是否在评估期间对带有点击的交互帧上的sigmoid掩码logits进行二值化。
        use_mask_input_as_output_without_sam (bool): 是否在具有掩码输入的帧上直接使用输入掩码而不使用SAM提示编码器和掩码解码器。

    Methods:
        forward_image: 通过编码器处理图像批次以提取多层次特征。
        track_step: 执行单个跟踪步骤，更新目标掩码和内存特征。
        set_binarize: 为VideoPredictor设置二值化。
        set_imgsz: 设置图像大小以使模型兼容不同的图像尺寸。

    Examples:
        >>> model = SAM2Model(image_encoder, memory_attention, memory_encoder)
        >>> image_batch = torch.rand(1, 3, 512, 512)
        >>> features = model.forward_image(image_batch)
        >>> track_results = model.track_step(0, True, features, None, None, None, {})
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=512,
        backbone_stride=16,
        sigmoid_scale_for_mem_enc=1.0,
        sigmoid_bias_for_mem_enc=0.0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        """为具有基于内存跟踪的视频目标分割初始化SAM2Model。

        Args:
            image_encoder (nn.Module): 用于提取图像特征的视觉编码器。
            memory_attention (nn.Module): 用于关注内存特征的模块。
            memory_encoder (nn.Module): 用于生成内存表示的编码器。
            num_maskmem (int): 可访问的内存帧数量。
            image_size (int): 输入图像的大小。
            backbone_stride (int): 图像主干网络输出的步幅。
            sigmoid_scale_for_mem_enc (float): 掩码sigmoid概率的缩放因子。
            sigmoid_bias_for_mem_enc (float): 掩码sigmoid概率的偏置因子。
            binarize_mask_from_pts_for_mem_enc (bool): 是否在评估期间对带有点击的交互帧上的sigmoid掩码logits进行二值化。
            use_mask_input_as_output_without_sam (bool): 是否在具有掩码输入的帧上直接输出输入掩码，而不使用SAM提示编码器和掩码解码器。
            max_cond_frames_in_attn (int): 参与内存注意力的最大调节帧数。
            directly_add_no_mem_embed (bool): 是否在第一帧上直接将无内存嵌入添加到图像特征。
            use_high_res_features_in_sam (bool): 是否在SAM掩码解码器中使用高分辨率特征图。
            multimask_output_in_sam (bool): 是否在初始调节帧的第一次点击时输出多个掩码。
            multimask_min_pt_num (int): 在SAM中使用多掩码输出的最小点击次数。
            multimask_max_pt_num (int): 在SAM中使用多掩码输出的最大点击次数。
            multimask_output_for_tracking (bool): 是否使用多掩码输出进行跟踪。
            use_multimask_token_for_obj_ptr (bool): 是否使用多掩码令牌作为目标指针。
            iou_prediction_use_sigmoid (bool): 是否使用sigmoid将IoU预测限制在[0-1]范围内。
            memory_temporal_stride_for_eval (int): 评估期间内存 bank 的时间步幅。
            non_overlap_masks_for_mem_enc (bool): 是否在评估期间对内存编码器中的目标掩码应用非重叠约束。
            use_obj_ptrs_in_encoder (bool): 是否在编码器中交叉关注来自其他帧的目标指针。
            max_obj_ptrs_in_encoder (int): 编码器交叉注意力中来自其他帧的最大目标指针数。
            add_tpos_enc_to_obj_ptrs (bool): 是否在编码器中向目标指针添加时间位置编码。
            proj_tpos_enc_in_obj_ptrs (bool): 是否为目标指针中的时间位置编码添加额外的线性投影层。
            use_signed_tpos_enc_to_obj_ptrs (bool): 是否在目标指针的时间位置编码中使用有符号距离。
            only_obj_ptrs_in_the_past_for_eval (bool): 是否在评估期间仅关注过去的目标指针。
            pred_obj_scores (bool): 是否预测帧中是否存在目标。
            pred_obj_scores_mlp (bool): 是否使用MLP预测目标分数。
            fixed_no_obj_ptr (bool): 当没有目标存在时是否使用固定的无目标指针。
            soft_no_obj_ptr (bool): 是否柔和地混合无目标指针以便于恢复和错误缓解。
            use_mlp_for_obj_ptr_proj (bool): 是否使用MLP进行目标指针投影。
            no_obj_embed_spatial (bool): 是否使用空间无目标嵌入。
            sam_mask_decoder_extra_args: SAM掩码解码器的额外参数。
            compile_image_encoder (bool): 是否编译图像编码器。
        """
            pred_obj_scores (bool): Whether to predict if there is an object in the frame.
            pred_obj_scores_mlp (bool): Whether to use an MLP to predict object scores.
            fixed_no_obj_ptr (bool): Whether to have a fixed no-object pointer when there is no object present.
            soft_no_obj_ptr (bool): Whether to mix in no-object pointer softly for easier recovery and error mitigation.
            use_mlp_for_obj_ptr_proj (bool): Whether to use MLP for object pointer projection.
            no_obj_embed_spatial (bool): Whether to add no-object embedding to spatial frames.
            sam_mask_decoder_extra_args (dict | None): Extra arguments for constructing the SAM mask decoder.
            compile_image_encoder (bool): Whether to compile the image encoder for faster inference.
        """
        super().__init__()

        # Part 1: 图像主干网络
        self.image_encoder = image_encoder
        # 在高分辨率设置下使用级别0、1、2，或仅使用级别2作为默认设置
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # 一个卷积层，将掩码提示下采样到步幅4（与低分辨率SAM掩码logits的步幅相同），
            # 并将其尺度从0~1转换为SAM logit尺度，以便输入到SAM掩码解码器中生成指针。
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # 这些选项需要一起使用
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: 内存注意力，用于将当前帧的视觉特征与过去帧的内存（和目标指针）进行条件化
        self.memory_attention = memory_attention
        self.hidden_dim = memory_attention.d_model

        # Part 3: 前一帧输出的内存编码器
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(self.memory_encoder.out_proj, "weight"):
            # 如果内存在通道维度上有压缩
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # 可访问的内存数量
        # 内存的时间编码
        self.maskmem_tpos_enc = torch.nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # 一个单独的token，用于指示没有来自前帧的内存嵌入
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # 对输出的原始掩码logits应用sigmoid（将其从(-inf, +inf)范围转换到(0, 1)范围），
        # 然后再馈入内存编码器
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # 在具有掩码输入的帧上，是否直接输出输入掩码而不使用SAM提示编码器和掩码解码器
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM风格的提示编码器（用于掩码和点输入）
        # 和SAM风格的掩码解码器用于最终掩码输出
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.add_all_frames_to_correct_as_cond = True

        # 模型编译
        if compile_image_encoder:
            # 编译前向函数（而不是完整模块）以允许加载检查点。
            LOGGER.info("Image encoder compilation is enabled. First forward pass will be slow.")
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        """Return the device on which the model's parameters are stored."""
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        """Process image and prompt inputs to generate object masks and scores in video sequences."""
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference."
            "See notebooks/video_predictor_example.ipynb for an example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder for image segmentation tasks."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # Build PromptEncoder and MaskDecoder from SAM (hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = SAM2MaskDecoder(
            num_multimask_outputs=3,
            transformer=SAM2TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """Forward pass through SAM prompt encoders and mask heads.

        This method processes image features and optional point/mask inputs to generate object masks and scores.

        Args:
            backbone_features (torch.Tensor): Image features with shape (B, C, H, W).
            point_inputs (dict[str, torch.Tensor] | None): Dictionary containing point prompts with keys 'point_coords'
                (Tensor of shape (B, P, 2) with float32 dtype, containing absolute pixel-unit coordinates in (x, y)
                format for P input points) and 'point_labels' (Tensor of shape (B, P) with int32 dtype, where 1 means
                positive clicks, 0 means negative clicks, and -1 means padding).
            mask_inputs (torch.Tensor | None): Mask of shape (B, 1, H*16, W*16), float or bool, with the same spatial
                size as the image.
            high_res_features (list[torch.Tensor] | None): List of two feature maps with shapes (B, C, 4*H, 4*W) and (B,
                C, 2*H, 2*W) respectively, used as high-resolution feature maps for SAM decoder.
            multimask_output (bool): If True, output 3 candidate masks and their IoU estimates; if False, output only 1
                mask and its IoU estimate.

        Returns:
            low_res_multimasks (torch.Tensor): Tensor of shape (B, M, H*4, W*4) with SAM output mask logits.
            high_res_multimasks (torch.Tensor): Tensor of shape (B, M, H*16, W*16) with upsampled mask logits.
            ious (torch.Tensor): Tensor of shape (B, M) with estimated IoU for each output mask.
            low_res_masks (torch.Tensor): Tensor of shape (B, 1, H*4, W*4) with the best low-resolution mask.
            high_res_masks (torch.Tensor): Tensor of shape (B, 1, H*16, W*16) with the best high-resolution mask.
            obj_ptr (torch.Tensor): Tensor of shape (B, C) with object pointer vector for the output mask.
            object_score_logits (torch.Tensor): Tensor of shape (B, 1) with object score logits.

        Examples:
            >>> backbone_features = torch.rand(1, 256, 32, 32)
            >>> point_inputs = {"point_coords": torch.rand(1, 2, 2), "point_labels": torch.tensor([[1, 0]])}
            >>> mask_inputs = torch.rand(1, 1, 512, 512)
            >>> results = model._forward_sam_heads(backbone_features, point_inputs, mask_inputs)
            >>> (
            ...     low_res_multimasks,
            ...     high_res_multimasks,
            ...     ious,
            ...     low_res_masks,
            ...     high_res_masks,
            ...     obj_ptr,
            ...     object_score_logits,
            ... ) = results
        """
        B = backbone_features.shape[0]
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.shape[0] == B and sam_point_labels.shape[0] == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device, dtype=backbone_features.dtype)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.to(backbone_features.dtype),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Spatial memory mask is a *hard* choice between obj and no obj, consistent with actual mask prediction
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.to(obj_ptr.dtype)

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr
        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, mask_inputs, backbone_features=None, high_res_features=None):
        """Process mask inputs directly as output, bypassing SAM encoder/decoder."""
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.shape[0], 1).float()
        if not self.use_obj_ptrs_in_encoder or backbone_features is None or high_res_features is None:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(mask_inputs.shape[0], self.hidden_dim, device=mask_inputs.device)
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float.to(backbone_features.dtype)),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Process image batch through encoder to extract multi-level features for SAM model."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def _prepare_backbone_features(self, backbone_out, batch=1):
        """Prepare and flatten visual features from the image backbone output for further processing."""
        if batch > 1:  # expand features if there's more than one prompt
            backbone_out = {
                **backbone_out,
                "backbone_fpn": [feat.expand(batch, -1, -1, -1) for feat in backbone_out["backbone_fpn"]],
                "vision_pos_enc": [pos.expand(batch, -1, -1, -1) for pos in backbone_out["vision_pos_enc"]],
            }
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Prepare memory-conditioned features by fusing current frame's visual features with previous memories."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            return current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frame's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with r>1), in which case
            # we take (self.num_maskmem - 2) frames among every r-th frames plus the last frame.
            r = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
                elif not track_in_reverse:
                    # first find the nearest frame among every r-th frames before this frame
                    # for r=1, this would be (frame_idx - 2)
                    prev_frame_idx = ((frame_idx - 2) // r) * r
                    # then seek further among every r-th frames
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                else:
                    # first find the nearest frame among every r-th frames after this frame
                    # for r=1, this would be (frame_idx + 2)
                    prev_frame_idx = -(-(frame_idx + 2) // r) * r
                    # then seek further among every r-th frames
                    prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to inference device (it's a no-op if it's already on device).
                feats = prev["maskmem_features"].to(device=device, non_blocking=device.type == "cuda")
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device=device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if pos_and_ptrs:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device, dtype=current_vision_feats[-1].dtype)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to transformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # Reshape output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode frame features and masks into a new memory representation for video segmentation."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).to(pix_feat.dtype)
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)  # sigmoid already applied
        maskmem_features = maskmem_out["vision_features"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None, None]) * self.no_obj_embed_spatial[
                ..., None, None
            ].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_out["vision_pos_enc"]

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        """Perform a single tracking step, updating object masks and memory features based on current frame inputs."""
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(mask_inputs, pix_feat, high_res_features)
        else:
            # Fuse visual features with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        return sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        """Run memory encoder on predicted mask to encode it into a new memory feature for future frames."""
        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        """Perform a single tracking step, updating object masks and memory features based on current frame inputs."""
        sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )
        _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = sam_outputs

        current_out = {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptr": obj_ptr,
        }
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Run memory encoder on the predicted mask to encode it into a new memory feature (for use in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Determine whether to use multiple mask outputs in the SAM head based on configuration and inputs."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )

    @staticmethod
    def _apply_non_overlapping_constraints(pred_masks):
        """Apply non-overlapping constraints to masks, keeping the highest scoring object per location."""
        batch_size = pred_masks.shape[0]
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def set_binarize(self, binarize=False):
        """Set binarize for VideoPredictor."""
        self.binarize_mask_from_pts_for_mem_enc = binarize

    def set_imgsz(self, imgsz):
        """Set image size to make model compatible with different image sizes."""
        if hasattr(self.image_encoder, "set_imgsz"):
            self.image_encoder.set_imgsz(imgsz)
        self.image_size = imgsz[0]
        self.sam_prompt_encoder.input_image_size = imgsz
        self.sam_prompt_encoder.image_embedding_size = [
            x // self.backbone_stride for x in imgsz
        ]  # fixed ViT patch size of 16
        self.sam_prompt_encoder.mask_input_size = [
            x // self.backbone_stride * 4 for x in imgsz
        ]  # fixed ViT patch size of 16
        self.sam_image_embedding_size = self.image_size // self.backbone_stride  # update image embedding size


class SAM3Model(SAM2Model):
    """SAM3Model class for Segment Anything Model 3 with memory-based video object segmentation capabilities."""

    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=1008,
        backbone_stride=14,
        sigmoid_scale_for_mem_enc=1,
        sigmoid_bias_for_mem_enc=0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        """SAM3Model class for Segment Anything Model 3 with memory-based video object segmentation capabilities."""
        super().__init__(
            image_encoder,
            memory_attention,
            memory_encoder,
            num_maskmem,
            image_size,
            backbone_stride,
            sigmoid_scale_for_mem_enc,
            sigmoid_bias_for_mem_enc,
            binarize_mask_from_pts_for_mem_enc,
            use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn,
            directly_add_no_mem_embed,
            use_high_res_features_in_sam,
            multimask_output_in_sam,
            multimask_min_pt_num,
            multimask_max_pt_num,
            multimask_output_for_tracking,
            use_multimask_token_for_obj_ptr,
            iou_prediction_use_sigmoid,
            memory_temporal_stride_for_eval,
            non_overlap_masks_for_mem_enc,
            use_obj_ptrs_in_encoder,
            max_obj_ptrs_in_encoder,
            add_tpos_enc_to_obj_ptrs,
            proj_tpos_enc_in_obj_ptrs,
            use_signed_tpos_enc_to_obj_ptrs,
            only_obj_ptrs_in_the_past_for_eval,
            pred_obj_scores,
            pred_obj_scores_mlp,
            fixed_no_obj_ptr,
            soft_no_obj_ptr,
            use_mlp_for_obj_ptr_proj,
            no_obj_embed_spatial,
            sam_mask_decoder_extra_args,
            compile_image_encoder,
        )
        self.sam_mask_decoder = SAM2MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Process image batch through encoder to extract multi-level features for SAM model."""
        backbone_out = self.image_encoder.forward_image_sam2(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def set_imgsz(self, imgsz: tuple[int, int]):
        """Set the image size for the model and mask downsampler."""
        super().set_imgsz(imgsz)
        self.memory_encoder.mask_downsampler.interpol_size = [size // 14 * 16 for size in imgsz]

    @staticmethod
    def _suppress_shrinked_masks(pred_masks, new_pred_masks, shrink_threshold=0.3):
        """Suppress masks that shrink in area after applying pixelwise non-overlapping constraints."""
        area_before = (pred_masks > 0).sum(dim=(-1, -2))
        area_after = (new_pred_masks > 0).sum(dim=(-1, -2))
        area_before = torch.clamp(area_before, min=1.0)
        area_ratio = area_after / area_before
        keep = area_ratio >= shrink_threshold
        keep_mask = keep[..., None, None].expand_as(pred_masks)
        pred_masks_after = torch.where(keep_mask, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks_after

    def _suppress_object_pw_area_shrinkage(self, pred_masks):
        """This function suppresses masks that shrink in area after applying pixelwise non-overlapping constraints. Note
        that the final output can still be overlapping.
        """
        # Apply pixel-wise non-overlapping constraint based on mask scores
        pixel_level_non_overlapping_masks = self._apply_non_overlapping_constraints(pred_masks)
        # Fully suppress masks with high shrinkage (probably noisy) based on the pixel wise non-overlapping constraints
        # NOTE: The output of this function can be a no op if none of the masks shrink by a large factor.
        pred_masks = self._suppress_shrinked_masks(pred_masks, pixel_level_non_overlapping_masks)
        return pred_masks
