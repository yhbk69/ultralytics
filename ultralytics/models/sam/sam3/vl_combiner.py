# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

# 版权所有 (c) Meta Platforms, Inc. 及其附属公司。保留所有权利。

"""提供将视觉骨干与语言骨干组合的工具。"""

from __future__ import annotations

from copy import copy

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .necks import Sam3DualViTDetNeck


class SAM3VLBackbone(nn.Module):
    """此骨干将视觉骨干和语言骨干组合在一起，不进行融合。因此它更像是一个
    便利的包装器来同时处理两个骨干。

    它添加了对激活检查点和编译的支持。
    """

    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp=0,
    ):
        """初始化骨干组合器。

        :param visual: 要使用的视觉骨干
        :param text: 要使用的文本编码器
        """
        super().__init__()
        self.vision_backbone: Sam3DualViTDetNeck = torch.compile(visual) if compile_visual else visual
        self.language_backbone = text
        self.scalp = scalp
        # 允许在整个视觉和语言骨干上运行激活检查点
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone

    def forward(
        self,
        samples: torch.Tensor,
        captions: list[str],
        input_boxes: torch.Tensor = None,
        additional_text: list[str] | None = None,
    ):
        """骨干组合器的前向传播。

        :param samples: 输入图像
        :param captions: 输入标题
        :param input_boxes: 如果文本中包含框的占位符，此参数包含其空间特征张量
        :param additional_text: 可用于在同一次前向传播中编码额外的文本（不同于标题）
        :return: 包含以下键的输出字典：
            - vision_features: 视觉骨干的输出
            - language_features: 语言骨干的输出
            - language_mask: 语言骨干的注意力掩码
            - vision_pos_enc: 视觉骨干的位置编码
            - (可选) additional_text_features: 语言骨干对额外文本的输出
            - (可选) additional_text_mask: 语言骨干对额外文本的注意力掩码
        """
        output = self.forward_image(samples)
        output.update(self.forward_text(captions, input_boxes, additional_text))
        return output

    def forward_image(self, samples: torch.Tensor):
        """视觉骨干的前向传播，同时获取 SAM3 和 SAM2 特征。"""
        # 通过骨干网络前向传播
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(samples)
        if self.scalp > 0:
            # 丢弃最低分辨率特征
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None

        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        return {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

    def forward_image_sam2(self, samples: torch.Tensor):
        """视觉骨干的前向传播，仅获取 SAM2 特征。"""
        xs = self.vision_backbone.trunk(samples)
        x = xs[-1]  # simpleFPN

        assert self.vision_backbone.sam2_convs is not None, "SAM2 neck is not available."
        sam2_features, sam2_pos = self.vision_backbone.sam_forward_feature_levels(x, self.vision_backbone.sam2_convs)

        if self.scalp > 0:
            # 丢弃最低分辨率特征
            sam2_features, sam2_pos = (
                sam2_features[: -self.scalp],
                sam2_pos[: -self.scalp],
            )

        return {
            "vision_features": sam2_features[-1],
            "vision_pos_enc": sam2_pos,
            "backbone_fpn": sam2_features,
        }

    def forward_text(self, captions, input_boxes=None, additional_text=None):
        """文本编码器的前向传播。"""
        output = {}

        # 通过 text_encoder 前向传播
        text_to_encode = copy(captions)
        if additional_text is not None:
            # 如果有 additional_text，我们将它们搭载到此次前向传播中。
            # 它们稍后将用于输出对齐
            text_to_encode += additional_text

        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            text_attention_mask, text_memory, text_embeds = self.language_backbone(text_to_encode, input_boxes)

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[-len(additional_text) :]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds  # 前向传播到编码器之前的文本嵌入

        return output

    def set_imgsz(self, imgsz: list[int] = [1008, 1008]):
        """设置视觉骨干的图像尺寸。"""
        self.vision_backbone.set_imgsz(imgsz)
