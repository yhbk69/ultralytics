# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

# 版权所有 (c) Meta Platforms, Inc. 及其附属公司。保留所有权利。

"""Neck 是视觉骨干网络与检测模型其余部分之间的接口。"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class Sam3DualViTDetNeck(nn.Module):
    """实现类似 ViTDet 中简单 FPN 的 Neck，支持双 Neck（用于 SAM3 和 SAM2）。"""

    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        """
        类似 ViTDet 的 SimpleFPN Neck
        （来自 detectron2，做了轻微适配）
        支持双 Neck 设置，即拥有两个相同的 Neck（分别用于 SAM3 和 SAM2），但权重不同。

        :param trunk: 骨干网络
        :param position_encoding: 使用的位置编码
        :param d_model: 模型维度
        :param scale_factors: 每个 FPN 级别的缩放因子元组
        :param add_sam2_neck: 是否添加第二个 Neck 用于 SAM2
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # 假设 sam2 Neck 是原始 Neck 的克隆
            self.sam2_convs = deepcopy(self.convs)

    def forward(
        self, tensor_list: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """获取特征图和位置编码。"""
        xs = self.trunk(tensor_list)
        x = xs[-1]  # simpleFPN
        sam3_out, sam3_pos = self.sam_forward_feature_levels(x, self.convs)
        if self.sam2_convs is None:
            return sam3_out, sam3_pos, None, None
        sam2_out, sam2_pos = self.sam_forward_feature_levels(x, self.sam2_convs)
        return sam3_out, sam3_pos, sam2_out, sam2_pos

    def sam_forward_feature_levels(
        self, x: torch.Tensor, convs: nn.ModuleList
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """运行 Neck 卷积并计算每个特征级别的位置编码。"""
        outs, poss = [], []
        for conv in convs:
            feat = conv(x)
            outs.append(feat)
            poss.append(self.position_encoding(feat).to(feat.dtype))
        return outs, poss

    def set_imgsz(self, imgsz: list[int] = [1008, 1008]):
        """设置骨干网络 trunk 的图像尺寸。"""
        self.trunk.set_imgsz(imgsz)
