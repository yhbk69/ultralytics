# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""模型头部模块。"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "OBB", "Classify", "Detect", "Pose", "RTDETRDecoder", "Segment", "YOLOEDetect", "YOLOESegment", "v10Detect"


class Detect(nn.Module):
    """YOLO 检测头，用于目标检测模型。

    该类实现了 YOLO 模型中用于预测边界框和类别概率的检测头。
    支持训练和推理模式，以及可选的端到端检测能力。

    Attributes:
        dynamic (bool): 强制网格重建。
        export (bool): 导出模式标志。
        format (str): 导出格式。
        end2end (bool): 端到端检测模式。
        max_det (int): 每张图像的最大检测数。
        shape (tuple): 输入形状。
        anchors (torch.Tensor): 锚点。
        strides (torch.Tensor): 特征图步幅。
        legacy (bool): 向后兼容 v3/v5/v8/v9/v11 模型。
        xyxy (bool): 输出格式，xyxy 或 xywh。
        nc (int): 类别数。
        nl (int): 检测层数。
        reg_max (int): DFL 通道数。
        no (int): 每个锚点的输出数。
        stride (torch.Tensor): 构建时计算的步幅。
        cv2 (nn.ModuleList): 边界框回归卷积层。
        cv3 (nn.ModuleList): 分类卷积层。
        dfl (nn.Module): 分布焦点损失层。
        one2one_cv2 (nn.ModuleList): 一对一边界框回归卷积层。
        one2one_cv3 (nn.ModuleList): 一对一分类卷积层。

    Methods:
        forward: 执行前向传播并返回预测结果。
        bias_init: 初始化检测头偏置。
        decode_bboxes: 从预测中解码边界框。
        postprocess: 后处理模型预测。

    Examples:
        创建一个 80 类的检测头
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # 强制网格重建
    export = False  # 导出模式
    format = None  # 导出格式
    max_det = 300  # 最大检测数
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # 初始化
    strides = torch.empty(0)  # 初始化
    legacy = False  # 向后兼容 v3/v5/v8/v9 模型
    xyxy = False  # xyxy 或 xywh 输出

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 YOLO 检测层，指定类别数和通道数。

        Args:
            nc (int): 类别数。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.reg_max = reg_max  # DFL 通道数 (ch[0] // 16 来缩放 n/s/m/l/x 对应 4/8/12/16/20)
        self.no = nc + self.reg_max * 4  # 每个锚点的输出数
        self.stride = torch.zeros(self.nl)  # 构建时计算的步幅
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # 通道数
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """返回一对多头部组件，此处用于 v3/v5/v8/v9/v11 向后兼容。"""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """返回一对一头部组件。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """检查模型是否具有 one2one，用于 v3/v5/v8/v9/v11 向后兼容。"""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """覆盖端到端检测模式。"""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框和类别概率。"""
        if box_head is None or cls_head is None:  # 融合推理时
            return dict()
        bs = x[0].shape[0]  # 批次大小
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """拼接并返回预测的边界框和类别概率。"""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率（基于多层特征图）。

        Args:
            x (dict[str, torch.Tensor]): 检测层的预测字典。

        Returns:
            (torch.Tensor): 解码后的边界框和类别概率拼接张量。
        """
        # 推理路径
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """基于锚点和步幅获取解码后的边界框。"""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """初始化 Detect() 偏置，警告：需要步幅可用。"""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # 从
            a[-1].bias.data[:] = 2.0  # 边界框
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # 分类 (.01 目标, 80 类, 640 图像)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # 从
                a[-1].bias.data[:] = 2.0  # 边界框
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # 分类 (.01 目标, 80 类, 640 图像)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """从预测中解码边界框。"""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """后处理 YOLO 模型预测。

        Args:
            preds (torch.Tensor): 原始预测，形状为 (batch_size, num_anchors, 4 + nc)，最后一维
                格式为 [x1, y1, x2, y2, 类别概率]。

        Returns:
            (torch.Tensor): 处理后的预测，形状为 (batch_size, min(max_det, num_anchors), 6)，最后一维
                格式为 [x1, y1, x2, y2, 最大类别概率, 类别索引]。
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从分数中获取 top-k 索引。

        Args:
            scores (torch.Tensor): 分数张量，形状为 (batch_size, num_anchors, num_classes)。
            max_det (int): 每张图像的最大检测数。

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): top 分数、类别索引和过滤后的索引。
        """
        batch_size, anchors, nc = scores.shape  # 例如 shape(16,8400,84)
        # 导出时直接使用 max_det 以兼容 TensorRT（要求 k 为常量），
        # 否则使用 min(max_det, anchors) 以确保 Python 推理时小输入的安全性
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # 原始索引
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """移除一对多头部以优化推理。"""
        self.cv2 = self.cv3 = None


class Segment(Detect):
    """YOLO 分割头，用于分割模型。

    该类扩展了 Detect 头，增加了掩码预测能力，用于实例分割任务。

    Attributes:
        nm (int): 掩码数。
        npr (int): 原型数。
        proto (Proto): 原型生成模块。
        cv4 (nn.ModuleList): 掩码系数卷积层。

    Methods:
        forward: 返回模型输出和掩码系数。

    Examples:
        创建一个分割头
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 YOLO 模型属性，如掩码数、原型数和卷积层。

        Args:
            nc (int): 类别数。
            nm (int): 掩码数。
            npr (int): 原型数。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # 掩码数
        self.npr = npr  # 原型数
        self.proto = Proto(ch[0], self.npr, self.nm)  # 原型

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """返回一对多头部组件，此处用于向后兼容。"""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """返回一对一头部组件。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """训练时返回模型输出和掩码系数，否则返回输出和掩码系数。"""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # 掩码原型
        if isinstance(preds, dict):  # 训练及训练期间验证时
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率，拼接掩码系数。"""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和掩码系数。"""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # 批次大小
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """后处理 YOLO 模型预测。

        Args:
            preds (torch.Tensor): 原始预测，形状为 (batch_size, num_anchors, 4 + nc + nm)，最后一维
                格式为 [x1, y1, x2, y2, 类别概率, 掩码系数]。

        Returns:
            (torch.Tensor): 处理后的预测，形状为 (batch_size, min(max_det, num_anchors), 6 + nm)，最后一维
                格式为 [x1, y1, x2, y2, 最大类别概率, 类别索引, 掩码系数]。
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """移除一对多头部以优化推理。"""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """YOLO26 分割头，用于分割模型。

    该类扩展了 Segment 头，使用 Proto26 进行实例分割任务中的掩码预测。

    Attributes:
        nm (int): 掩码数。
        npr (int): 原型数。
        proto (Proto26): 原型生成模块。
        cv4 (nn.ModuleList): 掩码系数卷积层。

    Methods:
        forward: 返回模型输出和掩码系数。

    Examples:
        创建一个分割头
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 YOLO 模型属性，如掩码数、原型数和卷积层。

        Args:
            nc (int): 类别数。
            nm (int): 掩码数。
            npr (int): 原型数。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # 原型

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """训练时返回模型输出和掩码系数，否则返回输出和掩码系数。"""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # 掩码原型
        if isinstance(preds, dict):  # 训练及训练期间验证时
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """移除一对多头部和原型模块的额外部分以优化推理。"""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO 旋转目标检测头，用于带旋转的检测模型。

    该类扩展了 Detect 头，增加了带旋转角度的旋转边界框预测能力。

    Attributes:
        ne (int): 额外参数数。
        cv4 (nn.ModuleList): 角度预测卷积层。
        angle (torch.Tensor): 预测的旋转角度。

    Methods:
        forward: 拼接并返回预测的边界框和类别概率。
        decode_bboxes: 解码旋转边界框。

    Examples:
        创建一个 OBB 检测头
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 OBB，指定类别数 nc 和层通道数 ch。

        Args:
            nc (int): 类别数。
            ne (int): 额外参数数。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # 额外参数数

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """返回一对多头部组件，此处用于向后兼容。"""
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        """返回一对一头部组件。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率，拼接旋转角度。"""
        # 为 decode_bboxes 方便使用
        self.angle = x["angle"]
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和角度。"""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # 批次大小
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """解码旋转边界框。"""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """后处理 YOLO 模型预测。

        Args:
            preds (torch.Tensor): 原始预测，形状为 (batch_size, num_anchors, 4 + nc + ne)，最后一维
                格式为 [x, y, w, h, 类别概率, 角度]。

        Returns:
            (torch.Tensor): 处理后的预测，形状为 (batch_size, min(max_det, num_anchors), 7)，最后一维
                格式为 [x, y, w, h, 最大类别概率, 类别索引, 角度]。
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        """移除一对多头部以优化推理。"""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """YOLO26 旋转目标检测头，用于带旋转的检测模型。该类扩展了 OBB 头，修改了角度处理方式，
    输出原始角度预测（不经过 sigmoid 变换），与原始 OBB 类不同。

    Attributes:
        ne (int): 额外参数数。
        cv4 (nn.ModuleList): 角度预测卷积层。
        angle (torch.Tensor): 预测的旋转角度。

    Methods:
        forward_head: 拼接并返回预测的边界框、类别概率和原始角度。

    Examples:
        创建一个 OBB26 检测头
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x)
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和原始角度。"""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # 批次大小
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits（原始输出，不经过 sigmoid 变换）
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO 关键点检测头，用于姿态估计模型。

    该类扩展了 Detect 头，增加了关键点预测能力，用于姿态估计任务。

    Attributes:
        kpt_shape (tuple): 关键点数和维度数（2 表示 x,y 或 3 表示 x,y,可见性）。
        nk (int): 关键点总数值数。
        cv4 (nn.ModuleList): 关键点预测卷积层。

    Methods:
        forward: 执行 YOLO 模型的前向传播并返回预测结果。
        kpts_decode: 从预测中解码关键点。

    Examples:
        创建一个姿态检测头
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 YOLO 网络，使用默认参数和卷积层。

        Args:
            nc (int): 类别数。
            kpt_shape (tuple): 关键点数和维度数（2 表示 x,y 或 3 表示 x,y,可见性）。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # 关键点数和维度数（2 表示 x,y 或 3 表示 x,y,可见性）
        self.nk = kpt_shape[0] * kpt_shape[1]  # 关键点总数值

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """返回一对多头部组件，此处用于向后兼容。"""
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        """返回一对一头部组件。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率，拼接关键点。"""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和关键点。"""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # 批次大小
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """后处理 YOLO 模型预测。

        Args:
            preds (torch.Tensor): 原始预测，形状为 (batch_size, num_anchors, 4 + nc + nk)，最后一维
                格式为 [x1, y1, x2, y2, 类别概率, 关键点]。

        Returns:
            (torch.Tensor): 处理后的预测，形状为 (batch_size, min(max_det, num_anchors), 6 + self.nk)，
                最后一维格式为 [x1, y1, x2, y2, 最大类别概率, 类别索引, 关键点]。
        """
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """移除一对多头部以优化推理。"""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """从预测中解码关键点。"""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS 缺陷 https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """YOLO26 关键点检测头，用于姿态估计模型。

    该类扩展了 Pose 头，使用归一化流进行关键点预测，用于姿态估计任务。

    Attributes:
        kpt_shape (tuple): 关键点数和维度数（2 表示 x,y 或 3 表示 x,y,可见性）。
        nk (int): 关键点总数值数。
        cv4 (nn.ModuleList): 关键点预测卷积层。

    Methods:
        forward: 执行 YOLO 模型的前向传播并返回预测结果。
        kpts_decode: 从预测中解码关键点。

    Examples:
        创建一个姿态检测头
        >>> pose = Pose26(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """初始化 YOLO 网络，使用默认参数和卷积层。

        Args:
            nc (int): 类别数。
            kpt_shape (tuple): 关键点数和维度数（2 表示 x,y 或 3 表示 x,y,可见性）。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # 每个关键点的 sigma_x, sigma_y
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """返回一对多头部组件，此处用于向后兼容。"""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        """返回一对一头部组件。"""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和关键点。"""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # 批次大小
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """移除一对多头部以优化推理。"""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """从预测中解码关键点。"""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN 修复
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS 缺陷 https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    """YOLO 分类头，即 x(b,c1,20,20) -> x(b,c2)。

    该类实现了将特征图转换为类别预测的分类头。

    Attributes:
        export (bool): 导出模式标志。
        conv (Conv): 用于特征变换的卷积层。
        pool (nn.AdaptiveAvgPool2d): 全局平均池化层。
        drop (nn.Dropout): 正则化用的 Dropout 层。
        linear (nn.Linear): 最终分类的线性层。

    Methods:
        forward: 对输入特征图执行前向传播。

    Examples:
        创建一个分类头
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # 导出模式

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """初始化 YOLO 分类头，将输入张量从 (b,c1,20,20) 变换为 (b,c2) 形状。

        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出类别数。
            k (int): 卷积核大小。
            s (int): 步幅。
            p (int, optional): 填充。
            g (int): 分组数。
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 大小
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 变为 x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # 变为 x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """对输入特征图执行前向传播。"""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # 获取最终输出
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """用于将 YOLO 检测模型与文本嵌入语义理解相结合的检测头。

    该类扩展了标准 Detect 头，整合文本嵌入以增强目标检测任务中的语义理解。

    Attributes:
        cv3 (nn.ModuleList): 嵌入特征卷积层。
        cv4 (nn.ModuleList): 文本-视觉对齐的对比头层。

    Methods:
        forward: 拼接并返回预测的边界框和类别概率。
        bias_init: 初始化检测头偏置。

    Examples:
        创建一个 WorldDetect 头
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """初始化 YOLO 检测层，指定类别数 nc 和层通道数 ch。

        Args:
            nc (int): 类别数。
            embed (int): 嵌入维度。
            with_bn (bool): 是否在对比头中使用批归一化。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """拼接并返回预测的边界框和类别概率。"""
        feats = [xi.clone() for xi in x]  # 保存原始特征用于锚点生成
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # 使用不同文本推理时 nc 可能会变化
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """初始化 Detect() 偏置，警告：需要步幅可用。"""
        m = self  # self.model[-1]  # Detect() 模块
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # 名义类别频率
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # 从
            a[-1].bias.data[:] = 1.0  # 边界框
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # 分类 (.01 目标, 80 类, 640 图像)


class LRPCHead(nn.Module):
    """轻量级区域建议与分类头，用于高效目标检测。

    该头结合区域建议过滤和分类，支持动态词汇表的高效检测。

    Attributes:
        vocab (nn.Module): 词汇/分类层。
        pf (nn.Module): 建议过滤模块。
        loc (nn.Module): 定位模块。
        enabled (bool): 头是否启用。

    Methods:
        conv2linear: 将 1x1 卷积层转换为线性层。
        forward: 处理分类和定位特征以生成检测建议。

    Examples:
        创建一个 LRPC 头
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """初始化 LRPCHead，包含词汇、建议过滤和定位组件。

        Args:
            vocab (nn.Module): 词汇/分类模块。
            pf (nn.Module): 建议过滤模块。
            loc (nn.Module): 定位模块。
            enabled (bool): 是否启用头部功能。
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """将 1x1 卷积层转换为线性层。"""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """处理分类和定位特征以生成检测建议。"""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )


class YOLOEDetect(Detect):
    """用于将 YOLO 检测模型与文本嵌入语义理解相结合的检测头。

    该类扩展了标准 Detect 头，支持通过文本嵌入和视觉提示嵌入进行文本引导检测，
    增强语义理解。

    Attributes:
        is_fused (bool): 模型是否已融合用于推理。
        cv3 (nn.ModuleList): 嵌入特征卷积层。
        cv4 (nn.ModuleList): 文本-视觉对齐的对比头层。
        reprta (Residual): 文本提示嵌入的残差块。
        savpe (SAVPE): 空间感知视觉提示嵌入模块。
        embed (int): 嵌入维度。

    Methods:
        fuse: 将文本特征与模型权重融合以实现高效推理。
        get_tpe: 获取带归一化的文本提示嵌入。
        get_vpe: 获取带空间感知的视觉提示嵌入。
        forward_lrpc: 使用融合的文本嵌入处理特征，用于无提示模型。
        forward: 使用类别提示嵌入处理特征以生成检测结果。
        bias_init: 初始化检测头的偏置。

    Examples:
        创建一个 YOLOEDetect 头
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """初始化 YOLO 检测层，指定类别数 nc 和层通道数 ch。

        Args:
            nc (int): 类别数。
            embed (int): 嵌入维度。
            with_bn (bool): 是否在对比头中使用批归一化。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # 用新的 cv3 覆盖
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """将文本特征与模型权重融合以实现高效推理。"""
        if txt_feats is None:  # 表示移除一对多分支
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        if self.cv3 and self.cv4:
            self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """将文本提示嵌入与模型权重融合以实现高效推理。"""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """获取带归一化的文本提示嵌入。"""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """获取带空间感知的视觉提示嵌入。"""
        if vpe.shape[1] == 0:  # 无视觉提示嵌入
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """使用类别提示嵌入处理特征以生成检测结果。"""
        if hasattr(self, "lrpc"):  # 用于无提示推理
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """使用融合的文本嵌入处理特征，为无提示模型生成检测结果。"""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """解码预测的边界框用于推理。"""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """返回一对多头部组件，此处用于 v3/v5/v8/v9/v11 向后兼容。"""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """返回一对一头部组件。"""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """拼接并返回预测的边界框、类别概率和对比分数。"""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # 融合推理时
            return dict()
        bs = x[0].shape[0]  # 批次大小
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # 使用不同文本推理时 nc 可能会变化
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """初始化 Detect() 偏置，警告：需要步幅可用。"""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # 边界框
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # 边界框
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """带文本嵌入能力的 YOLO 分割头。

    该类扩展了 YOLOEDetect，增加了掩码预测能力，用于带文本引导语义理解的实例分割任务。

    Attributes:
        nm (int): 掩码数。
        npr (int): 原型数。
        proto (Proto): 原型生成模块。
        cv5 (nn.ModuleList): 掩码系数卷积层。

    Methods:
        forward: 返回模型输出和掩码系数。

    Examples:
        创建一个 YOLOESegment 头
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """初始化 YOLOESegment，指定类别数、掩码参数和嵌入维度。

        Args:
            nc (int): 类别数。
            nm (int): 掩码数。
            npr (int): 原型数。
            embed (int): 嵌入维度。
            with_bn (bool): 是否在对比头中使用批归一化。
            reg_max (int): DFL 最大通道数。
            end2end (bool): 是否使用端到端无 NMS 检测。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """返回一对多头部组件，此处用于 v3/v5/v8/v9/v11 向后兼容。"""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """返回一对一头部组件。"""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """使用融合的文本嵌入处理特征，为无提示模型生成检测结果。"""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """训练时返回模型输出和掩码系数，否则返回输出和掩码系数。"""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # 掩码原型
        if isinstance(preds, dict):  # 训练及训练期间验证时
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """解码预测的边界框和类别概率，拼接掩码系数。"""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """拼接并返回预测的边界框、类别概率和掩码系数。"""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # 批次大小
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """后处理 YOLO 模型预测。

        Args:
            preds (torch.Tensor): 原始预测，形状为 (batch_size, num_anchors, 4 + nc + nm)，最后一维
                格式为 [x1, y1, x2, y2, 类别概率, 掩码系数]。

        Returns:
            (torch.Tensor): 处理后的预测，形状为 (batch_size, min(max_det, num_anchors), 6 + nm)，最后一维
                格式为 [x1, y1, x2, y2, 最大类别概率, 类别索引, 掩码系数]。
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """将文本特征与模型权重融合以实现高效推理。"""
        super().fuse(txt_feats)
        if txt_feats is None:  # 表示移除一对多分支
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """使用 Proto26 生成掩码的 YOLOE 风格分割头模块。

    该类扩展了 YOLOESegment 的功能，通过集成 Proto26 生成模块和卷积层来预测掩码系数，
    实现分割能力。

    Args:
        nc (int): 类别数。默认为 80。
        nm (int): 掩码数。默认为 32。
        npr (int): 原型通道数。默认为 256。
        embed (int): 嵌入维度。默认为 512。
        with_bn (bool): 是否使用批归一化。默认为 False。
        reg_max (int): DFL 最大通道数。默认为 16。
        end2end (bool): 是否使用端到端检测模式。默认为 False。
        ch (tuple[int, ...]): 每个尺度的输入通道数。

    Attributes:
        nm (int): 分割掩码数。
        npr (int): 原型通道数。
        proto (Proto26): 分割用原型生成模块。
        cv5 (nn.ModuleList): 从特征生成掩码系数的卷积层。
        one2one_cv5 (nn.ModuleList, optional): cv5 的深拷贝，用于端到端检测分支。
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """初始化 YOLOESegment26，指定类别数、掩码参数和嵌入维度。"""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # 原型

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """训练时返回模型输出和掩码系数，否则返回输出和掩码系数。"""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semseg=False)  # 掩码原型

        if isinstance(preds, dict):  # 训练及训练期间验证时
            if self.end2end and not hasattr(self, "lrpc"):  # 非无提示模式
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """实时可变形 Transformer 解码器（RTDETRDecoder）模块，用于目标检测。

    该解码器模块利用 Transformer 架构和可变形卷积来预测图像中目标的边界框和类别标签。
    它整合了多层特征，通过一系列 Transformer 解码器层输出最终预测。

    Attributes:
        export (bool): 导出模式标志。
        hidden_dim (int): 隐藏层维度。
        nhead (int): 多头注意力中的头数。
        nl (int): 特征层数。
        nc (int): 类别数。
        num_queries (int): 查询点数。
        num_decoder_layers (int): 解码器层数。
        input_proj (nn.ModuleList): 骨干特征的输入投影层。
        decoder (DeformableTransformerDecoder): Transformer 解码器模块。
        denoising_class_embed (nn.Embedding): 去噪的类别嵌入。
        num_denoising (int): 去噪查询数。
        label_noise_ratio (float): 训练时的标签噪声比。
        box_noise_scale (float): 训练时的边界框噪声缩放。
        learnt_init_query (bool): 是否学习初始查询嵌入。
        tgt_embed (nn.Embedding): 查询的目标嵌入。
        query_pos_head (MLP): 查询位置头。
        enc_output (nn.Sequential): 编码器输出层。
        enc_score_head (nn.Linear): 编码器分数预测头。
        enc_bbox_head (MLP): 编码器边界框预测头。
        dec_score_head (nn.ModuleList): 解码器分数预测头。
        dec_bbox_head (nn.ModuleList): 解码器边界框预测头。

    Methods:
        forward: 执行前向传播并返回边界框和分类分数。

    Examples:
        创建一个 RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # 导出模式
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # 隐藏维度
        nq: int = 300,  # 查询数
        ndp: int = 4,  # 解码器点数
        nh: int = 8,  # 注意力头数
        ndl: int = 6,  # 解码器层数
        d_ffn: int = 1024,  # 前馈网络维度
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # 训练参数
        nd: int = 100,  # 去噪数
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """初始化 RTDETRDecoder 模块，使用给定参数。

        Args:
            nc (int): 类别数。
            ch (tuple): 骨干网络特征图的通道数。
            hd (int): 隐藏层维度。
            nq (int): 查询点数。
            ndp (int): 解码器点数。
            nh (int): 多头注意力中的头数。
            ndl (int): 解码器层数。
            d_ffn (int): 前馈网络维度。
            dropout (float): Dropout 率。
            act (nn.Module): 激活函数。
            eval_idx (int): 评估索引。
            nd (int): 去噪数。
            label_noise_ratio (float): 标签噪声比。
            box_noise_scale (float): 边界框噪声缩放。
            learnt_init_query (bool): 是否学习初始查询嵌入。
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # 特征层数
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # 骨干特征投影
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # 注意：简化版本，但与 .pt 权重不一致。
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer 模块
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # 去噪部分
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # 解码器嵌入
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # 编码器头
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # 解码器头
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """执行模块的前向传播，返回输入的边界框和分类分数。

        Args:
            x (list[torch.Tensor]): 骨干网络的特征图列表。
            batch (dict, optional): 训练的批次信息。

        Returns:
            outputs (tuple | torch.Tensor): 训练时返回边界框、分数和其他元数据的元组。
                推理时返回形状为 (bs, num_queries, 6) 的张量，包含边界框、置信度和类别标签。
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # 输入投影和嵌入
        feats, shapes = self._get_encoder_input(x)

        # 准备去噪训练
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # 解码器
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        if self.training and dn_meta is None:
            # 触碰 denoising_class_embed 以使 DDP 在批次无 GT 时也认为其被使用
            dec_bboxes = dec_bboxes + 0 * self.denoising_class_embed.weight.sum()
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, num_queries, 4), (bs, num_queries, nc)
        y = self.postprocess(dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid())
        return y if self.export else (y, x)

    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """后处理预测，选择 top-k 检测。

        Args:
            boxes (torch.Tensor): 预测的边界框，形状为 (batch_size, num_queries, 4)，xywh 格式。
            scores (torch.Tensor): 类别分数，形状为 (batch_size, num_queries, nc)。

        Returns:
            (torch.Tensor): 处理后的预测，形状为 (batch_size, num_queries, 6)，最后一维格式为
                [cx, cy, w, h, 最大类别概率, 类别索引]。
        """
        scores, index = scores.flatten(1).topk(self.num_queries)
        query_idx = index // self.nc
        boxes = boxes.gather(dim=1, index=query_idx.unsqueeze(-1).expand(-1, -1, 4))
        return torch.cat([boxes, scores[..., None], (index % self.nc)[..., None].float()], dim=-1)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """为给定形状生成锚点边界框，使用指定网格大小并验证。

        Args:
            shapes (list): 特征图形状列表。
            grid_size (float, optional): 网格单元的基础大小。
            dtype (torch.dtype, optional): 张量的数据类型。
            device (str, optional): 创建张量的设备。
            eps (float, optional): 数值稳定性用的小值。

        Returns:
            anchors (torch.Tensor): 生成的锚点框。
            valid_mask (torch.Tensor): 锚点的有效掩码。
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """处理并返回编码器输入，通过获取输入的投影特征并拼接。

        Args:
            x (list[torch.Tensor]): 骨干网络的特征图列表。

        Returns:
            feats (torch.Tensor): 处理后的特征。
            shapes (list): 特征图形状列表。
        """
        # 获取投影特征
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # 获取编码器输入
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从提供的特征和形状生成并准备解码器所需的输入。

        Args:
            feats (torch.Tensor): 编码器处理后的特征。
            shapes (list): 特征图形状列表。
            dn_embed (torch.Tensor, optional): 去噪嵌入。
            dn_bbox (torch.Tensor, optional): 去噪边界框。

        Returns:
            embeddings (torch.Tensor): 解码器的查询嵌入。
            refer_bbox (torch.Tensor): 参考边界框。
            enc_bboxes (torch.Tensor): 编码的边界框。
            enc_scores (torch.Tensor): 编码的分数。
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # 准备解码器输入
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # 查询选择
        # (bs*num_queries,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # 动态锚点 + 静态内容
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """初始化或重置模型各组件的参数，使用预定义的权重和偏置。"""
        # 类别和边界框头初始化
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # 注意：`linear_init` 中的权重初始化在使用自定义数据集训练时会导致 NaN。
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """v10 检测头，来自 https://arxiv.org/pdf/2405.14458。

    该类实现了 YOLOv10 检测头，采用双分配训练和一致的双重预测，
    以提高效率和性能。

    Attributes:
        end2end (bool): 端到端检测模式。
        max_det (int): 最大检测数。
        cv3 (nn.ModuleList): 轻量分类头层。
        one2one_cv3 (nn.ModuleList): 一对一分类头层。

    Methods:
        __init__: 初始化 v10Detect 对象，指定类别数和输入通道。
        forward: 执行 v10Detect 模块的前向传播。
        bias_init: 初始化 Detect 模块的偏置。
        fuse: 移除一对多头部以优化推理。

    Examples:
        创建一个 v10Detect 头
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """初始化 v10Detect 对象，指定类别数和输入通道。

        Args:
            nc (int): 类别数。
            ch (tuple): 骨干网络特征图的通道数元组。
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))  # 通道数
        # 轻量分类头
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """移除一对多头部以优化推理。"""
        self.cv2 = self.cv3 = None
