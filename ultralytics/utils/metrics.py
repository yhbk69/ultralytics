# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""模型验证指标。"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, TryExcept, checks, plt_settings

OKS_SIGMA = (
    np.array(
        [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
        dtype=np.float32,
    )
    / 10.0
)
RLE_WEIGHT = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5])


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """给定 box1 和 box2，计算交集占 box2 面积的比例。

    Args:
        box1 (np.ndarray): 形状 (N, 4) 的 numpy 数组，表示 x1y1x2y2 格式的 N 个边界框。
        box2 (np.ndarray): 形状 (M, 4) 的 numpy 数组，表示 x1y1x2y2 格式的 M 个边界框。
        iou (bool, optional): 若为 True 则计算标准 IoU，否则返回 inter_area/box2_area。
        eps (float, optional): 避免除零的小值。

    Returns:
        (np.ndarray): 形状 (N, M) 的 numpy 数组，表示交集占 box2 面积的比例。
    """
    # 获取边界框坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # 交集面积
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 面积
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # 交集占 box2 面积比
    return inter_area / (area + eps)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """计算边界框的交并比（IoU）。

    Args:
        box1 (torch.Tensor): 形状 (N, 4) 的张量，表示 (x1, y1, x2, y2) 格式的 N 个边界框。
        box2 (torch.Tensor): 形状 (M, 4) 的张量，表示 (x1, y1, x2, y2) 格式的 M 个边界框。
        eps (float, optional): 避免除零的小值。

    Returns:
        (torch.Tensor): NxM 张量，包含 box1 和 box2 中每对元素的 IoU 值。

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # 注意：需要 .float() 以获得准确的 IoU 值
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = 交集 / (面积1 + 面积2 - 交集)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """计算边界框之间的交并比（IoU）。

    此函数支持 `box1` 和 `box2` 的各种形状，只要最后一个维度为 4。例如，
    可以传入形状为 (4,)、(N, 4)、(B, N, 4) 或 (B, N, 1, 4) 的张量。内部代码会在
    `xywh=True` 时将最后一个维度拆分为 (x, y, w, h)，在 `xywh=False` 时拆分为 (x1, y1, x2, y2)。

    Args:
        box1 (torch.Tensor): 表示一个或多个边界框的张量，最后一个维度为 4。
        box2 (torch.Tensor): 表示一个或多个边界框的张量，最后一个维度为 4。
        xywh (bool, optional): 若为 True，输入框为 (x, y, w, h) 格式；若为 False，为 (x1, y1, x2, y2) 格式。
        GIoU (bool, optional): 若为 True，计算广义 IoU。
        DIoU (bool, optional): 若为 True，计算距离 IoU。
        CIoU (bool, optional): 若为 True，计算完全 IoU。
        eps (float, optional): 避免除零的小值。

    Returns:
        (torch.Tensor): 根据指定标志返回 IoU、GIoU、DIoU 或 CIoU 值。
    """
    # 获取边界框坐标
    if xywh:  # 从 xywh 转换为 xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 交集面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # 并集面积
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # 凸包（最小外接矩形）宽度
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 凸包高度
        if CIoU or DIoU:  # 距离或完全 IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # 凸包对角线平方
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # 中心距离**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # 凸包面积
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """计算掩码 IoU。

    Args:
        mask1 (torch.Tensor): 形状 (N, n) 的张量，N 为真值目标数量，n 为图像宽高之积。
        mask2 (torch.Tensor): 形状 (M, n) 的张量，M 为预测目标数量，n 为图像宽高之积。
        eps (float, optional): 避免除零的小值。

    Returns:
        (torch.Tensor): 形状 (N, M) 的张量，表示掩码 IoU。
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (面积1 + 面积2) - 交集
    return intersection / (union + eps)


def kpt_iou(
    kpt1: torch.Tensor, kpt2: torch.Tensor, area: torch.Tensor, sigma: list[float], eps: float = 1e-7
) -> torch.Tensor:
    """计算目标关键点相似度（OKS）。

    Args:
        kpt1 (torch.Tensor): 形状 (N, 17, 3) 的张量，表示真值关键点。
        kpt2 (torch.Tensor): 形状 (M, 17, 3) 的张量，表示预测关键点。
        area (torch.Tensor): 形状 (N,) 的张量，表示真值面积。
        sigma (list[float]): 包含 17 个值的列表，表示关键点尺度。
        eps (float, optional): 避免除零的小值。

    Returns:
        (torch.Tensor): 形状 (N, M) 的张量，表示关键点相似度。
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # 来自 cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # 来自公式
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从定向边界框生成协方差矩阵。

    Args:
        boxes (torch.Tensor): 形状 (N, 5) 的张量，表示 xywhr 格式的旋转边界框。

    Returns:
        (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 协方差矩阵分量 (a, b, c)，协方差矩阵为 [[a, c], [c, b]]，各形状 (N, 1)。
    """
    # 高斯边界框，忽略中心点（前两列），因为此处不需要。
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """计算定向边界框之间的概率 IoU。

    Args:
        obb1 (torch.Tensor): 真值 OBB，形状 (N, 5)，xywhr 格式。
        obb2 (torch.Tensor): 预测 OBB，形状 (N, 5)，xywhr 格式。
        CIoU (bool, optional): 若为 True，计算 CIoU。
        eps (float, optional): 避免除零的小值。

    Returns:
        (torch.Tensor): OBB 相似度，形状 (N,)。

    Notes:
        OBB 格式: [center_x, center_y, width, height, rotation_angle]。

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # 仅包含宽高比部分
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """计算定向边界框之间的概率 IoU。

    Args:
        obb1 (torch.Tensor | np.ndarray): 形状 (N, 5) 的张量，表示 xywhr 格式的真值 OBB。
        obb2 (torch.Tensor | np.ndarray): 形状 (M, 5) 的张量，表示 xywhr 格式的预测 OBB。
        eps (float, optional): 避免除零的小值。

    Returns:
        (torch.Tensor): 形状 (N, M) 的张量，表示 OBB 相似度。

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    """计算平滑的正负二元交叉熵目标。

    Args:
        eps (float, optional): 标签平滑的 epsilon 值。

    Returns:
        pos (float): 正标签平滑 BCE 目标。
        neg (float): 负标签平滑 BCE 目标。

    References:
        https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix(DataExportMixin):
    """用于计算和更新目标检测与分类任务混淆矩阵的类。

    属性:
        task (str): 任务类型，'detect' 或 'classify'。
        matrix (np.ndarray): 混淆矩阵，维度取决于任务。
        nc (int): 类别数量。
        names (dict[int, str]): 类别名称，用作图上的标签。
        matches (dict | None): 包含按 TP、FP 和 FN 分类的真值和预测索引。
    """

    def __init__(self, names: dict[int, str] = {}, task: str = "detect", save_matches: bool = False):
        """初始化 ConfusionMatrix 实例。

        Args:
            names (dict[int, str], optional): 类别名称，用作图上的标签。
            task (str, optional): 任务类型，'detect' 或 'classify'。
            save_matches (bool, optional): 保存 GT、TP、FP、FN 的索引用于可视化。
        """
        self.task = task
        self.nc = len(names)  # 类别数量
        self.matrix = np.zeros((self.nc, self.nc)) if self.task == "classify" else np.zeros((self.nc + 1, self.nc + 1))
        self.names = names  # 类别名称
        self.matches = {} if save_matches else None

    def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None:
        """将匹配结果追加到最后一个批次的 TP、FP、FN 或 GT 列表中。

        此方法通过将特定批次数据追加到相应的匹配类型（真正例、假正例或假负例）来更新匹配字典。

        Args:
            mtype (str): 匹配类型标识符（'TP'、'FP'、'FN' 或 'GT'）。
            batch (dict[str, Any]): 包含检测结果数据的批次，键如 'bboxes'、'cls'、'conf'、'keypoints'、'masks'。
            idx (int): 要从批次中追加的特定检测的索引。

        Notes:
            对于掩码，同时处理重叠和非重叠情况。当 masks.max() > 1.0 时，表示 overlap_mask=True，形状 (1, H, W)，否则使用直接索引。
        """
        if self.matches is None:
            return
        for k, v in batch.items():
            if k in {"bboxes", "cls", "conf", "keypoints"}:
                self.matches[mtype][k] += v[[idx]]
            elif k == "masks":
                # 注意：masks.max() > 1.0 表示 overlap_mask=True，形状 (1, H, W)
                self.matches[mtype][k] += [v[0] == idx + 1] if v.max() > 1.0 else [v[idx]]

    def process_cls_preds(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        """更新分类任务的混淆矩阵。

        Args:
            preds (list[torch.Tensor]): 预测的类别标签。
            targets (list[torch.Tensor]): 真值类别标签。
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(
        self,
        detections: dict[str, torch.Tensor],
        batch: dict[str, Any],
        conf: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        """更新目标检测任务的混淆矩阵。

        Args:
            detections (dict[str, torch.Tensor]): 包含检测到的边界框及其关联信息的字典。应包含 'cls'、'conf' 和 'bboxes' 键，其中 'bboxes' 可以是常规框的 Array[N, 4] 或带角度 OBB 的 Array[N, 5]。
            batch (dict[str, Any]): 包含真值数据的批次字典，含 'bboxes' (Array[M, 4]| Array[M, 5]) 和 'cls' (Array[M]) 键，M 为真值目标数量。
            conf (float, optional): 检测的置信度阈值。
            iou_thres (float, optional): 匹配检测和真值的 IoU 阈值。
        """
        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
        if self.matches is not None:  # 仅在启用可视化时
            self.matches = {k: defaultdict(list) for k in {"TP", "FP", "FN", "GT"}}
            for i in range(gt_cls.shape[0]):
                self._append_matches("GT", batch, i)  # 存储 GT
        is_obb = gt_bboxes.shape[1] == 5  # 检查框是否包含 OBB 角度
        conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf  # 如果传入默认 conf 值则应用 0.25
        no_pred = detections["cls"].shape[0] == 0
        if gt_cls.shape[0] == 0:  # 检查标签是否为空
            if not no_pred:
                detections = {k: detections[k][detections["conf"] > conf] for k in detections}
                detection_classes = detections["cls"].int().tolist()
                for i, dc in enumerate(detection_classes):
                    self.matrix[dc, self.nc] += 1  # FP
                    self._append_matches("FP", detections, i)
            return
        if no_pred:
            gt_classes = gt_cls.int().tolist()
            for i, gc in enumerate(gt_classes):
                self.matrix[self.nc, gc] += 1  # FN
                self._append_matches("FN", batch, i)
            return

        detections = {k: detections[k][detections["conf"] > conf] for k in detections}
        gt_classes = gt_cls.int().tolist()
        detection_classes = detections["cls"].int().tolist()
        bboxes = detections["bboxes"]
        iou = batch_probiou(gt_bboxes, bboxes) if is_obb else box_iou(gt_bboxes, bboxes)

        x = torch.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                dc = detection_classes[m1[j].item()]
                self.matrix[dc, gc] += 1  # 类别正确则为 TP，否则同时为 FP 和 FN
                if dc == gc:
                    self._append_matches("TP", detections, m1[j].item())
                else:
                    self._append_matches("FP", detections, m1[j].item())
                    self._append_matches("FN", batch, i)
            else:
                self.matrix[self.nc, gc] += 1  # FN
                self._append_matches("FN", batch, i)

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # FP
                self._append_matches("FP", detections, i)

    def matrix(self):
        """Return the confusion matrix."""
        return self.matrix

    def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
        """返回真正例和假正例。

        Returns:
            tp (np.ndarray): 真正例。
            fp (np.ndarray): 假正例。
        """
        tp = self.matrix.diagonal()  # 真正例
        fp = self.matrix.sum(1) - tp  # 假正例
        # fn = self.matrix.sum(0) - tp  # 假负例（漏检）
        return (tp, fp) if self.task == "classify" else (tp[:-1], fp[:-1])  # 如果 task=detect 则移除背景类

    def plot_matches(self, img: torch.Tensor, im_file: str, save_dir: Path) -> None:
        """为每张图像绘制 GT、TP、FP、FN 的网格图。

        Args:
            img (torch.Tensor): 要绘制的图像。
            im_file (str): 用于保存可视化的图像文件名。
            save_dir (Path): 保存可视化的目录。
        """
        if not self.matches:
            return
        from .ops import xyxy2xywh
        from .plotting import plot_images

        # 创建 4 个批次 (GT, TP, FP, FN)
        labels = defaultdict(list)
        for i, mtype in enumerate(["GT", "FP", "TP", "FN"]):
            mbatch = self.matches[mtype]
            if "conf" not in mbatch:
                mbatch["conf"] = torch.tensor([1.0] * len(mbatch["bboxes"]), device=img.device)
            mbatch["batch_idx"] = torch.ones(len(mbatch["bboxes"]), device=img.device) * i
            for k in mbatch:
                labels[k] += mbatch[k]

        labels = {k: torch.stack(v, 0) if len(v) else torch.empty(0) for k, v in labels.items()}
        if self.task != "obb" and labels["bboxes"].shape[0]:
            labels["bboxes"] = xyxy2xywh(labels["bboxes"])
        (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        plot_images(
            labels,
            img.repeat(4, 1, 1, 1),
            paths=["Ground Truth", "False Positives", "True Positives", "False Negatives"],
            fname=save_dir / "visualizations" / Path(im_file).name,
            names=self.names,
            max_subplots=4,
            conf_thres=0.001,
        )

    @TryExcept(msg="ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
        """使用 matplotlib 绘制混淆矩阵并保存到文件。

        Args:
            normalize (bool, optional): 是否归一化混淆矩阵。
            save_dir (str, optional): 保存图表的目录。
            on_plot (callable, optional): 图表渲染时传递路径和数据的可选回调。
        """
        import matplotlib.pyplot as plt  # 作用域以加速 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # 归一化列
        array[array < 0.005] = np.nan  # 不标注（否则会显示为 0.00）

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        names, n = list(self.names.values()), self.nc
        if self.nc >= 100:  # 大类别数时下采样
            k = max(2, self.nc // 60)  # 下采样步长，始终 > 1
            keep_idx = slice(None, None, k)  # 创建切片而非数组
            names = names[keep_idx]  # 切片类别名称
            array = array[keep_idx, :][:, keep_idx]  # 切片矩阵行和列
            n = (self.nc + k - 1) // k  # 保留的类别数
        nc = n if self.task == "classify" else n + 1  # 如需要则调整背景
        ticklabels = "auto"
        if 0 < nc < 99:
            ticklabels = names if self.task == "classify" else [*names, "background"]
        xy_ticks = np.arange(len(ticklabels)) if ticklabels != "auto" else np.arange(nc)
        tick_fontsize = max(6, 15 - 0.1 * nc)  # 最小尺寸为 6
        label_fontsize = max(6, 12 - 0.1 * nc)
        title_fontsize = max(6, 12 - 0.1 * nc)
        btm = max(0.1, 0.25 - 0.001 * nc)  # 最小值为 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制空矩阵 RuntimeWarning: All-NaN slice encountered
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            ax.xaxis.set_label_position("bottom")
            if nc < 30:  # 为混淆矩阵的每个单元格添加分数
                color_threshold = 0.45 * (1 if normalize else np.nanmax(array))  # 文本颜色阈值
                for i, row in enumerate(array[:nc]):
                    for j, val in enumerate(row[:nc]):
                        val = array[i, j]
                        if np.isnan(val):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}" if normalize else f"{int(val)}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white" if val > color_threshold else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
        ax.set_title(title, fontsize=title_fontsize, pad=20)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
        if ticklabels != "auto":
            ax.set_xticklabels(ticklabels, fontsize=tick_fontsize, rotation=90, ha="center")
            ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
        for s in {"left", "right", "bottom", "top", "outline"}:
            if s != "outline":
                ax.spines[s].set_visible(False)  # 混淆矩阵图没有外框
            cbar.ax.spines[s].set_visible(False)
        fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)  # 调整布局以确保边距相等
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname, {"type": "confusion_matrix", "matrix": self.matrix.tolist()})

    def print(self):
        """将混淆矩阵打印到控制台。"""
        for i in range(self.matrix.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix[i])))

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, float]]:
        """生成混淆矩阵的摘要表示（字典列表），可选归一化。适用于将矩阵导出为 CSV、XML、HTML、JSON 或 SQL 等格式。

        Args:
            normalize (bool): 是否归一化混淆矩阵值。
            decimals (int): 输出值的小数位数。

        Returns:
            (list[dict[str, float]]): 字典列表，每个字典表示一个预测类别及其对应的所有真实类别值。

        Examples:
            >>> results = model.val(data="coco8.yaml", plots=True)
            >>> cm_dict = results.confusion_matrix.summary(normalize=True, decimals=5)
            >>> print(cm_dict)
        """
        import re

        names = list(self.names.values()) if self.task == "classify" else [*list(self.names.values()), "background"]
        clean_names, seen = [], set()
        for name in names:
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            original_clean = clean_name
            counter = 1
            while clean_name.lower() in seen:
                clean_name = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean_name.lower())
            clean_names.append(clean_name)
        array = (self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)).round(decimals)
        return [
            dict({"Predicted": clean_names[i]}, **{clean_names[j]: array[i, j] for j in range(len(clean_names))})
            for i in range(len(clean_names))
        ]


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """分数 f 的盒式滤波器。"""
    nf = round(len(y) * f * 2) // 2 + 1  # 滤波元素数量（必须为奇数）
    p = np.ones(nf // 2)  # 全 1 填充
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y 填充
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y 平滑


@plt_settings()
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {},
    on_plot=None,
):
    """绘制精确率-召回率曲线。

    Args:
        px (np.ndarray): PR 曲线的 X 值。
        py (np.ndarray): PR 曲线的 Y 值。
        ap (np.ndarray): 平均精度值。
        save_dir (Path, optional): 保存图表的路径。
        names (dict[int, str], optional): 类别索引到类别名称的映射字典。
        on_plot (callable, optional): 图表保存后调用的函数。
    """
    import matplotlib.pyplot as plt  # 作用域以加速 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # 如果类别数 < 21 则显示每个类别的图例
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # 绘制(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="gray")  # 绘制(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        # 传递 PR 曲线数据用于交互式绘图（类别名称存储在模型级别）
        # 转置 py 以匹配其他曲线: y[class][point] 格式
        on_plot(save_dir, {"type": "pr_curve", "x": px.tolist(), "y": py.T.tolist(), "ap": ap.tolist()})


@plt_settings()
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    """绘制指标-置信度曲线。

    Args:
        px (np.ndarray): 指标-置信度曲线的 X 值。
        py (np.ndarray): 指标-置信度曲线的 Y 值。
        save_dir (Path, optional): 保存图表的路径。
        names (dict[int, str], optional): 类别索引到类别名称的映射字典。
        xlabel (str, optional): X 轴标签。
        ylabel (str, optional): Y 轴标签。
        on_plot (callable, optional): 图表保存后调用的函数。
    """
    import matplotlib.pyplot as plt  # 作用域以加速 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # 如果类别数 < 21 则显示每个类别的图例
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # 绘制(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")  # 绘制(confidence, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        # 传递指标-置信度曲线数据用于交互式绘图（类别名称存储在模型级别）
        on_plot(save_dir, {"type": f"{ylabel.lower()}_curve", "x": px.tolist(), "y": py.tolist()})


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """给定召回率和精确率曲线，计算平均精度（AP）。

    Args:
        recall (list[float]): 召回率曲线。
        precision (list[float]): 精确率曲线。

    Returns:
        ap (float): 平均精度。
        mpre (np.ndarray): 精确率包络曲线。
        mrec (np.ndarray): 在开头和末尾添加哨兵值的修改后召回率曲线。
    """
    # 在开头和末尾添加哨兵值
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    # 计算精确率包络
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 计算曲线下面积
    method = "interp"  # 方法: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101 点插值 (COCO)
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz 已弃用
        ap = func(np.interp(x, mrec, mpre), x)  # 积分
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # x 轴（召回率）变化的点
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # 曲线下面积

    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """计算目标检测评估中每个类别的平均精度。

    Args:
        tp (np.ndarray): 指示检测是否正确的二值数组（True 或 False）。
        conf (np.ndarray): 检测的置信度分数数组。
        pred_cls (np.ndarray): 检测的预测类别数组。
        target_cls (np.ndarray): 目标的真实类别数组。
        plot (bool, optional): 是否绘制 PR 曲线。
        on_plot (callable, optional): 图表渲染时传递路径和数据的回调。
        save_dir (Path, optional): 保存 PR 曲线的目录。
        names (dict[int, str], optional): 用于绘制 PR 曲线的类别名称字典。
        eps (float, optional): 避免除零的小值。
        prefix (str, optional): 保存图表文件的前缀字符串。

    Returns:
        tp (np.ndarray): 每个类别在最大 F1 指标阈值下的真正例计数。
        fp (np.ndarray): 每个类别在最大 F1 指标阈值下的假正例计数。
        p (np.ndarray): 每个类别在最大 F1 指标阈值下的精确率值。
        r (np.ndarray): 每个类别在最大 F1 指标阈值下的召回率值。
        f1 (np.ndarray): 每个类别在最大 F1 指标阈值下的 F1 分数值。
        ap (np.ndarray): 每个类别在不同 IoU 阈值下的平均精度。
        unique_classes (np.ndarray): 有数据的唯一类别数组。
        p_curve (np.ndarray): 每个类别的精确率曲线。
        r_curve (np.ndarray): 每个类别的召回率曲线。
        f1_curve (np.ndarray): 每个类别的 F1 分数曲线。
        x (np.ndarray): 曲线的 X 轴值。
        prec_values (np.ndarray): 每个类别在 mAP@0.5 的精确率值。
    """
    # 按目标性排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 查找唯一类别
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # 类别数量, number of detections

    # 为每个类别创建精确率-召回率曲线并计算 AP
    x, prec_values = np.linspace(0, 1, 1000), []

    # 平均精度、精确率和召回率曲线
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # 标签数
        n_p = i.sum()  # 预测数
        if n_p == 0 or n_l == 0:
            continue

        # 累积 FP 和 TP
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # 召回率
        recall = tpc / (n_l + eps)  # 召回率曲线
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # 负 x、xp，因为 xp 递减

        # 精确率
        precision = tpc / (tpc + fpc)  # 精确率曲线
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # pr_score 处的 p

        # 从召回率-精确率曲线计算 AP
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # mAP@0.5 处的精确率

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # 计算 F1（精确率和召回率的调和平均）
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # 字典: 仅包含有数据的类别
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # 最大 F1 索引
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # 最大 F1 处的精确率、召回率、F1 值
    tp = (r * nt).round()  # 真正例
    fp = (tp / (p + eps) - tp).round()  # 假正例
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class Metric(SimpleClass):
    """用于计算 Ultralytics YOLO 模型评估指标的类。

    属性:
        p (list): 每个类别的精确率。形状: (nc,)。
        r (list): 每个类别的召回率。形状: (nc,)。
        f1 (list): 每个类别的 F1 分数。形状: (nc,)。
        all_ap (list): 所有类别和所有 IoU 阈值的 AP 分数。形状: (nc, 10)。
        ap_class_index (list): 每个 AP 分数的类别索引。形状: (nc,)。
        nc (int): 类别数量。

    方法:
        ap50: 所有类别在 IoU 阈值 0.5 的 AP。
        ap: 所有类别在 IoU 阈值 0.5 到 0.95 的 AP。
        mp: 所有类别的平均精确率。
        mr: 所有类别的平均召回率。
        map50: 所有类别在 IoU 阈值 0.5 的平均 AP。
        map75: 所有类别在 IoU 阈值 0.75 的平均 AP。
        map: 所有类别在 IoU 阈值 0.5 到 0.95 的平均 AP。
        mean_results: 结果均值，返回 mp、mr、map50、map。
        class_result: 类别相关结果，返回 p[i]、r[i]、ap50[i]、ap[i]。
        maps: 每个类别的 mAP。
        fitness: 作为指标加权组合的模型适应度。
        update: 用新的评估结果更新指标属性。
        curves: 提供用于访问特定指标（如精确率、召回率、F1 等）的曲线列表。
        curves_results: 提供用于访问特定指标（如精确率、召回率、F1 等）的结果列表。
    """

    def __init__(self) -> None:
        """初始化 Metric 实例，用于计算 YOLO 模型的评估指标。"""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0
        self.image_metrics = {}

    @property
    def ap50(self) -> np.ndarray | list:
        """返回所有类别在 IoU 阈值 0.5 的平均精度（AP）。

        Returns:
            (np.ndarray | list): 形状 (nc,) 的 AP50 值数组，如不可用则为空列表。
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self) -> np.ndarray | list:
        """返回所有类别在 IoU 阈值 0.5-0.95 的平均精度（AP）。

        Returns:
            (np.ndarray | list): 形状 (nc,) 的 AP50-95 值数组，如不可用则为空列表。
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        """返回所有类别的平均精确率。

        Returns:
            (float): 所有类别的平均精确率。
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        """返回所有类别的平均召回率。

        Returns:
            (float): 所有类别的平均召回率。
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        """返回 IoU 阈值 0.5 的平均精度均值（mAP）。

        Returns:
            (float): IoU 阈值 0.5 的 mAP。
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self) -> float:
        """返回 IoU 阈值 0.75 的平均精度均值（mAP）。

        Returns:
            (float): IoU 阈值 0.75 的 mAP。
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        """返回 IoU 阈值 0.5-0.95（步长 0.05）的平均精度均值（mAP）。

        Returns:
            (float): IoU 阈值 0.5-0.95（步长 0.05）的 mAP。
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self) -> list[float]:
        """返回结果均值：mp、mr、map50、map。"""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """返回类别相关结果：p[i]、r[i]、ap50[i]、ap[i]。"""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self) -> np.ndarray:
        """返回每个类别的 mAP。"""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self) -> float:
        """返回作为指标加权组合的模型适应度。"""
        w = [0.0, 0.0, 0.0, 1.0]  # [P, R, mAP@0.5, mAP@0.5:0.95] 的权重
        return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())

    def update(self, results: tuple):
        """用新的结果集更新评估指标。

        Args:
            results (tuple): 包含评估指标的元组：
                - p (list): 每个类别的精确率。
                - r (list): 每个类别的召回率。
                - f1 (list): 每个类别的 F1 分数。
                - all_ap (list): 所有类别和所有 IoU 阈值的 AP 分数。
                - ap_class_index (list): 每个 AP 分数的类别索引。
                - p_curve (list): 每个类别的精确率曲线。
                - r_curve (list): 每个类别的召回率曲线。
                - f1_curve (list): 每个类别的 F1 曲线。
                - px (list): 曲线的 X 值。
                - prec_values (list): 每个类别的精确率值。
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    def clear_image_metrics(self) -> None:
        """清除当前验证运行中存储的逐图像指标。"""
        self.image_metrics.clear()

    @property
    def curves(self) -> list:
        """返回用于访问特定指标曲线的曲线列表。"""
        return []

    @property
    def curves_results(self) -> list[list]:
        """返回用于访问特定指标曲线的结果列表。"""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]

    def update_image_metrics(self, tp: np.ndarray, target_cls: np.ndarray, pred_cls: np.ndarray, im_name: str) -> None:
        """更新 IoU 阈值 0.5 处的逐图像精确率、召回率、F1、TP、FP 和 FN。

        Args:
            tp (np.ndarray): 形状 (num_preds, num_iou_thresholds) 的真正例数组，使用第一列（IoU >= 0.5）。
            target_cls (np.ndarray): 图像的真值类别标签。
            pred_cls (np.ndarray): 图像的预测类别标签。
            im_name (str): 用作逐图像键的图像文件名。
        """
        # 使用默认 IoU=0.5 列以匹配验证器的图像级匹配策略。
        tp = int(tp[:, 0].sum())
        num_preds = pred_cls.shape[0]
        num_targets = target_cls.shape[0]
        fp = num_preds - tp
        fn = num_targets - tp
        if num_preds == 0 and num_targets == 0:
            # 无 GT 且无预测的图像是平凡正确的调用，因此报告完美分数，而非使用下面标准的 0/0 回退将 P/R/F1 置零。
            precision = recall = f1 = 1.0
        else:
            precision = tp / num_preds if num_preds else 0.0
            recall = tp / num_targets if num_targets else 0.0
            denom = precision + recall
            f1 = 2 * precision * recall / denom if denom else 0.0
        self.image_metrics[im_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }


class DetMetrics(SimpleClass, DataExportMixin):
    """Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

    Attributes:
        names (dict[int, str]): A dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        update_stats: 通过追加新值到现有统计集合来更新统计数据。
        process: 处理目标检测的预测结果并更新指标。
        clear_stats: 清除存储的统计数据。
        keys: 返回用于访问特定指标的键列表。
        mean_results: 计算检测目标的均值并返回精确率、召回率、mAP50 和 mAP50-95。
        class_result: 返回目标检测模型在特定类别上的评估结果。
        maps: 返回每个类别的平均精度均值（mAP）分数。
        fitness: 返回 box 对象的适应度。
        ap_class_index: 返回每个类别的平均精度索引。
        results_dict: 返回计算得到的性能指标和统计数据的字典。
        curves: 返回用于访问特定指标曲线的曲线列表。
        curves_results: 返回计算得到的性能指标和统计数据列表。
        summary: Generate a summarized representation of per-class detection metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """使用类别名称初始化 DetMetrics 实例。

        Args:
            names (dict[int, str], optional): 类别名称字典。
        """
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.nt_per_class = None
        self.nt_per_image = None

    def update_stats(self, stat: dict[str, Any]) -> None:
        """通过追加新值到现有统计集合来更新统计数据。

        Args:
            stat (dict[str, Any]): 包含要追加的新统计值的字典。键应与 self.stats 中的现有键匹配。
        """
        for k in self.stats.keys():
            self.stats[k].append(stat[k])
        self.box.update_image_metrics(stat["tp"], stat["target_cls"], stat["pred_cls"], stat["im_name"])

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """处理目标检测的预测结果并更新指标。

        Args:
            save_dir (Path): 保存图表的目录。默认 Path(".")。
            plot (bool): 是否绘制精确率-召回率曲线。默认 False。
            on_plot (callable, optional): 图表生成后调用的函数。默认 None。

        Returns:
            (dict[str, np.ndarray]): 包含连接的统计数组的字典。
        """
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}  # 转为 numpy
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
        return stats

    def clear_stats(self):
        """清除存储的统计数据。"""
        for v in self.stats.values():
            v.clear()

    def clear_image_metrics(self) -> None:
        """清除存储的逐图像指标。"""
        self.box.clear_image_metrics()

    @property
    def keys(self) -> list[str]:
        """返回用于访问特定指标的键列表。"""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self) -> list[float]:
        """计算检测目标的均值并返回精确率、召回率、mAP50 和 mAP50-95。"""
        return self.box.mean_results()

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """返回目标检测模型在特定类别上的评估结果。"""
        return self.box.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """返回每个类别的平均精度均值（mAP）分数。"""
        return self.box.maps

    @property
    def fitness(self) -> float:
        """返回 box 对象的适应度。"""
        return self.box.fitness()

    @property
    def ap_class_index(self) -> list:
        """返回每个类别的平均精度索引。"""
        return self.box.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """返回计算得到的性能指标和统计数据的字典。"""
        keys = [*self.keys, "fitness"]
        values = ((float(x) if hasattr(x, "item") else x) for x in ([*self.mean_results(), self.fitness]))
        return dict(zip(keys, values))

    @property
    def curves(self) -> list[str]:
        """返回用于访问特定指标曲线的曲线列表。"""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self) -> list[list]:
        """返回计算得到的性能指标和统计数据列表。"""
        return self.box.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """生成每个类别的检测指标摘要表示（字典列表）。包含共享标量指标（mAP、mAP50、mAP75）以及每个类别的精确率、召回率和 F1 分数。

        Args:
            normalize (bool): 对于 Detect 指标，默认全部归一化到 [0-1]。
            decimals (int): 指标值的小数位数。

        Returns:
            (list[dict[str, Any]]): 字典列表，每个字典表示一个类别及其对应的指标值。

        Examples:
           >>> results = model.val(data="coco8.yaml")
           >>> detection_summary = results.summary()
           >>> print(detection_summary)
        """
        per_class = {
            "Box-P": self.box.p,
            "Box-R": self.box.r,
            "Box-F1": self.box.f1,
        }
        return [
            {
                "Class": self.names[self.ap_class_index[i]],
                "Images": self.nt_per_image[self.ap_class_index[i]],
                "Instances": self.nt_per_class[self.ap_class_index[i]],
                **{k: round(v[i], decimals) for k, v in per_class.items()},
                "mAP50": round(self.class_result(i)[2], decimals),
                "mAP50-95": round(self.class_result(i)[3], decimals),
            }
            for i in range(len(per_class["Box-P"]))
        ]


class SegmentMetrics(DetMetrics):
    """Calculate and aggregate detection and segmentation metrics over a given set of classes.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        process: 处理给定预测集上的检测和分割指标。
        keys: 返回用于访问指标的键列表。
        mean_results: 返回边界框和分割结果的均值指标。
        class_result: 返回指定类别索引的分类结果。
        maps: 返回目标检测和分割模型的 mAP 分数。
        fitness: 返回分割和边界框模型的适应度分数。
        curves: 返回用于访问特定指标曲线的曲线列表。
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class segmentation metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """使用类别名称初始化 SegmentMetrics 实例。

        Args:
            names (dict[int, str], optional): 类别名称字典。
        """
        DetMetrics.__init__(self, names)
        self.seg = Metric()
        self.stats["tp_m"] = []  # 添加额外的掩码统计

    def update_stats(self, stat: dict[str, Any]) -> None:
        """通过追加新值到现有统计集合来更新统计数据。

        Args:
            stat (dict[str, Any]): 包含要追加的新统计值的字典。键应与 self.stats 中的现有键匹配。
        """
        super().update_stats(stat)  # 更新 box 统计
        self.seg.update_image_metrics(stat["tp_m"], stat["target_cls"], stat["pred_cls"], stat["im_name"])

    def clear_image_metrics(self) -> None:
        """清除存储的逐图像指标。"""
        super().clear_image_metrics()
        self.seg.clear_image_metrics()

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """处理给定预测集上的检测和分割指标。

        Args:
            save_dir (Path): 保存图表的目录。默认 Path(".")。
            plot (bool): 是否绘制精确率-召回率曲线。默认 False。
            on_plot (callable, optional): 图表生成后调用的函数。默认 None。

        Returns:
            (dict[str, np.ndarray]): 包含连接的统计数组的字典。
        """
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # 处理 box 统计
        results_mask = ap_per_class(
            stats["tp_m"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        return stats

    @property
    def keys(self) -> list[str]:
        """返回用于访问指标的键列表。"""
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self) -> list[float]:
        """返回边界框和分割结果的均值指标。"""
        return DetMetrics.mean_results(self) + self.seg.mean_results()

    def class_result(self, i: int) -> list[float]:
        """返回指定类别索引的分类结果。"""
        return DetMetrics.class_result(self, i) + self.seg.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """返回目标检测和分割模型的 mAP 分数。"""
        return DetMetrics.maps.fget(self) + self.seg.maps

    @property
    def fitness(self) -> float:
        """返回分割和边界框模型的适应度分数。"""
        return self.seg.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        """返回用于访问特定指标曲线的曲线列表。"""
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """返回计算得到的性能指标和统计数据列表。"""
        return DetMetrics.curves_results.fget(self) + self.seg.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """生成每个类别的分割指标摘要表示（字典列表）。包含 box 和 mask 的标量指标（mAP、mAP50、mAP75）以及每个类别的精确率、召回率和 F1 分数。

        Args:
            normalize (bool): 对于 Segment 指标，默认全部归一化到 [0-1]。
            decimals (int): 指标值的小数位数。

        Returns:
            (list[dict[str, Any]]): 字典列表，每个字典表示一个类别及其对应的指标值。

        Examples:
            >>> results = model.val(data="coco8-seg.yaml")
            >>> seg_summary = results.summary(decimals=4)
            >>> print(seg_summary)
        """
        per_class = {
            "Mask-P": self.seg.p,
            "Mask-R": self.seg.r,
            "Mask-F1": self.seg.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)  # 获取 box 摘要
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
        return summary


class PoseMetrics(DetMetrics):
    """Calculate and aggregate detection and pose metrics over a given set of classes.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        pose (Metric): An instance of the Metric class to calculate pose metrics.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        process: 处理给定预测集上的检测和姿态指标。
        keys: 返回用于访问指标的键列表。
        mean_results: 返回 box 和姿态的均值结果。
        class_result: 返回特定类别 i 的逐类别检测结果。
        maps: 返回 box 和姿态检测每个类别的平均精度均值（mAP）。
        fitness: 返回姿态和 box 检测的组合适应度分数。
        curves: 返回用于访问特定指标曲线的曲线列表。
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class pose metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """使用类别名称初始化 PoseMetrics 类。

        Args:
            names (dict[int, str], optional): 类别名称字典。
        """
        super().__init__(names)
        self.pose = Metric()
        self.stats["tp_p"] = []  # 添加额外的姿态统计

    def update_stats(self, stat: dict[str, Any]) -> None:
        """通过追加新值到现有统计集合来更新统计数据。

        Args:
            stat (dict[str, Any]): 包含要追加的新统计值的字典。键应与 self.stats 中的现有键匹配。
        """
        super().update_stats(stat)  # 更新 box 统计
        self.pose.update_image_metrics(stat["tp_p"], stat["target_cls"], stat["pred_cls"], stat["im_name"])

    def clear_image_metrics(self) -> None:
        """清除存储的逐图像指标。"""
        super().clear_image_metrics()
        self.pose.clear_image_metrics()

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """处理给定预测集上的检测和姿态指标。

        Args:
            save_dir (Path): 保存图表的目录。默认 Path(".")。
            plot (bool): 是否绘制精确率-召回率曲线。默认 False。
            on_plot (callable, optional): Function to call after plots are generated.

        Returns:
            (dict[str, np.ndarray]): 包含连接的统计数组的字典。
        """
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # 处理 box 统计
        results_pose = ap_per_class(
            stats["tp_p"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        return stats

    @property
    def keys(self) -> list[str]:
        """返回评估指标键列表。"""
        return [
            *DetMetrics.keys.fget(self),
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self) -> list[float]:
        """返回 box 和姿态的均值结果。"""
        return DetMetrics.mean_results(self) + self.pose.mean_results()

    def class_result(self, i: int) -> list[float]:
        """返回特定类别 i 的逐类别检测结果。"""
        return DetMetrics.class_result(self, i) + self.pose.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """返回 box 和姿态检测每个类别的平均精度均值（mAP）。"""
        return DetMetrics.maps.fget(self) + self.pose.maps

    @property
    def fitness(self) -> float:
        """返回姿态和 box 检测的组合适应度分数。"""
        return self.pose.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        """返回用于访问特定指标曲线的曲线列表。"""
        return [
            *DetMetrics.curves.fget(self),
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """返回计算得到的性能指标和统计数据列表。"""
        return DetMetrics.curves_results.fget(self) + self.pose.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """生成每个类别的姿态指标摘要表示（字典列表）。包含 box 和姿态的标量指标（mAP、mAP50、mAP75）以及每个类别的精确率、召回率和 F1 分数。

        Args:
            normalize (bool): 对于 Pose 指标，默认全部归一化到 [0-1]。
            decimals (int): 指标值的小数位数。

        Returns:
            (list[dict[str, Any]]): 字典列表，每个字典表示一个类别及其对应的指标值。

        Examples:
            >>> results = model.val(data="coco8-pose.yaml")
            >>> pose_summary = results.summary(decimals=4)
            >>> print(pose_summary)
        """
        per_class = {
            "Pose-P": self.pose.p,
            "Pose-R": self.pose.r,
            "Pose-F1": self.pose.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)  # 获取 box 摘要
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
        return summary


class ClassifyMetrics(SimpleClass, DataExportMixin):
    """Class for computing classification metrics including top-1 and top-5 accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        top5 (float): The top-5 accuracy.
        speed (dict[str, float]): A dictionary containing the time taken for each step in the pipeline.

    Methods:
        process: 处理目标类别和预测类别以计算指标。
        fitness: 返回 Top-1 和 Top-5 准确率的均值作为适应度分数。
        results_dict: 返回包含模型性能指标和适应度分数的字典。
        keys: 返回 results_dict 属性的键列表。
        curves: 返回用于访问特定指标曲线的曲线列表。
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: 生成分类指标（Top-1 和 Top-5 准确率）的单行摘要。
    """

    def __init__(self) -> None:
        """初始化 ClassifyMetrics 实例。"""
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

    def process(self, targets: torch.Tensor, pred: torch.Tensor):
        """处理目标类别和预测类别以计算指标。

        Args:
            targets (torch.Tensor): 目标类别。
            pred (torch.Tensor): 预测类别。
        """
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) 准确率
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self) -> float:
        """返回 Top-1 和 Top-5 准确率的均值作为适应度分数。"""
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self) -> dict[str, float]:
        """返回包含模型性能指标和适应度分数的字典。"""
        return dict(zip([*self.keys, "fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self) -> list[str]:
        """返回 results_dict 属性的键列表。"""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self) -> list:
        """返回用于访问特定指标曲线的曲线列表。"""
        return []

    @property
    def curves_results(self) -> list:
        """返回用于访问特定指标曲线的结果列表。"""
        return []

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, float]]:
        """生成分类指标（Top-1 和 Top-5 准确率）的单行摘要。

        Args:
            normalize (bool): 对于 Classify 指标，默认全部归一化到 [0-1]。
            decimals (int): 指标值的小数位数。

        Returns:
            (list[dict[str, float]]): 包含一个字典的列表，该字典包含 Top-1 和 Top-5 分类准确率。

        Examples:
            >>> results = model.val(data="imagenet10")
            >>> classify_summary = results.summary(decimals=4)
            >>> print(classify_summary)
        """
        return [{"top1_acc": round(self.top1, decimals), "top5_acc": round(self.top5, decimals)}]


class OBBMetrics(DetMetrics):
    """用于评估定向边界框（OBB）检测的指标。

    属性:
        names (dict[int, str]): 类别名称字典。
        box (Metric): 用于存储检测结果的 Metric 类实例。
        speed (dict[str, float]): 存储检测过程各部分执行时间的字典。
        stats (dict[str, list]): 包含真正例、置信度分数、预测类别、目标类别和目标图像列表的字典。
        nt_per_class: 每个类别的目标数。
        nt_per_image: 每张图像的目标数。

    References:
        https://arxiv.org/pdf/2106.06072.pdf
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """使用类别名称初始化 OBBMetrics 实例。

        Args:
            names (dict[int, str], optional): 类别名称字典。
        """
        DetMetrics.__init__(self, names)
