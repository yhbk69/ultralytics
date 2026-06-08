# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
import time

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.ops import xywh2xyxy


def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # 类别数（可选）
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """对预测结果执行非极大值抑制（NMS）。

    基于置信度和 IoU 阈值应用 NMS 来过滤重叠的边界框。支持多种检测格式，
    包括标准框、旋转框和掩码。

    参数:
        prediction (torch.Tensor): 预测结果，形状为 (batch_size, num_classes + 4 + num_masks, num_boxes)，
            包含框、类别和可选掩码。
        conf_thres (float): 过滤检测的置信度阈值。有效值在 0.0 到 1.0 之间。
        iou_thres (float): NMS 过滤的 IoU 阈值。有效值在 0.0 到 1.0 之间。
        classes (list[int], 可选): 要考虑的类别索引列表。如果为 None，考虑所有类别。
        agnostic (bool): 是否执行类别无关的 NMS。
        multi_label (bool): 每个框是否可以有多个标签。
        labels (list[torch.Tensor]): 每张图像的先验标签。
        max_det (int): 每张图像保留的最大检测数。
        nc (int): 类别数。此索引之后被视为掩码。
        max_time_img (float): 处理一张图像的最大时间（秒）。
        max_nms (int): NMS 的最大框数。
        max_wh (int): 最大框宽度和高度（像素）。
        rotated (bool): 是否处理旋转边界框（OBB）。
        end2end (bool): 模型是否为端到端且不需要 NMS。
        return_idxs (bool): 是否返回保留检测的索引。

    返回:
        (list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]): 每张图像的检测列表，形状为
            (num_boxes, 6 + num_masks)，包含 (x1, y1, x2, y2, 置信度, 类别, mask1, mask2, ...)。
            如果 return_idxs=True，返回 (output, keepi) 元组，其中 keepi 包含保留检测的索引。
    """
    # 检查
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 验证模式，output = (inference_out, loss_out)
        prediction = prediction[0]  # 只选择推理输出
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # 端到端模型 (BNC, 即 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # 批量大小 (BCN, 即 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # 类别数
    extra = prediction.shape[1] - nc - 4  # 额外信息数
    mi = 4 + nc  # 掩码起始索引
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # 候选框
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # 用于跟踪索引

    # 设置
    # min_wh = 2  # （像素）最小框宽度和高度
    time_limit = 2.0 + max_time_img * bs  # 超时时间（秒）
    multi_label &= nc > 1  # 每个框多个标签（增加 0.5ms/图）

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) 转为 shape(1,6300,84)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh 转为 xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # 存储保留的索引
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # 图像索引，(预测, 预测索引)
        # 应用约束
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # 宽度-高度
        filt = xc[xi]  # 置信度
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        # 如果是自动标注，拼接先验标签
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # 框
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # 类别
            x = torch.cat((x, v), 0)

        # 如果没有剩余则处理下一张图像
        if not x.shape[0]:
            continue

        # 检测矩阵 nx6 (xyxy, 置信度, 类别)
        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:  # 仅最佳类别
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        # 按类别过滤
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # 检查形状
        n = x.shape[0]  # 框数量
        if not n:  # 无框
            continue
        if n > max_nms:  # 框过多
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # 按置信度排序并移除多余框
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别
        scores = x[:, 4]  # 分数
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c  # 框（按类别偏移）
            # 速度策略：torchvision 用于验证或已加载（更快），TorchNMS 用于预测（更低延迟）
            if "torchvision" in sys.modules:
                import torchvision  # 限定作用域，因为导入较慢

                i = torchvision.ops.nms(boxes, scores, iou_thres)
            else:
                i = TorchNMS.nms(boxes, scores, iou_thres)
        i = i[:max_det]  # 限制检测数

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # 超过时间限制

    return (output, keepi) if return_idxs else output


class TorchNMS:
    """Ultralytics 自定义 NMS 实现，为 YOLO 优化。

    该类提供对边界框执行非极大值抑制（NMS）操作的静态方法，
    包括标准 NMS、快速 NMS 和多类场景的批量 NMS。

    方法:
        fast_nms: 使用上三角矩阵操作的 Fast-NMS。
        nms: 具有提前终止的优化 NMS，与 torchvision 行为完全一致。
        batched_nms: 用于类别感知抑制的批量 NMS。

    示例:
        对框和分数执行标准 NMS
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = TorchNMS.nms(boxes, scores, 0.5)
    """

    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> torch.Tensor:
        """来自 https://arxiv.org/pdf/1904.02689 的 Fast-NMS 实现，使用上三角矩阵操作。

        参数:
            boxes (torch.Tensor): xyxy 格式的边界框，形状为 (N, 4)。
            scores (torch.Tensor): 置信度分数，形状为 (N,)。
            iou_threshold (float): 抑制的 IoU 阈值。
            use_triu (bool): 是否使用 torch.triu 操作符进行上三角矩阵操作。
            iou_func (callable): 计算框之间 IoU 的函数。
            exit_early (bool): 如果没有框是否提前退出。

        返回:
            (torch.Tensor): NMS 后保留的框索引。

        示例:
            对一组框应用 NMS
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.fast_nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            # 注意：处理 len(boxes) 的情况，因此可通过消除 if-else 条件来导出
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            # 将这些分数置零确保额外的索引不会影响最终结果
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_  # 更新原始张量用于 NMSModel
            # 注意：返回固定长度的索引以避免 TFLite 重塑错误
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """具有提前终止的优化 NMS，与 torchvision 行为完全一致。

        参数:
            boxes (torch.Tensor): xyxy 格式的边界框，形状为 (N, 4)。
            scores (torch.Tensor): 置信度分数，形状为 (N,)。
            iou_threshold (float): 抑制的 IoU 阈值。

        返回:
            (torch.Tensor): NMS 后保留的框索引。

        示例:
            对一组框应用 NMS
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # 预分配并一次提取坐标
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        # 按分数降序排序
        order = scores.argsort(0, descending=True)

        # 预分配最大可能大小的保留列表
        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break
            # 剩余框的向量化 IoU 计算
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            # 快速交集和 IoU
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            # 提前退出：如果没有交集则跳过 IoU 计算
            if inter.sum() == 0:
                # 与当前框无重叠，保留所有剩余框
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            # 保留 IoU <= 阈值的框
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> torch.Tensor:
        """用于类别感知抑制的批量 NMS。

        参数:
            boxes (torch.Tensor): xyxy 格式的边界框，形状为 (N, 4)。
            scores (torch.Tensor): 置信度分数，形状为 (N,)。
            idxs (torch.Tensor): 类别索引，形状为 (N,)。
            iou_threshold (float): 抑制的 IoU 阈值。
            use_fast_nms (bool): 是否使用 Fast-NMS 实现。

        返回:
            (torch.Tensor): NMS 后保留的框索引。

        示例:
            跨多个类别应用批量 NMS
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> idxs = torch.tensor([0, 1])
            >>> keep = TorchNMS.batched_nms(boxes, scores, idxs, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # 策略：按类别索引偏移框以防止跨类别抑制
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )
