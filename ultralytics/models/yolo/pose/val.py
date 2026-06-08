# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """用于基于姿态模型进行验证的类，继承自 DetectionValidator。

    该验证器专为姿态估计任务设计，处理关键点并实现姿态评估的
    专用指标。

    Attributes:
        sigma (np.ndarray): OKS 计算的 sigma 值，可为 OKS_SIGMA 或 1 除以关键点数量。
        kpt_shape (list[int]): 关键点形状，COCO 格式通常为 [17, 3]。
        args (dict): 验证器参数，包括设置为 "pose" 的任务类型。
        metrics (PoseMetrics): 姿态评估的指标对象。

    Methods:
        preprocess: 通过将关键点数据转换为浮点数并移到设备上来预处理批次。
        get_desc: 以字符串格式返回评估指标的描述。
        init_metrics: 初始化 YOLO 模型的姿态估计指标。
        _prepare_batch: 通过将关键点转换为浮点数并缩放到原始尺寸来准备批次。
        _prepare_pred: 为姿态处理准备并缩放预测中的关键点。
        _process_batch: 通过计算检测与真实标签之间的 IoU 返回正确的预测矩阵。
        plot_val_samples: 绘制并保存包含真实边界框和关键点的验证集样本。
        plot_predictions: 绘制并保存包含边界框和关键点的模型预测。
        save_one_txt: 将 YOLO 姿态检测结果以归一化坐标保存到文本文件。
        pred_to_json: 将 YOLO 预测转换为 COCO JSON 格式。
        eval_json: 使用 COCO JSON 格式评估目标检测模型。

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()

    Notes:
        该类继承 DetectionValidator 并添加了姿态专用功能。使用 OKS 计算的 sigma 值
        初始化，并设置 PoseMetrics 用于评估。使用 Apple MPS 时会显示警告，
        因为已知姿态模型存在 bug。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """初始化用于姿态估计验证的 PoseValidator 对象。

        该验证器专为姿态估计任务设计，处理关键点并实现姿态评估的
        专用指标。

        Args:
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的数据加载器。
            save_dir (Path | str, optional): 保存结果的目录。
            args (dict, optional): 验证器的参数，包括设置为 "pose" 的任务类型。
            _callbacks (dict, optional): 验证期间执行的回调函数字典。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """通过将关键点数据转换为浮点数并移到设备上来预处理批次。"""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def get_desc(self) -> str:
        """以字符串格式返回评估指标的描述。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化 YOLO 姿态验证的评估指标。

        Args:
            model (torch.nn.Module): 需要验证的模型。
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """对 YOLO 预测结果进行后处理，提取并重塑关键点用于姿态估计。

        该方法通过从预测的 'extra' 字段中提取关键点并按照关键点形状配置
        进行重塑，扩展了父类的后处理。关键点从扁平格式重塑为正确的
        维度结构（COCO 姿态格式通常为 [N, 17, 3]）。

        Args:
            preds (torch.Tensor): YOLO 姿态模型的原始预测张量，包含边界框、置信度
                分数、类别预测和关键点数据。

        Returns:
            (list[dict[str, torch.Tensor]]): 处理后的预测字典列表，每个字典包含：
                - 'bboxes': 边界框坐标
                - 'conf': 置信度分数
                - 'cls': 类别预测
                - 'keypoints': 重塑后的关键点坐标，形状为 (-1, *self.kpt_shape)

        Notes:
            如果预测中没有关键点（空关键点），则跳过该预测并继续
            处理下一个。关键点从包含基本检测之外的任务特定数据的
            'extra' 字段中提取。
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # 移除 extra（如果存在）
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """通过将关键点转换为浮点数并缩放到原始尺寸来准备处理的批次。

        Args:
            si (int): 批次中的样本索引。
            batch (dict[str, Any]): 包含批次数据的字典，键包括 'keypoints'、'batch_idx' 等。

        Returns:
            (dict[str, Any]): 关键点已缩放到原始图像尺寸的准备好的批次。

        Notes:
            该方法通过添加关键点处理扩展了父类的 _prepare_batch 方法。
            关键点从归一化坐标缩放到原始图像尺寸。
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """通过计算检测与真实标签之间的 IoU 返回正确的预测矩阵。

        Args:
            preds (dict[str, torch.Tensor]): 包含预测数据的字典，键 'cls' 表示类别预测，
                'keypoints' 表示关键点预测。
            batch (dict[str, Any]): 包含真实数据的字典，键 'cls' 表示类别标签，'bboxes'
                表示边界框，'keypoints' 表示关键点标注。

        Returns:
            (dict[str, np.ndarray]): 包含正确预测矩阵的字典，其中包括跨 10 个 IoU 级别的
                姿态真阳性 'tp_p'。

        Notes:
            面积计算中使用的 `0.53` 缩放因子参考自
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384。
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            # `0.53` 来自 https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # 使用关键点 IoU 更新 tp
        return tp

    def gather_stats(self) -> None:
        """从所有 GPU 收集统计信息。"""
        super().gather_stats()  # 从 DetectionValidator 收集统计信息
        self._gather_image_metrics(self.metrics.pose)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """将 YOLO 姿态检测结果以归一化坐标保存到文本文件。

        Args:
            predn (dict[str, torch.Tensor]): 包含键 'bboxes'、'conf'、'cls' 和 'keypoints' 的预测字典。
            save_conf (bool): 是否保存置信度分数。
            shape (tuple[int, int]): 原始图像形状 (高度, 宽度)。
            file (Path): 保存检测结果的输出文件路径。

        Notes:
            输出格式为：class_id x_center y_center width height confidence keypoints，其中
            关键点为每个点的归一化 (x, y, visibility) 值。
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测转换为 COCO JSON 格式。

        该方法接收预测张量和批次数据，将边界框从 YOLO 格式转换为 COCO
        格式，并将包含关键点的结果追加到内部 JSON 字典 (self.jdict) 中。

        Args:
            predn (dict[str, torch.Tensor]): 包含 'bboxes'、'conf'、'cls' 和 'kpts' 张量的
                预测字典。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file' 的批次字典。

        Notes:
            该方法从文件名字干中提取图像 ID（如果为数字则为整数，否则为字符串），
            将边界框从 xyxy 转换为 xywh 格式，并在保存到 JSON 字典前
            将坐标从中心点调整为左上角。
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["kpts"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # 关键点

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测缩放到原始图像尺寸。"""
        return {
            **super().scale_preds(predn, pbatch),
            "kpts": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """使用 COCO JSON 格式评估目标检测模型。"""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # 标注文件
        pred_json = self.save_dir / "predictions.json"  # 预测结果
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])
