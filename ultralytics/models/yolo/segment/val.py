# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, mask_iou


class SegmentationValidator(DetectionValidator):
    """用于基于分割模型进行验证的类，继承自 DetectionValidator。

    该验证器处理分割模型的评估，同时处理边界框和掩码预测以
    计算检测和分割任务的 mAP 等指标。

    Attributes:
        plot_masks (list): 存储用于绘图的掩码列表。
        process (callable): 基于 save_json 和 save_txt 标记处理掩码的函数。
        args (SimpleNamespace): 验证器的参数。
        metrics (SegmentMetrics): 分割任务的指标计算器。
        stats (dict): 存储验证期间统计信息的字典。

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """初始化 SegmentationValidator，设置任务为 'segment'，指标为 SegmentMetrics。

        Args:
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的数据加载器。
            save_dir (Path, optional): 保存结果的目录。
            args (dict, optional): 验证器的参数。
            _callbacks (dict, optional): 回调函数字典。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """为 YOLO 分割验证预处理图像批次。

        Args:
            batch (dict[str, Any]): 包含图像和标注的批次。

        Returns:
            (dict[str, Any]): 预处理后的批次。
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化指标并根据 save_json 标志选择掩码处理函数。

        Args:
            model (torch.nn.Module): 需要验证的模型。
        """
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        # 更精确 vs 更快
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> str:
        """返回评估指标的格式化描述。"""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """对 YOLO 预测结果进行后处理，返回包含 proto 的输出检测结果。

        Args:
            preds (list[torch.Tensor]): 模型的原始预测结果。

        Returns:
            (list[dict[str, torch.Tensor]]): 包含掩码的已处理检测预测。
        """
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]  # 从 proto 获取图像尺寸
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """通过处理图像和目标来准备用于验证的批次。

        Args:
            si (int): 批次中的样本索引。
            batch (dict[str, Any]): 包含图像和标注的批次数据。

        Returns:
            (dict[str, Any]): 包含处理后标注的准备好的批次。
        """
        prepared_batch = super()._prepare_batch(si, batch)
        nl = prepared_batch["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // 4 for s in prepared_batch["imgsz"]]
            if masks.shape[1:] != mask_size:
                masks = F.interpolate(masks[None], mask_size, mode="bilinear", align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared_batch["masks"] = masks
        return prepared_batch

    def gather_stats(self) -> None:
        """从所有 GPU 收集统计信息。"""
        super().gather_stats()  # 从 DetectionValidator 收集统计信息
        self._gather_image_metrics(self.metrics.seg)

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """基于边界框和可选掩码计算批次的正确预测矩阵。

        Args:
            preds (dict[str, torch.Tensor]): 包含预测的字典，键包括 'cls' 和 'masks'。
            batch (dict[str, Any]): 包含批次数据的字典，键包括 'cls' 和 'masks'。

        Returns:
            (dict[str, np.ndarray]): 包含正确预测矩阵的字典，包括掩码 IoU 的 'tp_m'。

        Examples:
            >>> preds = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> batch = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> correct_preds = validator._process_batch(preds, batch)

        Notes:
            - 该方法计算预测掩码与真实掩码之间的 IoU。
            - 重叠掩码根据 overlap_mask 参数设置进行处理。
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_m = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            iou = mask_iou(batch["masks"].flatten(1), preds["masks"].flatten(1).float())  # float, uint8
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_m": tp_m})  # 使用掩码 IoU 更新 tp
        return tp

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """绘制包含掩码和边界框的批次预测。

        Args:
            batch (dict[str, Any]): 包含图像和标注的批次。
            preds (list[dict[str, torch.Tensor]]): 模型的预测列表。
            ni (int): 批次索引。
        """
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)  # 绘制边界框

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """将 YOLO 检测结果以归一化坐标按特定格式保存到 txt 文件。

        Args:
            predn (dict[str, torch.Tensor]): 包含 'bboxes'、'conf'、'cls' 和 'masks' 键的预测字典。
            save_conf (bool): 是否保存置信度分数。
            shape (tuple[int, int]): 原始图像的形状。
            file (Path): 保存检测结果的文件路径。
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测序列化为 COCO JSON 格式用于评估。

        Args:
            predn (dict[str, torch.Tensor]): 包含 bboxes、masks、置信度分数和类别的预测结果。
            pbatch (dict[str, Any]): 包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file' 的批次字典。
        """

        def to_string(counts: list[int]) -> str:
            """将 RLE 对象转换为紧凑的字符串表示。每个计数值经过差分编码和变长编码
            转换为字符串。

            Args:
                counts (list[int]): RLE 计数值列表。
            """
            result = []

            for i in range(len(counts)):
                x = int(counts[i])

                # 对第二个条目之后的所有计数值应用差分编码
                if i > 2:
                    x -= int(counts[i - 2])

                # 对值进行变长编码
                while True:
                    c = x & 0x1F  # Take 5 bits
                    x >>= 5

                    # 如果设置了符号位 (0x10)，若 x != -1 则继续；
                    # 否则，若 x != 0 则继续
                    more = (x != -1) if (c & 0x10) else (x != 0)
                    if more:
                        c |= 0x20  # 设置继续位
                    c += 48  # 转换为 ASCII
                    result.append(chr(c))
                    if not more:
                        break

            return "".join(result)

        def multi_encode(pixels: torch.Tensor) -> list[int]:
            """使用游程编码 (RLE) 转换多个二值掩码。

            Args:
                pixels (torch.Tensor): 一个 2D 张量，每行表示一个展平的二值掩码，形状为 [N, H*W]。

            Returns:
                (list[list[int]]): 每个掩码的 RLE 计数值列表。
            """
            transitions = pixels[:, 1:] != pixels[:, :-1]
            row_idx, col_idx = torch.where(transitions)
            col_idx = col_idx + 1

            # 计算游程长度
            counts = []
            for i in range(pixels.shape[0]):
                positions = col_idx[row_idx == i]
                if len(positions):
                    count = torch.diff(positions).tolist()
                    count.insert(0, positions[0].item())
                    count.append(len(pixels[i]) - positions[-1].item())
                else:
                    count = [len(pixels[i])]

                # 确保以背景 (0) 计数开始
                if pixels[i][0].item() == 1:
                    count = [0, *count]
                counts.append(count)

            return counts

        pred_masks = predn["masks"].transpose(2, 1).contiguous().view(len(predn["masks"]), -1)  # N, H*W
        h, w = predn["masks"].shape[1:3]
        counts = multi_encode(pred_masks)
        rles = []
        for c in counts:
            rles.append({"size": [h, w], "counts": to_string(c)})
        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r  # 分割

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """将预测缩放到原始图像尺寸。"""
        return {
            **super().scale_preds(predn, pbatch),
            "masks": ops.scale_masks(predn["masks"][None], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])[
                0
            ].byte(),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """返回 COCO 风格的实例分割评估指标。"""
        pred_json = self.save_dir / "predictions.json"  # 预测结果
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # 标注文件
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
