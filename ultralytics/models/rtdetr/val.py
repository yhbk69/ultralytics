# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRDataset(YOLODataset):
    """实时检测与跟踪（RT-DETR）数据集类，继承自基础 YOLODataset 类。

    该专用数据集类专为 RT-DETR 目标检测模型设计，并针对实时检测和跟踪任务进行了优化。

    Attributes:
        augment (bool): 是否应用数据增强。
        rect (bool): 是否使用矩形训练。
        use_segments (bool): 是否使用分割掩码。
        use_keypoints (bool): 是否使用关键点标注。
        imgsz (int): 训练的目标图像尺寸。

    Methods:
        load_image: 从数据集索引加载一张图像。
        build_transforms: 为数据集构建变换流水线。

    Examples:
        初始化 RT-DETR 数据集
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw0, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, **kwargs):
        """通过继承 YOLODataset 类来初始化 RTDETRDataset 类。

        此构造函数设置一个专门为 RT-DETR（实时检测与跟踪）模型优化的数据集，基于 YOLODataset 的基础功能构建。

        Args:
            *args (Any): 传递给父类 YOLODataset 的可变长度参数列表。
            data (dict | None): 包含数据集信息的字典。如果为 None，将使用默认值。
            **kwargs (Any): 传递给父类 YOLODataset 的额外关键字参数。
        """
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """从数据集索引 'i' 加载一张图像。

        Args:
            i (int): 要加载的图像索引。
            rect_mode (bool, optional): 是否使用矩形模式进行批量推理。

        Returns:
            im (np.ndarray): 加载的图像，NumPy 数组格式。
            hw_original (tuple[int, int]): 原始图像尺寸，格式为 (高度, 宽度)。
            hw_resized (tuple[int, int]): 缩放后的图像尺寸，格式为 (高度, 宽度)。

        Examples:
            从数据集加载一张图像
            >>> dataset = RTDETRDataset(img_path="path/to/images")
            >>> image, hw0, hw = dataset.load_image(0)
        """
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """为数据集构建变换流水线。

        Args:
            hyp (dict, optional): 变换的超参数。

        Returns:
            (Compose): 变换函数的组合。
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scale_fill=True)])
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """RTDETRValidator 继承自 DetectionValidator 类，为 RT-DETR（实时 DETR）目标检测模型提供专门的验证能力。

    该类允许为验证构建 RTDETR 特定的数据集，应用置信度阈值进行后处理，并相应更新评估指标。

    Attributes:
        args (Namespace): 验证的配置参数。
        data (dict): 数据集配置字典。

    Methods:
        build_dataset: 构建用于验证的 RTDETR 数据集。
        postprocess: 对预测输出应用置信度阈值过滤。

    Examples:
        初始化并运行 RT-DETR 验证
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Notes:
        有关属性和方法的更多详细信息，请参考父类 DetectionValidator。
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """构建 RTDETR 数据集。

        Args:
            img_path (str): 包含图像的文件夹路径。
            mode (str, optional): `train` 模式或 `val` 模式，用户可以为每种模式自定义不同的数据增强。
            batch (int, optional): 批次大小，用于 `rect` 训练。

        Returns:
            (RTDETRDataset): 为 RT-DETR 验证配置的数据集。
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # 无数据增强
            hyp=self.args,
            rect=False,  # 不使用矩形训练
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """原样返回预测结果，因为 RT-DETR 在后处理中处理缩放。"""
        return predn

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """对预测输出应用后处理。

        Top-k 选择已在解码器头部内部执行。此方法将归一化的 xywh 坐标转换为像素 xyxy 格式。

        Args:
            preds (torch.Tensor | list | tuple): 模型的预测结果，形状为 (batch_size, num_queries, 6)，
                其中最后一维为 [cx, cy, w, h, score, class]。

        Returns:
            (list[dict[str, torch.Tensor]]): 每张图像的字典列表，每个字典包含：
                - 'bboxes': 形状为 (N, 4) 的张量，包含 xyxy 像素格式的边界框坐标
                - 'conf': 形状为 (N,) 的张量，包含置信度分数
                - 'cls': 形状为 (N,) 的张量，包含类别索引
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.args.imgsz
        scores, labels = scores.squeeze(-1), labels.squeeze(-1)
        masks = scores > self.args.conf

        return [
            {"bboxes": bbox[m], "conf": score[m], "cls": label[m]}
            for bbox, score, label, m in zip(bboxes, scores, labels, masks)
        ]

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """将 YOLO 预测结果序列化为 COCO json 格式。

        Args:
            predn (dict[str, torch.Tensor]): 预测结果字典，包含 'bboxes'、'conf' 和 'cls' 键，
                分别对应边界框坐标、置信度分数和类别预测。
            pbatch (dict[str, Any]): 批次字典，包含 'imgsz'、'ori_shape'、'ratio_pad' 和 'im_file'。
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # 原始空间坐标的预测
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # 原始空间坐标的预测
        box = ops.xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # 将 xy 中心点转换为左上角坐标
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )
