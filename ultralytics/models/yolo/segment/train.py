# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """用于基于分割模型进行训练的类，继承自 DetectionTrainer。

    该训练器专门处理分割任务，使用分割专用功能扩展检测训练器，
    包括模型初始化、验证和可视化。

    Attributes:
        loss_names (tuple[str]): 训练期间使用的损失分量名称。

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """初始化 SegmentationTrainer 对象。

        Args:
            cfg (dict): 包含默认训练设置的配置字典。
            overrides (dict, optional): 覆盖默认配置的参数字典。
            _callbacks (dict, optional): 训练期间执行的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """初始化并返回使用指定配置和权重的 SegmentationModel。

        Args:
            cfg (dict | str, optional): 模型配置。可以是字典、YAML 文件路径或 None。
            weights (str | Path, optional): 预训练权重文件的路径。
            verbose (bool): 是否在初始化期间显示模型信息。

        Returns:
            (SegmentationModel): 初始化后的分割模型，如果指定则加载权重。

        Examples:
            >>> trainer = SegmentationTrainer()
            >>> model = trainer.get_model(cfg="yolo26n-seg.yaml")
            >>> model = trainer.get_model(weights="yolo26n-seg.pt", verbose=False)
        """
        model = SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 SegmentationValidator 实例。"""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
