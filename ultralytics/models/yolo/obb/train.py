# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK


class OBBTrainer(yolo.detect.DetectionTrainer):
    """用于基于旋转边界框 (OBB) 模型进行训练的类，继承自 DetectionTrainer。

    该训练器专注于训练检测旋转边界框的 YOLO 模型，适用于检测
    任意角度的目标，而非仅限轴对齐矩形。

    Attributes:
        loss_names (tuple): 训练期间使用的损失分量名称，包括 box_loss、cls_loss、dfl_loss
            和 angle_loss。

    Methods:
        get_model: 返回使用指定配置和权重初始化的 OBBModel。
        get_validator: 返回用于 YOLO 模型验证的 OBBValidator 实例。

    Examples:
        >>> from ultralytics.models.yolo.obb import OBBTrainer
        >>> args = dict(model="yolo26n-obb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = OBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """初始化用于训练旋转边界框 (OBB) 模型的 OBBTrainer 对象。

        Args:
            cfg (dict, optional): 训练器的配置字典。包含训练参数和模型配置。
            overrides (dict, optional): 配置的参数字典覆盖项。此处的任何值将
                覆盖 cfg 中的值。
            _callbacks (dict, optional): 训练期间调用的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> OBBModel:
        """返回使用指定配置和权重初始化的 OBBModel。

        Args:
            cfg (str | dict, optional): 模型配置。可以是 YAML 配置文件的路径、包含配置参数的字典，
                或为 None 使用默认配置。
            weights (str | Path, optional): 预训练权重文件的路径。如果为 None，则使用随机初始化。
            verbose (bool): 是否在初始化期间显示模型信息。

        Returns:
            (OBBModel): 使用指定配置和权重初始化的 OBBModel。

        Examples:
            >>> trainer = OBBTrainer()
            >>> model = trainer.get_model(cfg="yolo26n-obb.yaml", weights="yolo26n-obb.pt")
        """
        model = OBBModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 OBBValidator 实例。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "angle_loss"
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
