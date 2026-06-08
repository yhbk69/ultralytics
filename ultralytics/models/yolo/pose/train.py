# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model


class PoseTrainer(yolo.detect.DetectionTrainer):
    """用于训练 YOLO 姿态估计模型的类，继承自 DetectionTrainer。

    该训练器专门处理姿态估计任务，管理模型训练、验证以及
    边界框和姿态关键点的可视化。

    Attributes:
        args (dict): 训练的配置参数。
        model (PoseModel): 正在训练的姿态估计模型。
        data (dict): 包含关键点形状信息的数据集配置。
        loss_names (tuple): 训练中使用的损失分量名称。

    Methods:
        get_model: 获取使用指定配置的姿态估计模型。
        set_model_attributes: 在模型上设置关键点形状属性。
        get_validator: 创建用于模型评估的验证器实例。
        plot_training_samples: 可视化包含关键点的训练样本。
        get_dataset: 获取数据集并确保包含必需的 kpt_shape 键。

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """初始化用于训练 YOLO 姿态估计模型的 PoseTrainer 对象。

        Args:
            cfg (dict, optional): 包含训练参数的默认配置字典。
            overrides (dict, optional): 覆盖默认配置的参数字典。
            _callbacks (dict, optional): 训练期间执行的回调函数字典。

        Notes:
            无论 overrides 中提供什么值，该训练器都会自动将任务设置为 'pose'。
            使用 Apple MPS 设备时会发出警告，因为已知姿态模型存在 bug。
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        """获取使用指定配置和权重的姿态估计模型。

        Args:
            cfg (str | Path | dict, optional): 模型配置文件的路径或字典。
            weights (str | Path, optional): 模型权重文件的路径。
            verbose (bool): 是否显示模型信息。

        Returns:
            (PoseModel): 初始化后的姿态估计模型。
        """
        model = PoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """设置 PoseModel 的关键点形状属性。"""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        """返回用于验证的 PoseValidator 类实例。"""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        if getattr(unwrap_model(self.model).model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """获取数据集并确保包含必需的 `kpt_shape` 键。

        Returns:
            (dict): 包含训练/验证/测试数据集和类别名称的字典。

        Raises:
            KeyError: 如果数据集中不存在 `kpt_shape` 键。
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data
