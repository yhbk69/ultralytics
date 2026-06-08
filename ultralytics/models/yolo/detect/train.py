# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """用于基于检测模型进行训练的类，继承自 BaseTrainer。

    该训练器专注于目标检测任务，处理训练 YOLO 目标检测模型的
    特定需求，包括数据集构建、数据加载、预处理和模型配置。

    Attributes:
        model (DetectionModel): 正在训练的 YOLO 检测模型。
        data (dict): 包含数据集信息的字典，包括类别名称和类别数量。
        loss_names (tuple): 训练中使用的损失分量名称 (box_loss、cls_loss、dfl_loss)。

    Methods:
        build_dataset: 为训练或验证构建 YOLO 数据集。
        get_dataloader: 构建并返回指定模式的数据加载器。
        preprocess_batch: 通过缩放和转换为浮点数来预处理图像批次。
        set_model_attributes: 根据数据集信息设置模型属性。
        get_model: 返回 YOLO 检测模型。
        get_validator: 返回用于模型评估的验证器。
        label_loss_items: 返回带有已标记训练损失项的损失字典。
        progress_string: 返回训练进度的格式化字符串。
        plot_training_samples: 绘制训练样本及其标注。
        plot_training_labels: 创建 YOLO 模型的带标签训练图。
        auto_batch: 根据模型内存需求计算最优批次大小。

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """初始化用于训练 YOLO 目标检测模型的 DetectionTrainer 对象。

        Args:
            cfg (dict, optional): 包含训练参数的默认配置字典。
            overrides (dict, optional): 覆盖默认配置的参数字典。
            _callbacks (dict, optional): 训练期间执行的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """为训练或验证构建 YOLO 数据集。

        Args:
            img_path (str): 包含图像的文件夹路径。
            mode (str): 'train' 模式或 'val' 模式，用户可以为每种模式自定义不同的数据增强。
            batch (int, optional): 批次大小，用于 'rect' 模式。

        Returns:
            (Dataset): 为指定模式配置的 YOLO 数据集对象。
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """构建并返回指定模式的数据加载器。

        Args:
            dataset_path (str): 数据集路径。
            batch_size (int): 每批次的图像数量。
            rank (int): 分布式训练的进程排名。
            mode (str): 'train' 为训练数据加载器，'val' 为验证数据加载器。

        Returns:
            (DataLoader): PyTorch 数据加载器对象。
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # 在 DDP 下仅初始化一次数据集 *.cache
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """通过缩放和转换为浮点数来预处理图像批次。

        Args:
            batch (dict): 包含 'img' 张量的批次数据字典。

        Returns:
            (dict): 包含归一化图像的处理后批次。
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),  # 最小 imgsz
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),  # 最大 imgsz
                )
                // self.stride
                * self.stride
            )  # 尺寸
            sf = sz / max(imgs.shape[2:])  # 缩放因子
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # 新形状（拉伸为 gs 的倍数）
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """根据数据集信息设置模型属性。"""
        # Nl = de_parallel(self.model).model[-1].nl  # 检测层数（用于缩放超参数）
        # self.args.box *= 3 / nl  # 按层缩放
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # 按类别和层缩放
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # 按图像尺寸和层缩放
        self.model.nc = self.data["nc"]  # 将类别数量附加到模型
        self.model.names = self.data["names"]  # 将类别名称附加到模型
        self.model.args = self.args  # 将超参数附加到模型
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self):
        """计算并设置类别权重以处理类别不平衡。

        类别权重基于训练数据集中逆类别频率计算，
        并取 cls_pw 的幂次（0 < cls_pw <= 1 抑制，cls_pw > 1 放大）。
        最终权重经过归一化，使其均值等于 1.0。
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        class_counts = np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)

        weights = (1.0 / class_counts) ** self.args.cls_pw  # apply power directly
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """返回 YOLO 检测模型。

        Args:
            cfg (str, optional): 模型配置文件的路径。
            weights (str, optional): 模型权重的路径。
            verbose (bool): 是否显示模型信息。

        Returns:
            (DetectionModel): YOLO 检测模型。
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 DetectionValidator。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """返回带有已标记训练损失项张量的损失字典。

        Args:
            loss_items (list[float], optional): 损失值列表。
            prefix (str): 返回字典中键的前缀。

        Returns:
            (dict | list): 如果提供了 loss_items 则返回已标记损失项的字典，否则返回键的列表。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # 将张量转换为保留 5 位小数的浮点数
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """返回包含 epoch、GPU 内存、损失、实例数和尺寸的训练进度格式化字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制训练样本及其标注。

        Args:
            batch (dict[str, Any]): 包含批次数据的字典。
            ni (int): 用于命名输出文件的批次索引。
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """创建 YOLO 模型的带标签训练图。"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """通过计算模型内存占用来获得最优批次大小。

        Returns:
            (int): 最优批次大小。
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 马赛克增强需要4倍
        n = len(train_dataset)
        del train_dataset  # 释放内存
        return super().auto_batch(max_num_obj, dataset_size=n)
