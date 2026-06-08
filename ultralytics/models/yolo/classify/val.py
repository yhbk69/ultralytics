# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):
    """用于基于分类模型进行验证的类，继承自 BaseValidator。

    该验证器处理分类模型的验证过程，包括指标计算、混淆矩阵
    生成和结果可视化。

    Attributes:
        targets (list[torch.Tensor]): 真实类别标签。
        pred (list[torch.Tensor]): 模型预测结果。
        metrics (ClassifyMetrics): 用于计算和存储分类指标的对象。
        names (dict): 类别索引到类别名称的映射。
        nc (int): 类别数量。
        confusion_matrix (ConfusionMatrix): 评估模型跨类别性能的矩阵。

    Methods:
        get_desc: 返回汇总分类指标的格式化字符串。
        init_metrics: 初始化混淆矩阵、类别名称和跟踪容器。
        preprocess: 通过将数据移到设备上来预处理输入批次。
        update_metrics: 使用模型预测和批次目标更新运行指标。
        finalize_metrics: 完成指标计算，包括混淆矩阵和处理速度。
        postprocess: 从模型输出中提取主要预测结果。
        get_stats: 计算并返回指标字典。
        build_dataset: 为验证创建 ClassificationDataset 实例。
        get_dataloader: 构建并返回分类验证的数据加载器。
        print_results: 打印分类模型的评估指标。
        plot_val_samples: 绘制验证图像样本及其真实标签。
        plot_predictions: 绘制图像及其预测类别标签。

    Examples:
        >>> from ultralytics.models.yolo.classify import ClassificationValidator
        >>> args = dict(model="yolo26n-cls.pt", data="imagenet10")
        >>> validator = ClassificationValidator(args=args)
        >>> validator()

    Notes:
        Torchvision 分类模型也可以传递给 'model' 参数，例如 model='resnet18'。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """使用数据加载器、保存目录和其他参数初始化 ClassificationValidator。

        Args:
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的数据加载器。
            save_dir (str | Path, optional): 保存结果的目录。
            args (dict, optional): 包含模型和验证配置的参数。
            _callbacks (dict, optional): 验证期间调用的回调函数字典。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self) -> str:
        """返回汇总分类指标的格式化字符串。"""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """初始化混淆矩阵、类别名称以及预测和目标的跟踪容器。"""
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []
        self.confusion_matrix = ConfusionMatrix(names=model.names)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """将输入批次的数据移到设备上并转换为适当的数据类型，完成预处理。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """使用模型预测和批次目标更新运行指标。

        Args:
            preds (torch.Tensor): 模型预测结果，通常为每个类别的 logits 或概率。
            batch (dict): 包含图像和类别标签的批次数据。

        Notes:
            该方法将 top-N 预测结果（按置信度降序排列）追加到预测列表中
            以供后续评估。N 限制为 5 和类别数量的最小值。
        """
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self) -> None:
        """完成指标计算，包括混淆矩阵和处理速度。

        Examples:
            >>> validator = ClassificationValidator()
            >>> validator.pred = [torch.tensor([[0, 1, 2]])]  # 一个样本的 Top-3 预测
            >>> validator.targets = [torch.tensor([0])]  # 真实类别
            >>> validator.finalize_metrics()
            >>> print(validator.metrics.confusion_matrix)  # 访问混淆矩阵

        Notes:
            该方法处理累积的预测和目标以生成混淆矩阵，
            可选地绘制图表，并用速度信息更新指标对象。
        """
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.metrics.confusion_matrix = self.confusion_matrix

    def postprocess(self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]) -> torch.Tensor:
        """如果模型输出是列表或元组格式，则提取主要预测结果。"""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self) -> dict[str, float]:
        """通过处理目标和预测来计算并返回指标字典。"""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def gather_stats(self) -> None:
        """从所有 GPU 收集统计信息。"""
        if RANK == 0:
            gathered_preds = [None] * dist.get_world_size()
            gathered_targets = [None] * dist.get_world_size()
            dist.gather_object(self.pred, gathered_preds, dst=0)
            dist.gather_object(self.targets, gathered_targets, dst=0)
            self.pred = [pred for rank in gathered_preds for pred in rank]
            self.targets = [targets for rank in gathered_targets for targets in rank]
        elif RANK > 0:
            dist.gather_object(self.pred, None, dst=0)
            dist.gather_object(self.targets, None, dst=0)

    def build_dataset(self, img_path: str) -> ClassificationDataset:
        """为验证创建 ClassificationDataset 实例。"""
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path: Path | str, batch_size: int) -> torch.utils.data.DataLoader:
        """构建并返回分类验证的数据加载器。

        Args:
            dataset_path (str | Path): 数据集目录的路径。
            batch_size (int): 每批次的样本数量。

        Returns:
            (torch.utils.data.DataLoader): 分类验证数据集的 DataLoader 对象。
        """
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self) -> None:
        """打印分类模型的评估指标。"""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # 打印格式
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """绘制验证图像样本及其真实标签。

        Args:
            batch (dict[str, Any]): 包含批次数据的字典，包含 'img'（图像）和 'cls'（类别标签）。
            ni (int): 用于命名输出文件的批次索引。

        Examples:
            >>> validator = ClassificationValidator()
            >>> batch = {"img": torch.rand(16, 3, 224, 224), "cls": torch.randint(0, 10, (16,))}
            >>> validator.plot_val_samples(batch, 0)
        """
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # 添加批次索引用于绘图
        plot_images(
            labels=batch,
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """绘制图像及其预测类别标签并保存可视化结果。

        Args:
            batch (dict[str, Any]): 包含图像和其他信息的批次数据。
            preds (torch.Tensor): 模型预测结果，形状为 (batch_size, num_classes)。
            ni (int): 用于命名输出文件的批次索引。

        Examples:
            >>> validator = ClassificationValidator()
            >>> batch = {"img": torch.rand(16, 3, 224, 224)}
            >>> preds = torch.rand(16, 10)  # 16 张图像，10 个类别
            >>> validator.plot_predictions(batch, preds, 0)
        """
        batched_preds = dict(
            img=batch["img"],
            batch_idx=torch.arange(batch["img"].shape[0]),
            cls=torch.argmax(preds, dim=1),
            conf=torch.amax(preds, dim=1),
        )
        plot_images(
            batched_preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 预测
