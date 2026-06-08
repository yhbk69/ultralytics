# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
SAM模型接口。

此模块提供Ultralytics的Segment Anything Model（SAM）接口，专为实时图像分割任务设计。
SAM模型支持可提示分割，在图像分析中具有无与伦比的通用性，并在SA-1B数据集上训练。
它具有零样本性能能力，使其能够在没有先验知识的情况下适应新的图像分布和任务。

主要特性：
    - 可提示分割
    - 实时性能
    - 零样本迁移能力
    - 基于SA-1B数据集训练
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info

from .predict import Predictor, SAM2Predictor, SAM3Predictor


class SAM(Model):
    """SAM（Segment Anything Model）接口类，用于实时图像分割任务。

    此类提供Ultralytics的Segment Anything Model（SAM）接口，专为可提示分割设计，在图像分析中具有高度通用性。
    它支持各种提示，如边界框、点或标签，并具有零样本性能能力。

    Attributes:
        model (torch.nn.Module): 加载的SAM模型。
        is_sam2 (bool): 指示模型是否为SAM2变体。
        task (str): 任务类型，SAM模型设为"segment"。

    Methods:
        predict: 对给定的图像或视频源执行分割预测。
        info: 记录SAM模型的信息。

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        ...     print(f"检测到 {len(r.masks)} 个掩码")
    """

    def __init__(self, model: str = "sam_b.pt") -> None:
        """初始化SAM（Segment Anything Model）实例。

        Args:
            model (str): 预训练SAM模型文件的路径。文件应为.pt或.pth扩展名。

        Raises:
            NotImplementedError: 如果模型文件扩展名不是.pt或.pth。
        """
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        self.is_sam2 = "sam2" in Path(model).stem
        self.is_sam3 = "sam3" in Path(model).stem
        super().__init__(model=model, task="segment")

    def _load(self, weights: str, task=None):
        """将指定的权重加载到SAM模型中。

        Args:
            weights (str): 权重文件的路径。应为包含模型参数的.pt或.pth文件。
            task (str | None): 任务名称。如果提供，指定模型加载的特定任务。

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> sam._load("path/to/custom_weights.pt")
        """
        if self.is_sam3:
            from .build_sam3 import build_interactive_sam3

            self.model = build_interactive_sam3(weights)
        else:
            from .build import build_sam  # slow import

            self.model = build_sam(weights)

    def predict(self, source, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
        """对给定的图像或视频源执行分割预测。

        Args:
            source (str | PIL.Image | np.ndarray): 图像或视频文件的路径，或PIL.Image对象，或np.ndarray对象。
            stream (bool): 如果为True，启用实时流式处理。
            bboxes (list[list[float]] | None): 用于提示分割的边界框坐标列表。
            points (list[list[float]] | None): 用于提示分割的点坐标列表。
            labels (list[int] | None): 用于提示分割的标签列表。
            **kwargs (Any): 预测的额外关键字参数。

        Returns:
            (list): 模型预测结果。

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam.predict("image.jpg", points=[[500, 375]])
            >>> for r in results:
            ...     print(f"检测到 {len(r.masks)} 个掩码")
        """
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
        kwargs = {**overrides, **kwargs}
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
        """对给定的图像或视频源执行分割预测。

        此方法是'predict'方法的别名，为调用SAM模型进行分割任务提供了便捷方式。

        Args:
            source (str | PIL.Image | np.ndarray | None): 图像或视频文件的路径，或PIL.Image对象，或np.ndarray对象。
            stream (bool): 如果为True，启用实时流式处理。
            bboxes (list[list[float]] | None): 用于提示分割的边界框坐标列表。
            points (list[list[float]] | None): 用于提示分割的点坐标列表。
            labels (list[int] | None): 用于提示分割的标签列表。
            **kwargs (Any): 传递给predict方法的额外关键字参数。

        Returns:
            (list): 模型预测结果，通常包含分割掩码和其他相关信息。

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam("image.jpg", points=[[500, 375]])
            >>> print(f"检测到 {len(results[0].masks)} 个掩码")
        """
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed: bool = False, verbose: bool = True):
        """记录SAM模型的信息。

        Args:
            detailed (bool): 如果为True，显示模型层和操作的详细信息。
            verbose (bool): 如果为True，将信息打印到控制台。

        Returns:
            (tuple): 包含模型信息（模型的字符串表示）的元组。

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> info = sam.info()
            >>> print(info[0])  # 打印摘要信息
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, type[Predictor]]]:
        """提供从'segment'任务到其对应'Predictor'的映射。

        Returns:
            (dict[str, dict[str, type[Predictor]]]): 将'segment'任务映射到其对应Predictor类的字典。
                对于SAM2模型，映射到SAM2Predictor，否则映射到标准Predictor。

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> task_map = sam.task_map
            >>> print(task_map)
            {'segment': {'predictor': <class 'ultralytics.models.sam.predict.Predictor'>}}
        """
        return {
            "segment": {"predictor": SAM2Predictor if self.is_sam2 else SAM3Predictor if self.is_sam3 else Predictor}
        }
