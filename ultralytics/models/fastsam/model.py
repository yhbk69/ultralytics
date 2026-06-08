# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):
    """用于分割一切任务的 FastSAM 模型接口。

    该类继承自基础 Model 类，为 FastSAM（快速分割一切模型）实现提供特定功能，支持高效准确的分割以及可选的提示输入。

    Attributes:
        model (str): 预训练 FastSAM 模型文件的路径。
        task (str): 任务类型，对于 FastSAM 模型固定为 "segment"。

    Methods:
        predict: 对图像或视频源执行分割预测，支持可选的提示输入。
        task_map: 返回将分割任务映射到预测器和验证器类的字典。

    Examples:
        初始化 FastSAM 模型并运行预测
        >>> from ultralytics import FastSAM
        >>> model = FastSAM("FastSAM-x.pt")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

        使用边界框提示运行预测
        >>> results = model.predict("image.jpg", bboxes=[[100, 100, 200, 200]])
    """

    def __init__(self, model: str | Path = "FastSAM-x.pt"):
        """使用指定的预训练权重初始化 FastSAM 模型。"""
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM only supports pre-trained weights."
        super().__init__(model=model, task="segment")

    def predict(
        self,
        source,
        stream: bool = False,
        bboxes: list | None = None,
        points: list | None = None,
        labels: list | None = None,
        texts: list | None = None,
        **kwargs: Any,
    ):
        """对图像或视频源执行分割预测。

        支持使用边界框、点、标签和文本进行提示式分割。该方法将这些提示打包后传递给父类的 predict 方法进行处理。

        Args:
            source (str | PIL.Image | np.ndarray): 预测的输入源，可以是文件路径、URL、PIL 图像或 numpy 数组。
            stream (bool): 是否对视频输入启用实时流模式。
            bboxes (list, optional): 提示式分割的边界框坐标，格式为 [[x1, y1, x2, y2]]。
            points (list, optional): 提示式分割的点坐标，格式为 [[x, y]]。
            labels (list, optional): 提示式分割的类别标签。
            texts (list, optional): 用于分割引导的文本提示。
            **kwargs (Any): 传递给预测器的额外关键字参数。

        Returns:
            (list): 包含预测结果的 Results 对象列表。
        """
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """返回一个字典，将分割任务映射到相应的预测器和验证器类。"""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
