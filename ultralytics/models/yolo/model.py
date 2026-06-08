# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import ROOT, YAML


class YOLO(Model):
    """YOLO (You Only Look Once) 目标检测模型。

    该类为 YOLO 模型提供统一接口，根据模型文件名自动切换到专用模型类型
    （YOLOWorld 或 YOLOE）。支持多种计算机视觉任务，包括目标检测、
    分割、分类、姿态估计和旋转边界框检测。

    Attributes:
        model: 已加载的 YOLO 模型实例。
        task: 任务类型 (detect、segment、classify、pose、obb)。
        overrides: 模型的配置覆盖项。

    Methods:
        __init__: 初始化 YOLO 模型，自动检测类型。
        task_map: 将任务映射到对应的模型、训练器、验证器和预测器类。

    Examples:
        加载预训练的 YOLO26n 检测模型
        >>> model = YOLO("yolo26n.pt")

        加载预训练的 YOLO26n 分割模型
        >>> model = YOLO("yolo26n-seg.pt")

        从 YAML 配置初始化
        >>> model = YOLO("yolo26n.yaml")
    """

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        """初始化 YOLO 模型。

        该构造函数初始化 YOLO 模型，根据模型文件名自动切换到专用模型类型（YOLOWorld 或
        YOLOE）。

        Args:
            model (str | Path): 模型名称或模型文件路径，例如 'yolo26n.pt'、'yolo26n.yaml'。
            task (str, optional): YOLO 任务类型，例如 'detect'、'segment'、'classify'、'pose'、'obb'。
                默认为根据模型自动检测。
            verbose (bool): 加载时显示模型信息。
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # 如果是 YOLOWorld PyTorch 模型
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # 如果是 YOLOE PyTorch 模型
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # 继续使用默认 YOLO 初始化
            super().__init__(model=model, task=task, verbose=verbose)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # 如果是 RTDETR 检测头
                from ultralytics import RTDETR

                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """将检测头映射到模型、训练器、验证器和预测器类。"""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World 目标检测模型。

    YOLO-World 是一种开放词汇目标检测模型，可以基于文本描述检测目标，
    无需在特定类别上训练。它扩展了 YOLO 架构以支持实时开放词汇检测。

    Attributes:
        model: 已加载的 YOLO-World 模型实例。
        task: 始终设置为 'detect' 以进行目标检测。
        overrides: 模型的配置覆盖项。

    Methods:
        __init__: 使用预训练模型文件初始化 YOLOv8-World 模型。
        task_map: 将任务映射到对应的模型、训练器、验证器和预测器类。
        set_classes: 设置模型用于检测的类别名称。

    Examples:
        加载 YOLOv8-World 模型
        >>> model = YOLOWorld("yolov8s-world.pt")

        设置用于检测的自定义类别
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None:
        """使用预训练模型文件初始化 YOLOv8-World 模型。

        加载 YOLOv8-World 模型用于目标检测。如果未提供自定义类别名称，则分配默认的 COCO
        类别名称。

        Args:
            model (str | Path): 预训练模型文件的路径。支持 *.pt 和 *.yaml 格式。
            verbose (bool): 如果为 True，初始化时打印额外信息。
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # 当没有自定义名称时分配默认 COCO 类别名称
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """将检测头映射到模型、训练器、验证器和预测器类。"""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes: list[str]) -> None:
        """设置模型用于检测的类别名称。

        Args:
            classes (list[str]): 类别列表，例如 ["person"]。
        """
        self.model.set_classes(classes)
        # 如果提供了背景类别则移除
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # 重置方法的类别名称
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """YOLOE 目标检测与分割模型。

    YOLOE 是增强版 YOLO 模型，支持目标检测和实例分割任务，具有改进的性能以及
    视觉和文本位置嵌入等附加功能。

    Attributes:
        model: 已加载的 YOLOE 模型实例。
        task: 任务类型 (detect 或 segment)。
        overrides: 模型的配置覆盖项。

    Methods:
        __init__: 使用预训练模型文件初始化 YOLOE 模型。
        task_map: 将任务映射到对应的模型、训练器、验证器和预测器类。
        get_text_pe: 获取给定文本的文本位置嵌入。
        get_visual_pe: 获取给定图像和视觉特征的视觉位置嵌入。
        set_vocab: 设置 YOLOE 模型的词汇表和类别名称。
        get_vocab: 获取给定类别名称的词汇表。
        set_classes: 设置模型用于检测的类别名称和嵌入。
        val: 使用文本或视觉提示验证模型。
        predict: 对图像、视频、目录、流等运行预测。

    Examples:
        加载 YOLOE 检测模型
        >>> model = YOLOE("yoloe-11s-seg.pt")

        设置词汇表和类别名称
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

        使用视觉提示进行预测
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("image.jpg", visual_prompts=prompts)
    """

    def __init__(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        """使用预训练模型文件初始化 YOLOE 模型。

        Args:
            model (str | Path): 预训练模型文件的路径。支持 *.pt 和 *.yaml 格式。
            task (str, optional): 模型的任务类型。如果为 None 则自动检测。
            verbose (bool): 如果为 True，初始化时打印额外信息。
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """将检测头映射到模型、训练器、验证器和预测器类。"""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """获取给定文本的文本位置嵌入。"""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """获取给定图像和视觉特征的视觉位置嵌入。

        该方法根据输入图像从视觉特征中提取位置嵌入。要求模型
        是 YOLOEModel 的实例。

        Args:
            img (torch.Tensor): 输入图像张量。
            visual (torch.Tensor): 从图像中提取的视觉特征。

        Returns:
            (torch.Tensor): 视觉位置嵌入。

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = torch.rand(1, 1, 80, 80)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: list[str], names: list[str]) -> None:
        """设置 YOLOE 模型的词汇表和类别名称。

        该方法配置模型用于文本处理和分类任务的词汇表和类别名称。
        模型必须是 YOLOEModel 的实例。

        Args:
            vocab (list[str]): 包含模型用于文本处理的词元或单词的词汇表。
            names (list[str]): 模型可检测或分类的类别名称列表。

        Raises:
            AssertionError: 如果模型不是 YOLOEModel 的实例。

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """获取给定类别名称的词汇表。"""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        """设置模型用于检测的类别名称和嵌入。

        Args:
            classes (list[str]): 类别列表，例如 ["person"]。
            embeddings (torch.Tensor, optional): 与类别对应的嵌入。
        """
        # 验证没有背景类别
        assert " " not in classes
        assert isinstance(self.model, YOLOEModel)
        if sorted(list(self.model.names.values())) != sorted(classes):
            if embeddings is None:
                embeddings = self.get_text_pe(classes)  # 如果未提供嵌入则生成文本嵌入
            self.model.set_classes(classes, embeddings)

        # 重置方法的类别名称
        if self.predictor:
            self.predictor.model.names = self.model.names

    def val(
        self,
        validator=None,
        load_vp: bool = False,
        refer_data: str | None = None,
        **kwargs,
    ):
        """使用文本或视觉提示验证模型。

        Args:
            validator (callable, optional): 可调用的验证器函数。如果为 None，则加载默认验证器。
            load_vp (bool): 是否加载视觉提示。如果为 False，则使用文本提示。
            refer_data (str, optional): 视觉提示的参考数据路径。
            **kwargs (Any): 用于覆盖默认设置的额外关键字参数。

        Returns:
            (dict): 包含验证期间计算的指标的验证统计信息。
        """
        custom = {"rect": not load_vp}  # 方法默认值
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # 最高优先级的参数在右侧

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict[str, list] = {},
        refer_image=None,
        predictor=yolo.yoloe.YOLOEVPDetectPredictor,
        **kwargs,
    ):
        """对图像、视频、目录、流等运行预测。

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): 预测源。接受图像路径、目录
                路径、URL/YouTube 流、PIL 图像、numpy 数组或摄像头索引。
            stream (bool): 是否流式传输预测结果。如果为 True，结果将在计算时以生成器形式
                逐步产出。
            visual_prompts (dict[str, list]): 包含模型视觉提示的字典。非空时必须包含 'bboxes'
                和 'cls' 键。
            refer_image (str | PIL.Image | np.ndarray, optional): 视觉提示的参考图像。
            predictor (callable): 用于视觉提示预测的自定义预测器类。
                默认为 YOLOEVPDetectPredictor。
            **kwargs (Any): 传递给预测器的额外关键字参数。

        Returns:
            (list | generator): Results 对象列表，如果 stream=True 则为 Results 对象生成器。

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # 使用视觉提示
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
            if type(self.predictor) is not predictor:
                self.predictor = predictor(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                        "device": kwargs.get("device", None),
                        "half": kwargs.get("half", False),
                        "imgsz": kwargs.get("imgsz", self.overrides.get("imgsz", 640)),
                    },
                    _callbacks=self.callbacks,
                )

            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # 表示多张图像
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    # 注意：将视频/流的第一帧设为参考图像
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
                self.predictor = None  # 重置预测器
        elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
            self.predictor = None  # 如果没有视觉提示则重置预测器
        self.overrides["agnostic_nms"] = True  # 默认对 YOLOE 使用类别无关 NMS

        return super().predict(source, stream, **kwargs)
