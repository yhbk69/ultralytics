# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import cv2
import torch
from PIL import Image

from ultralytics.data.augment import classify_transforms
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class ClassificationPredictor(BasePredictor):
    """基于分类模型进行预测的类，继承自 BasePredictor 类。

    该预测器处理分类模型的特定需求，包括图像预处理和
    预测结果的后处理以生成分类结果。

    Attributes:
        args (dict): 预测器的配置参数。

    Methods:
        preprocess: 将输入图像转换为与模型兼容的格式。
        postprocess: 将模型预测结果处理为 Results 对象。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.classify import ClassificationPredictor
        >>> args = dict(model="yolo26n-cls.pt", source=ASSETS)
        >>> predictor = ClassificationPredictor(overrides=args)
        >>> predictor.predict_cli()

    Notes:
        - Torchvision 分类模型也可以传递给 'model' 参数，例如 model='resnet18'。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用指定配置初始化 ClassificationPredictor，并将任务设置为 'classify'。

        该构造函数初始化 ClassificationPredictor 实例，它继承自 BasePredictor 用于分类
        任务。无论输入配置如何，它都会确保任务设置为 'classify'。

        Args:
            cfg (dict): 包含预测设置的默认配置字典。
            overrides (dict, optional): 优先级高于 cfg 的配置覆盖项。
            _callbacks (dict, optional): 预测期间执行的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"

    def setup_source(self, source):
        """设置数据源、推理模式和分类变换。"""
        super().setup_source(source)
        updated = (
            self.model.model.transforms.transforms[0].size != max(self.imgsz)
            if hasattr(self.model.model, "transforms") and hasattr(self.model.model.transforms.transforms[0], "size")
            else False
        )
        self.transforms = (
            classify_transforms(self.imgsz) if updated or self.model.format != "pt" else self.model.model.transforms
        )

    def preprocess(self, img):
        """将输入图像转换为与模型兼容的张量格式，并进行适当的归一化。"""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # 将 uint8 转换为 fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """处理预测结果，返回包含分类概率的 Results 对象。

        Args:
            preds (torch.Tensor): 模型的原始预测结果。
            img (torch.Tensor): 预处理后的输入图像。
            orig_imgs (list[np.ndarray] | torch.Tensor): 预处理前的原始图像。

        Returns:
            (list[Results]): 包含每张图像分类结果的 Results 对象列表。
        """
        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
