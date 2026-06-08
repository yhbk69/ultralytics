# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class OBBPredictor(DetectionPredictor):
    """基于旋转边界框 (OBB) 模型进行预测的类，继承自 DetectionPredictor。

    该预测器处理旋转边界框检测任务，处理图像并返回带有旋转
    边界框的结果。

    Attributes:
        args (namespace): 预测器的配置参数。
        model (torch.nn.Module): 已加载的 YOLO OBB 模型。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.obb import OBBPredictor
        >>> args = dict(model="yolo26n-obb.pt", source=ASSETS)
        >>> predictor = OBBPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用可选的模型和数据配置覆盖项初始化 OBBPredictor。

        Args:
            cfg (dict, optional): 预测器的默认配置。
            overrides (dict, optional): 优先级高于默认配置的配置覆盖项。
            _callbacks (dict, optional): 预测期间调用的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def construct_result(self, pred, img, orig_img, img_path):
        """从预测结果构建 Results 对象。

        Args:
            pred (torch.Tensor): 预测的边界框、分数和旋转角度，形状为 (N, 7)，其中
                最后一维包含 [x, y, w, h, confidence, class_id, angle]。
            img (torch.Tensor): 预处理后的图像，形状为 (B, C, H, W)。
            orig_img (np.ndarray): 预处理前的原始图像。
            img_path (str): 原始图像的路径。

        Returns:
            (Results): 包含原始图像、图像路径、类别名称和旋转边界框的
                Results 对象。
        """
        rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
        rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
        obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
        return Results(orig_img, path=img_path, names=self.model.names, obb=obb)
