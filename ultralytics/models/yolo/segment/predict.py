# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class SegmentationPredictor(DetectionPredictor):
    """基于分割模型进行预测的类，继承自 DetectionPredictor。

    该类专门处理分割模型的输出，在预测结果中同时处理边界框和掩码。

    Attributes:
        args (dict): 预测器的配置参数。
        model (torch.nn.Module): 已加载的 YOLO 分割模型。
        batch (list): 当前正在处理的图像批次。

    Methods:
        postprocess: 应用非极大值抑制并处理分割检测结果。
        construct_results: 从预测构建 Results 对象列表。
        construct_result: 从单张预测构建单个 Results 对象。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo26n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用配置、覆盖项和回调函数初始化 SegmentationPredictor。

        该类专门处理分割模型的输出，在预测结果中同时处理边界框和掩码。

        Args:
            cfg (dict): 预测器的配置。
            overrides (dict, optional): 优先级高于 cfg 的配置覆盖项。
            _callbacks (dict, optional): 预测期间调用的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """对输入批次中的每张图像应用非极大值抑制并处理分割检测结果。

        Args:
            preds (tuple): 模型预测，包含边界框、分数、类别和掩码系数。
            img (torch.Tensor): 模型格式的输入图像张量，形状为 (B, C, H, W)。
            orig_imgs (list | torch.Tensor | np.ndarray): 原始图像或图像批次。

        Returns:
            (list): 包含批次中每张图像分割预测的 Results 对象列表。
                每个 Results 对象包含边界框和分割掩码。

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo26n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        # 提取原型 - PyTorch 模型则为元组，导出模型则为数组
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """从预测构建 Results 对象列表。

        Args:
            preds (list[torch.Tensor]): 预测的边界框、分数和掩码列表。
            img (torch.Tensor): 预处理后的图像。
            orig_imgs (list[np.ndarray]): 预处理前的原始图像列表。
            protos (torch.Tensor): 原型掩码张量，形状为 (B, C, H, W)。

        Returns:
            (list[Results]): 包含原始图像、图像路径、类别名称、边界框
                和掩码的 Results 对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """从预测构建单个 Results 对象。

        Args:
            pred (torch.Tensor): 预测的边界框、分数和掩码。
            img (torch.Tensor): 预处理后的图像。
            orig_img (np.ndarray): 预处理前的原始图像。
            img_path (str): 原始图像的路径。
            proto (torch.Tensor): 原型掩码。

        Returns:
            (Results): 包含原始图像、图像路径、类别名称、边界框和掩码的 Results 对象。
        """
        if pred.shape[0] == 0:  # 保存空框
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # NHW
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # NHW
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0  # 仅保留有掩码的预测
            if not all(keep):  # 大多数预测都有掩码
                pred, masks = pred[keep], masks[keep]  # 索引操作较慢
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
