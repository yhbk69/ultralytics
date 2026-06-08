# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops


class NASPredictor(DetectionPredictor):
    """用于目标检测的 Ultralytics YOLO NAS 预测器。

    该类继承自 Ultralytics 引擎的 DetectionPredictor，负责对 YOLO NAS 模型生成的原始预测结果进行后处理。
    它应用非极大值抑制和边界框缩放等操作，使结果适应原始图像尺寸。

    Attributes:
        args (Namespace): 包含后处理各种配置的命名空间，包括置信度阈值、IoU 阈值、类别无关 NMS 标志、
            最大检测数和类别过滤选项。
        model (torch.nn.Module): 用于推理的 YOLO NAS 模型。
        batch (list): 待处理的输入批次。

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> predictor = model.predictor

        假设 raw_preds, img, orig_imgs 可用
        >>> results = predictor.postprocess(raw_preds, img, orig_imgs)

    Notes:
        通常不直接实例化该类。它在 NAS 类内部使用。
    """

    def postprocess(self, preds_in, img, orig_imgs):
        """后处理 NAS 模型预测结果以生成最终的检测结果。

        该方法获取 YOLO NAS 模型的原始预测结果，转换边界框格式，并应用后处理操作以生成与 Ultralytics 结果
        可视化和分析工具兼容的最终检测结果。

        Args:
            preds_in (list): NAS 模型的原始预测结果，通常包含边界框和类别分数。
            img (torch.Tensor): 输入模型的图像张量，形状为 (B, C, H, W)。
            orig_imgs (list | torch.Tensor | np.ndarray): 预处理前的原始图像，用于将坐标缩放回原始尺寸。

        Returns:
            (list): 包含批次中每张图片处理后预测结果的 Results 对象列表。

        Examples:
            >>> predictor = NAS("yolo_nas_s").predictor
            >>> results = predictor.postprocess(raw_preds, img, orig_imgs)
        """
        boxes = ops.xyxy2xywh(preds_in[0][0])  # 将边界框格式从 xyxy 转换为 xywh
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)  # 将边界框与类别分数拼接
        return super().postprocess(preds, img, orig_imgs)
