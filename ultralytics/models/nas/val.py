# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

__all__ = ["NASValidator"]


class NASValidator(DetectionValidator):
    """用于目标检测的 Ultralytics YOLO NAS 验证器。

    继承自 Ultralytics 模型包中的 DetectionValidator，专门用于后处理 YOLO NAS 模型生成的原始预测结果。
    它执行非极大值抑制来移除重叠和低置信度的边界框，最终生成检测结果。

    Attributes:
        args (Namespace): 包含后处理各种配置的命名空间，如置信度和 IoU 阈值。
        lb (torch.Tensor): 用于多标签 NMS 的可选张量。

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> validator = model.validator
        >>> # 假设 raw_preds 可用
        >>> final_preds = validator.postprocess(raw_preds)

    Notes:
        通常不直接实例化该类，它在 NAS 类内部使用。
    """

    def postprocess(self, preds_in):
        """对预测输出应用非极大值抑制。"""
        boxes = ops.xyxy2xywh(preds_in[0][0])  # 将边界框格式从 xyxy 转换为 xywh
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)  # 将边界框与分数拼接并进行维度置换
        return super().postprocess(preds)
