# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
百度 RT-DETR 的接口，一个基于 Vision Transformer 的实时目标检测器。

RT-DETR 提供实时性能和高准确度，在 CUDA 配合 TensorRT 等加速后端中表现卓越。
它采用高效的混合编码器和基于 IoU 的查询选择来提升检测精度。

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """百度 RT-DETR 模型的接口，一个基于 Vision Transformer 的实时目标检测器。

    该模型提供实时性能和高准确度。它支持高效的混合编码、基于 IoU 的查询选择和可调节的推理速度。

    Attributes:
        model (str): 预训练模型的路径。

    Methods:
        task_map: 返回 RT-DETR 的任务映射，将任务与相应的 Ultralytics 类关联起来。

    Examples:
        使用预训练模型初始化 RT-DETR
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt") -> None:
        """使用给定的预训练模型文件初始化 RT-DETR 模型。

        Args:
            model (str): 预训练模型的路径。支持 .pt、.yaml 和 .yml 格式。
        """
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """返回 RT-DETR 的任务映射，将任务与相应的 Ultralytics 类关联起来。

        Returns:
            (dict): 一个将任务名称映射到 RT-DETR 模型对应 Ultralytics 任务类的字典。
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }
