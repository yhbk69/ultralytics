# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class PosePredictor(DetectionPredictor):
    """基于姿态模型进行预测的类，继承自 DetectionPredictor。

    该类专注于姿态估计，在继承 DetectionPredictor 的标准目标检测能力
    基础上，处理关键点检测。

    Attributes:
        args (namespace): 预测器的配置参数。
        model (torch.nn.Module): 已加载的具有关键点检测能力的 YOLO 姿态模型。

    Methods:
        construct_result: 从预测结果构建包含关键点的 Results 对象。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo26n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """初始化用于姿态估计任务的 PosePredictor。

        设置 PosePredictor 实例，配置用于姿态检测任务，并处理 Apple MPS
        的设备特定警告。

        Args:
            cfg (Any): 预测器的配置。
            overrides (dict, optional): 优先级高于 cfg 的配置覆盖项。
            _callbacks (dict, optional): 预测期间调用的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"

    def construct_result(self, pred, img, orig_img, img_path):
        """从预测结果构建包含关键点的 Results 对象。

        扩展父类实现，从预测中提取关键点数据并将其添加到
        Results 对象中。

        Args:
            pred (torch.Tensor): 预测的边界框、分数和关键点，形状为 (N, 6+K*D)，其中 N 是
                检测数量，K 是关键点数量，D 是关键点维度。
            img (torch.Tensor): 处理后的输入图像张量，形状为 (B, C, H, W)。
            orig_img (np.ndarray): 原始未处理图像的 numpy 数组。
            img_path (str): 原始图像文件的路径。

        Returns:
            (Results): 包含原始图像、图像路径、类别名称、边界框和
                关键点的 Results 对象。
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # 从预测中提取关键点并根据模型的关键点形状重塑
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        # 缩放关键点坐标以匹配原始图像尺寸
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result
