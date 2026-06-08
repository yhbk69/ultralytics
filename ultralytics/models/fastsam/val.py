# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.segment import SegmentationValidator


class FastSAMValidator(SegmentationValidator):
    """Ultralytics YOLO 框架中用于 FastSAM（分割一切模型）分割的自定义验证类。

    继承自 SegmentationValidator 类，专门针对 FastSAM 定制验证流程。该类将任务设置为 'segment'，并使用
    SegmentMetrics 进行评估。此外，禁用了绘图功能以避免验证过程中的错误。

    Attributes:
        dataloader (torch.utils.data.DataLoader): 用于验证的数据加载器对象。
        save_dir (Path): 验证结果保存目录。
        args (SimpleNamespace): 用于自定义验证过程的附加参数。
        _callbacks (dict): 验证期间要调用的回调函数字典。
        metrics (SegmentMetrics): 用于评估的分割指标计算器。

    Methods:
        __init__: 使用 FastSAM 的自定义设置初始化 FastSAMValidator。
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        """初始化 FastSAMValidator 类，将任务设置为 'segment'，指标设置为 SegmentMetrics。

        Args:
            dataloader (torch.utils.data.DataLoader, optional): 用于验证的 DataLoader。
            save_dir (Path, optional): 保存结果的目录。
            args (SimpleNamespace, optional): 验证器的配置。
            _callbacks (dict, optional): 验证期间要调用的回调函数字典。

        Notes:
            为避免错误，该类中禁用了 ConfusionMatrix 和其他相关指标的绘图功能。
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  # 禁用 ConfusionMatrix 和其他绘图以避免错误
