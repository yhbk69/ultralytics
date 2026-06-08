# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy, deepcopy

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import YOLOESegModel
from ultralytics.utils import RANK

from .train import YOLOETrainer, YOLOETrainerFromScratch, YOLOEVPTrainer
from .val import YOLOESegValidator


class YOLOESegTrainer(YOLOETrainer, SegmentationTrainer):
    """用于 YOLOE 分割模型的训练器类。

    该类结合 YOLOETrainer 和 SegmentationTrainer，专门为 YOLOE 分割模型
    提供训练功能，同时支持目标检测和实例分割能力。

    Attributes:
        cfg (dict): 训练参数字典。
        overrides (dict): 参数字典覆盖项。
        _callbacks (dict): 训练事件的回调函数字典。
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回使用指定配置和权重初始化的 YOLOESegModel。

        Args:
            cfg (dict | str, optional): 模型配置字典或 YAML 文件路径。
            weights (str, optional): 预训练权重文件路径。
            verbose (bool): 是否显示模型信息。

        Returns:
            (YOLOESegModel): 初始化后的 YOLOE 分割模型。
        """
        # 注意：此处的 `nc` 是单张图像中不同文本样本的最大数量，而非实际的 `nc`。
        # 注意：遵循官方配置，nc 当前硬编码为 80。
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """创建并返回用于 YOLOE 分割模型评估的验证器。

        Returns:
            (YOLOESegValidator): YOLOE 分割模型的验证器。
        """
        self.loss_names = "box", "seg", "cls", "dfl"
        return YOLOESegValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class YOLOEPESegTrainer(SegmentationTrainer):
    """使用线性探测方式微调 YOLOESeg 模型。

    该训练器专门使用线性探测方法微调 YOLOESeg 模型，
    通过冻结模型的大部分层并仅训练特定层，以高效适应新任务。

    Attributes:
        data (dict): 包含通道数、类别名称和类别数量的数据集配置。
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回使用指定配置和权重初始化的 YOLOESegModel，用于线性探测。

        Args:
            cfg (dict | str, optional): 模型配置字典或 YAML 文件路径。
            weights (str, optional): 预训练权重文件路径。
            verbose (bool): 是否显示模型信息。

        Returns:
            (YOLOESegModel): 初始化好的 YOLOE 分割模型，配置为线性探测模式。
        """
        # 注意：此处的 `nc` 是单张图像中不同文本样本的最大数量，而非实际的 `nc`。
        # 注意：遵循官方配置，nc 当前硬编码为 80。
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )

        del model.model[-1].savpe

        assert weights is not None, "Pretrained weights must be provided for linear probing."
        if weights:
            model.load(weights)

        model.eval()
        names = list(self.data["names"].values())
        # 注意：`get_text_pe` 与文本模型和 YOLOEDetect.reprta 相关，
        # 只要加载了正确的预训练权重，就能获得正确的结果。
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

        if getattr(model.model[-1], "one2one_cv3", None) is not None:
            model.model[-1].one2one_cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

        model.train()

        return model


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    """从头训练不使用预训练权重的 YOLOE 分割模型训练器。"""

    pass


class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    """用于支持视觉提示 (VP) 能力的 YOLOE 分割模型训练器。"""

    pass
