# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class YOLOEDetectValidator(DetectionValidator):
    """用于 YOLOE 检测模型的验证器类，处理文本和视觉提示嵌入。

    该类继承 DetectionValidator，为 YOLOE 模型提供专门的验证功能。
    它支持使用文本提示或从训练样本中提取的视觉提示嵌入进行验证，
    实现基于提示的目标检测灵活评估策略。

    Attributes:
        device (torch.device): 执行验证的设备。
        args (namespace): 验证的配置参数。
        dataloader (DataLoader): 验证数据的数据加载器。

    Methods:
        get_visual_pe: 从训练样本中提取视觉提示嵌入。
        preprocess: 预处理批次数据，确保视觉与图像在同一设备上。
        get_vpe_dataloader: 为 LVIS 训练视觉提示样本创建数据加载器。
        __call__: 使用文本或视觉提示嵌入运行验证。

    Examples:
        使用文本提示进行验证
        >>> validator = YOLOEDetectValidator()
        >>> stats = validator(model=model, load_vp=False)

        使用视觉提示进行验证
        >>> stats = validator(model=model, refer_data="path/to/data.yaml", load_vp=True)
    """

    @smart_inference_mode()
    def get_visual_pe(self, dataloader: torch.utils.data.DataLoader, model: YOLOEModel) -> torch.Tensor:
        """从训练样本中提取视觉提示嵌入。

        该方法处理数据加载器，使用 YOLOE 模型计算每个类别的视觉提示嵌入。
        它对嵌入进行归一化，并将没有样本的类别的嵌入设置为零。

        Args:
            dataloader (torch.utils.data.DataLoader): 提供训练样本的数据加载器。
            model (YOLOEModel): 用于提取视觉提示嵌入的 YOLOE 模型。

        Returns:
            (torch.Tensor): 形状为 (1, num_classes, embed_dim) 的视觉提示嵌入。
        """
        assert isinstance(model, YOLOEModel)
        names = [name.split("/", 1)[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "从样本中获取视觉提示嵌入"

        # 对每个类别计数样本
        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        # 提取视觉提示嵌入
        pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])  # (B, max_n, embed_dim)

            batch_idx = batch["batch_idx"]
            for i in range(preds.shape[0]):
                cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
                pad_cls[: cls.shape[0]] = cls
                for c in cls:
                    visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

        # 对存在样本的类别归一化嵌入，其他设置为零
        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def get_vpe_dataloader(self, data: dict[str, Any]) -> torch.utils.data.DataLoader:
        """为 LVIS 训练视觉提示样本创建数据加载器。

        该方法为视觉提示嵌入 (VPE) 准备数据加载器。它会向数据集
        应用必要的变换（包括 LoadVisualPrompt）以用于验证目的。

        Args:
            data (dict): 包含路径和设置的数据集配置字典。

        Returns:
            (torch.utils.data.DataLoader): 视觉提示样本的数据加载器。
        """
        dataset = build_yolo_dataset(
            self.args,
            data.get(self.args.split, data.get("val")),
            self.args.batch,
            data,
            mode="val",
            rect=False,
        )
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )

    @smart_inference_mode()
    def __call__(
        self,
        trainer: Any | None = None,
        model: YOLOEModel | str | None = None,
        refer_data: str | None = None,
        load_vp: bool = False,
    ) -> dict[str, Any]:
        """使用文本或视觉提示嵌入对模型运行验证。

        该方法根据 load_vp 标志，使用文本提示或视觉提示对模型进行验证。
        支持在训练期间（使用训练器对象）或独立验证（提供模型）进行验证。
        对于视觉提示，可以指定参考数据以从不同的数据集中提取嵌入。

        Args:
            trainer (object, optional): 包含模型和设备的训练器对象。
            model (YOLOEModel | str, optional): 要验证的模型。如果未提供 trainer 则为必需。
            refer_data (str, optional): 用于视觉提示的参考数据路径。
            load_vp (bool): 是否加载视觉提示。如果为 False，则使用文本提示。

        Returns:
            (dict): 验证期间计算的指标统计信息。
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("使用视觉提示进行验证。")
                self.args.half = False
                # 直接使用训练期间提取的视觉嵌入的相同数据加载器
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, vpe)
            else:
                LOGGER.info("使用文本提示进行验证。")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
            stats = super().__call__(trainer, model)
        else:
            if refer_data is not None:
                assert load_vp, "Refer data is only used for visual prompt validation."
            self.device = select_device(self.args.device, verbose=False)

            if isinstance(model, (str, Path)):
                from ultralytics.nn.tasks import load_checkpoint

                model, _ = load_checkpoint(model, device=self.device)  # 模型, ckpt
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/", 1)[0] for name in list(data["names"].values())]

            if refer_data is not None:
                eval_data = check_det_dataset(self.args.data)
                eval_names = [name.split("/", 1)[0] for name in list(eval_data["names"].values())]
                if names != eval_names:
                    LOGGER.warning(
                        f"Class names from refer data {names} do not match evaluation dataset {eval_names}. "
                        f"This may lead to incorrect validation results."
                    )

            if load_vp:
                LOGGER.info("使用视觉提示进行验证。")
                self.args.half = False
                dataloader = self.get_vpe_dataloader(data)
                vpe = self.get_visual_pe(dataloader, model)
                model.set_classes(names, vpe)
                stats = super().__call__(model=deepcopy(model))
            elif isinstance(model.model[-1], YOLOEDetect) and hasattr(model.model[-1], "lrpc"):  # 无提示
                return super().__call__(trainer, model)
            else:
                LOGGER.info("使用文本提示进行验证。")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                stats = super().__call__(model=deepcopy(model))
        return stats


class YOLOESegValidator(YOLOEDetectValidator, SegmentationValidator):
    """支持文本和视觉提示嵌入的 YOLOE 分割验证器。"""

    pass
