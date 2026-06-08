# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path

import torch

from ultralytics.data import YOLOConcatDataset, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from ..world.train_world import WorldTrainerFromScratch
from .val import YOLOEDetectValidator


class YOLOETrainer(DetectionTrainer):
    """用于 YOLOE 目标检测模型的训练器类。

    该类继承 DetectionTrainer，为 YOLOE 模型提供专门的训练功能，
    包括自定义模型初始化、验证以及支持多模态的数据集构建。

    Attributes:
        loss_names (tuple): 训练期间使用的损失分量名称。

    Methods:
        get_model: 初始化并返回使用指定配置的 YOLOEModel。
        get_validator: 返回用于模型验证的 YOLOEDetectValidator。
        build_dataset: 构建支持多模态训练的 YOLO 数据集。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """使用指定配置初始化 YOLOE 训练器。

        Args:
            cfg (dict): 包含 DEFAULT_CFG 默认训练设置的配置字典。
            overrides (dict, optional): 默认配置的参数字典覆盖项。
            _callbacks (dict, optional): 训练期间应用的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        overrides["overlap_mask"] = False
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """返回使用指定配置和权重初始化的 YOLOEModel。

        Args:
            cfg (dict | str, optional): 模型配置。可以是包含 'yaml_file' 键的字典、直接的
                YAML 文件路径或 None（使用默认配置）。
            weights (str | Path, optional): 预训练权重文件的路径。
            verbose (bool): 是否在初始化期间显示模型信息。

        Returns:
            (YOLOEModel): 初始化后的 YOLOE 模型。

        Notes:
            - 类别数量 (nc) 遵循官方配置，最多硬编码为 80。
            - 此处的 nc 参数表示单张图像中不同文本样本的最大数量，而非实际的类别数量。
        """
        # 注意：此处的 `nc` 是单张图像中不同文本样本的最大数量，而非实际的 `nc`。
        # 注意：遵循官方配置，nc 当前硬编码为 80。
        model = YOLOEModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """返回用于 YOLOE 模型验证的 YOLOEDetectValidator。"""
        self.loss_names = "box", "cls", "dfl"
        return YOLOEDetectValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """构建 YOLO 数据集。

        Args:
            img_path (str): 包含图像的文件夹路径。
            mode (str): 'train' 模式或 'val' 模式，用户可以为每种模式自定义不同的数据增强。
            batch (int, optional): 批次大小，用于矩形训练。

        Returns:
            (Dataset): 配置好用于训练或验证的 YOLO 数据集。
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )


class YOLOEPETrainer(DetectionTrainer):
    """使用线性探测方法微调 YOLOE 模型。

    该训练器冻结大部分模型层，仅训练特定的投影层，以在新数据集上高效微调，
    同时保留预训练特征。

    Methods:
        get_model: 初始化 YOLOEModel，除投影层外冻结所有层。
    """

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """返回使用指定配置和权重初始化的 YOLOEModel。

        Args:
            cfg (dict | str, optional): 模型配置。
            weights (str, optional): 预训练权重的路径。
            verbose (bool): 是否显示模型信息。

        Returns:
            (YOLOEModel): 初始化后的模型，除特定投影层外冻结所有层。
        """
        # 注意：此处的 `nc` 是单张图像中不同文本样本的最大数量，而非实际的 `nc`。
        # 注意：遵循官方配置，nc 当前硬编码为 80。
        model = YOLOEModel(
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
        model.model[-1].fuse(model.pe)  # 将文本嵌入融合到分类头中
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

        if getattr(model.model[-1], "one2one_cv3", None) is not None:
            model.model[-1].one2one_cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

        model.train()

        return model


class YOLOETrainerFromScratch(YOLOETrainer, WorldTrainerFromScratch):
    """从头训练支持文本嵌入的 YOLOE 模型。

    该训练器结合 YOLOE 训练能力和 World 训练特性，支持从头开始训练
    使用文本嵌入和 grounding 数据集的模型。

    Methods:
        build_dataset: 构建支持 grounding 的训练数据集。
        generate_text_embeddings: 生成并缓存训练用的文本嵌入。
    """

    def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
        """为训练或验证构建 YOLO 数据集。

        该方法根据模式和输入路径构建适当的数据集，处理标准 YOLO 数据集
        和不同格式的 grounding 数据集。

        Args:
            img_path (list[str] | str): 包含图像的文件夹路径或路径列表。
            mode (str): 'train' 模式或 'val' 模式，允许为每种模式自定义不同的数据增强。
            batch (int, optional): 批次大小，用于矩形训练/验证。

        Returns:
            (YOLOConcatDataset | Dataset): 构建的用于训练或验证的数据集。
        """
        return WorldTrainerFromScratch.build_dataset(self, img_path, mode, batch)

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path):
        """为文本样本列表生成文本嵌入。

        Args:
            texts (list[str]): 要编码的文本样本列表。
            batch (int): 处理的批次大小。
            cache_dir (Path): 保存/加载缓存嵌入的目录。

        Returns:
            (dict): 文本样本到其嵌入的映射字典。
        """
        model = unwrap_model(self.model).text_model
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, without_reprta=True, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map


class YOLOEPEFreeTrainer(YOLOEPETrainer, YOLOETrainerFromScratch):
    """训练无提示的 YOLOE 模型。

    该训练器结合线性探测能力和从头训练功能，用于训练推理时不需要
    文本提示的无提示 YOLOE 模型。

    Methods:
        get_validator: 返回标准 DetectionValidator 用于验证。
        preprocess_batch: 预处理批次，不包含文本特征。
        set_text_embeddings: 为数据集设置文本嵌入（无提示模式下为空操作）。
    """

    def get_validator(self):
        """返回用于 YOLO 模型验证的 DetectionValidator。"""
        self.loss_names = "box", "cls", "dfl"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def preprocess_batch(self, batch):
        """为 YOLOE 训练预处理图像批次，根据需要调整格式和维度。"""
        return DetectionTrainer.preprocess_batch(self, batch)

    def set_text_embeddings(self, datasets, batch: int):
        """无提示训练的空操作覆盖，不需要文本嵌入。

        Args:
            datasets (list[Dataset]): 包含待处理类别名称的数据集列表。
            batch (int): 处理文本嵌入的批次大小。
        """
        pass


class YOLOEVPTrainer(YOLOETrainerFromScratch):
    """使用视觉提示训练 YOLOE 模型。

    该训练器继承 YOLOETrainerFromScratch，支持基于视觉提示的训练，
    其中视觉线索与图像一起提供以引导检测过程。

    Methods:
        build_dataset: 构建带有视觉提示加载变换的数据集。
    """

    def build_dataset(self, img_path: list[str] | str, mode: str = "train", batch: int | None = None):
        """为训练或验证构建带视觉提示的 YOLO 数据集。

        Args:
            img_path (list[str] | str): 包含图像的文件夹路径或路径列表。
            mode (str): 'train' 模式或 'val' 模式，允许为每种模式自定义不同的数据增强。
            batch (int, optional): 批次大小，用于矩形训练/验证。

        Returns:
            (YOLOConcatDataset | Dataset): 配置好用于训练或验证的 YOLO 数据集，
                训练模式包含视觉提示。
        """
        dataset = super().build_dataset(img_path, mode, batch)
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return dataset

    def _close_dataloader_mosaic(self):
        """Close mosaic augmentation and add visual prompt loading to the training dataset."""
        super()._close_dataloader_mosaic()
        if isinstance(self.train_loader.dataset, YOLOConcatDataset):
            for d in self.train_loader.dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            self.train_loader.dataset.transforms.append(LoadVisualPrompt())
