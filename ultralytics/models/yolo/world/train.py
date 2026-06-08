# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


def on_pretrain_routine_end(trainer) -> None:
    """在预训练流程结束时设置模型类别和文本编码器。"""
    if RANK in {-1, 0}:
        # 设置用于评估的类别名称
        names = [name.split("/", 1)[0] for name in list(trainer.test_loader.dataset.data["names"].values())]
        unwrap_model(trainer.ema.ema).set_classes(names, cache_clip_model=False)


class WorldTrainer(DetectionTrainer):
    """用于在封闭数据集上微调 YOLO World 模型的训练器类。

    该训练器继承 DetectionTrainer，支持训练结合视觉和文本特征的 YOLO World 模型，
    以改进目标检测和理解。它处理文本嵌入生成和缓存，以加速多模态数据训练。

    Attributes:
        text_embeddings (dict[str, torch.Tensor] | None): 缓存的类别名称文本嵌入，用于加速训练。
        model (WorldModel): 正在训练的 YOLO World 模型。
        data (dict[str, Any]): 包含类别信息的数据集配置。
        args (Any): 训练参数和配置。

    Methods:
        get_model: 返回使用指定配置和权重初始化的 WorldModel。
        build_dataset: 为训练或验证构建 YOLO 数据集。
        set_text_embeddings: 为数据集设置文本嵌入以加速训练。
        generate_text_embeddings: 为文本样本列表生成文本嵌入。
        preprocess_batch: 为 YOLOWorld 训练预处理图像和文本批次。

    Examples:
        初始化并训练 YOLO World 模型
        >>> from ultralytics.models.yolo.world import WorldTrainer
        >>> args = dict(model="yolov8s-world.pt", data="coco8.yaml", epochs=3)
        >>> trainer = WorldTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """使用给定参数初始化 WorldTrainer 对象。

        Args:
            cfg (dict[str, Any]): 训练器的配置。
            overrides (dict[str, Any], optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        if overrides is None:
            overrides = {}
        assert not overrides.get("compile"), f"Training with 'model={overrides['model']}' requires 'compile=False'"
        super().__init__(cfg, overrides, _callbacks)
        self.text_embeddings = None

    def get_model(self, cfg=None, weights: str | None = None, verbose: bool = True) -> WorldModel:
        """返回使用指定配置和权重初始化的 WorldModel。

        Args:
            cfg (dict[str, Any] | str, optional): 模型配置。
            weights (str, optional): 预训练权重的路径。
            verbose (bool): 是否显示模型信息。

        Returns:
            (WorldModel): 初始化后的 WorldModel。
        """
        # 注意：此处的 `nc` 是单张图像中不同文本样本的最大数量，而非实际的 `nc`。
        # 注意：遵循官方配置，nc 当前硬编码为 80。
        model = WorldModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """为训练或验证构建 YOLO 数据集。

        Args:
            img_path (str): 包含图像的文件夹路径。
            mode (str): 'train' 模式或 'val' 模式，用户可以为每种模式自定义不同的数据增强。
            batch (int, optional): 批次大小，用于 'rect' 模式。

        Returns:
            (Any): 配置好用于训练或验证的 YOLO 数据集。
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )
        if mode == "train":
            self.set_text_embeddings([dataset], batch)  # 缓存文本嵌入以加速训练
        return dataset

    def set_text_embeddings(self, datasets: list[Any], batch: int | None) -> None:
        """通过缓存类别名称为数据集设置文本嵌入以加速训练。

        该方法从所有数据集中收集唯一的类别名称，然后为这些类别生成并缓存文本嵌入，
        以提高训练效率。

        Args:
            datasets (list[Any]): 从中提取类别名称的数据集列表。
            batch (int | None): 用于处理的批次大小。

        Notes:
            该方法从具有 'category_names' 属性的数据集中收集类别名称，
            然后使用第一个数据集的图像路径来确定缓存生成的文本嵌入的位置。
        """
        text_embeddings = {}
        for dataset in datasets:
            if not hasattr(dataset, "category_names"):
                continue
            text_embeddings.update(
                self.generate_text_embeddings(
                    list(dataset.category_names), batch, cache_dir=Path(dataset.img_path).parent
                )
            )
        self.text_embeddings = text_embeddings

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path) -> dict[str, torch.Tensor]:
        """为文本样本列表生成文本嵌入。

        Args:
            texts (list[str]): 要编码的文本样本列表。
            batch (int): 处理的批次大小。
            cache_dir (Path): 保存/加载缓存嵌入的目录。

        Returns:
            (dict[str, torch.Tensor]): 文本样本到其嵌入的映射字典。
        """
        model = "clip:ViT-B/32"
        cache_path = cache_dir / f"text_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"
        if cache_path.exists():
            LOGGER.info(f"Reading existed cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map
        LOGGER.info(f"Caching text embeddings to '{cache_path}'")
        assert self.model is not None
        txt_feats = unwrap_model(self.model).get_text_pe(texts, batch, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """为 YOLOWorld 训练预处理图像和文本批次。"""
        batch = DetectionTrainer.preprocess_batch(self, batch)

        # 添加文本特征
        texts = list(itertools.chain(*batch["texts"]))
        txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch
