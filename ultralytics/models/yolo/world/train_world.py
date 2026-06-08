# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DATASETS_DIR, DEFAULT_CFG, LOGGER
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import unwrap_model


class WorldTrainerFromScratch(WorldTrainer):
    """继承 WorldTrainer 用于在开放数据集上从头训练 World 模型的类。

    该训练器专门处理混合数据集，包括目标检测和 grounding 数据集，
    支持训练具有视觉-语言联合能力的 YOLO-World 模型。

    Attributes:
        cfg (dict): 包含模型训练默认参数的配置字典。
        overrides (dict): 自定义配置的参数字典覆盖项。
        _callbacks (dict): 训练各阶段执行的回调函数字典。
        data (dict): 包含训练/验证路径和元数据的最终处理后的数据配置。
        training_data (dict): 训练数据集路径到其配置的映射字典。

    Methods:
        build_dataset: 为训练或验证构建支持混合数据集的 YOLO 数据集。
        get_dataset: 从数据字典获取训练和验证路径。
        plot_training_labels: YOLO-World 训练时跳过标签图绘制。
        final_eval: 为 YOLO-World 模型执行最终评估和验证。

    Examples:
        >>> from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        >>> from ultralytics import YOLOWorld
        >>> data = dict(
        ...     train=dict(
        ...         yolo_data=["Objects365.yaml"],
        ...         grounding_data=[
        ...             dict(
        ...                 img_path="flickr30k/images",
        ...                 json_file="flickr30k/final_flickr_separateGT_train.json",
        ...             ),
        ...             dict(
        ...                 img_path="GQA/images",
        ...                 json_file="GQA/final_mixed_train_no_coco.json",
        ...             ),
        ...         ],
        ...     ),
        ...     val=dict(yolo_data=["lvis.yaml"]),
        ... )
        >>> model = YOLOWorld("yolov8s-worldv2.yaml")
        >>> model.train(data=data, trainer=WorldTrainerFromScratch)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """初始化 WorldTrainerFromScratch 对象。

        初始化用于从头训练 YOLO-World 模型的训练器，支持混合数据集，
        包括目标检测和 grounding 数据集以实现视觉-语言能力。

        Args:
            cfg (dict): 包含模型训练默认参数的配置字典。
            overrides (dict, optional): 自定义配置的参数字典覆盖项。
            _callbacks (dict, optional): 训练各阶段运行的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
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
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        datasets = [
            build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(
                # 从验证集分配 `nc` 作为训练时文本样本的最大数量以保持一致性
                self.args,
                im_path["img_path"],
                im_path["json_file"],
                batch,
                stride=gs,
                max_samples=self.data["nc"],
            )
            for im_path in img_path
        ]
        self.set_text_embeddings(datasets, batch)  # 缓存文本嵌入以加速训练
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    @staticmethod
    def check_data_config(data: dict | str | Path) -> dict:
        """检查并从 YAML 文件或字典加载数据配置。

        Args:
            data (dict | str | Path): 数据配置字典或 YAML 文件路径。

        Returns:
            (dict): 从 YAML 文件加载或直接传递的数据配置字典。
        """
        # 如果是字符串，则从 YAML 文件加载
        if not isinstance(data, dict):
            from ultralytics.utils import YAML

            return YAML.load(check_file(data))
        return data

    def get_dataset(self):
        """从数据字典获取训练和验证路径。

        处理数据配置以提取训练和验证数据集的路径，处理 YOLO 检测数据集
        和 grounding 数据集。

        Returns:
            (dict): 包含训练/验证路径和元数据的最终处理后的数据配置。

        Raises:
            AssertionError: 如果未找到训练或验证数据集，或验证集有多个数据集。
        """
        final_data = {}
        self.args.data = data_yaml = self.check_data_config(self.args.data)
        assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml 数据集
        assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml 数据集
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:  # 对于 lvis 数据集
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in {"train", "val"}:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            # 保存 grounding 数据（如果存在）
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
                for k in {"img_path", "json_file"}:
                    path = Path(g[k])
                    if not path.exists() and not path.is_absolute():
                        g[k] = str((DATASETS_DIR / g[k]).resolve())  # 相对于 DATASETS_DIR 的路径
            final_data[s] += grounding_data
        # 分配第一个验证数据集，因为当前仅支持一个验证集
        data["val"] = data["val"][0]
        final_data["val"] = final_data["val"][0]
        # 注意：为使训练正常工作，设置 `nc` 和 `names`
        final_data["nc"] = data["val"]["nc"]
        final_data["names"] = data["val"]["names"]
        # 注意：添加 lvis 的路径
        final_data["path"] = data["val"]["path"]
        final_data["channels"] = data["val"]["channels"]
        self.data = final_data
        if self.args.single_cls:  # 与基础训练器保持一致
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "object"}
            self.data["nc"] = 1
        self.training_data = {}
        for d in data["train"]:
            if self.args.single_cls:
                d["names"] = {0: "object"}
                d["nc"] = 1
            self.training_data[d["train"]] = d
        return final_data

    def plot_training_labels(self):
        """YOLO-World 训练时跳过标签图绘制。"""
        pass

    def final_eval(self):
        """为 YOLO-World 模型执行最终评估和验证。

        在运行评估前使用适当的数据集和分割信息配置验证器。

        Returns:
            (dict): 包含评估指标和结果的字典。
        """
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()
