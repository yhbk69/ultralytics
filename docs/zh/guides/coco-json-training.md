---
comments: true
description: 无需转换为 YOLO 格式，即可直接在 COCO JSON 标注上训练 Ultralytics YOLO。包含自定义数据集和训练器示例，提供完整可运行的检测训练代码。
keywords: COCO JSON 训练, COCO JSON 训练 YOLO, COCO JSON 无需转换, 自定义 YOLO 数据集, 自定义 YOLO 训练器, COCO 标注 YOLO, 直接 COCO 训练, Ultralytics YOLO, 目标检测训练, YOLODataset 子类, COCO 格式训练, 跳过标注转换
---

# 如何在不转换的情况下直接在 COCO JSON 上训练 YOLO

## 为什么直接在 COCO JSON 上训练

[COCO JSON](https://cocodataset.org/#format-data) 格式的[标注](https://www.ultralytics.com/glossary/data-labeling)可以直接用于 [Ultralytics YOLO](https://www.ultralytics.com/) 训练，无需先转换为 `.txt` 文件。实现方式是通过继承 `YOLODataset` 来实时解析 COCO JSON，并通过自定义训练器将其接入训练流程。

这种方法将 COCO JSON 作为唯一数据源——无需调用 `convert_coco()`，无需重新组织目录结构，也不产生中间标签文件。[YOLO26](../models/yolo26.md) 及所有其他 Ultralytics YOLO 检测模型均受支持。分割和姿态估计模型需要额外的标签字段（参见[常见问题](#是否支持分割和姿态估计)）。

!!! tip "想要一次性转换？"

    请参阅 [COCO 转 YOLO 转换指南](coco-to-yolo.md)，了解标准的 `convert_coco()` 工作流程。

## 架构概览

需要两个类：

1. **`COCODataset`** — 读取 COCO JSON，并在训练期间于内存中将[边界框](https://www.ultralytics.com/glossary/bounding-box)转换为 YOLO 格式
2. **`COCOTrainer`** — 重写 `build_dataset()` 方法，使用 `COCODataset` 替代默认的 `YOLODataset`

该实现遵循与内置 `GroundingDataset` 相同的模式，后者也是直接读取 JSON 标注。需要重写三个方法：`get_img_files()`、`cache_labels()` 和 `get_labels()`。

## 构建 COCO JSON 数据集类

`COCODataset` 类继承自 `YOLODataset`，并重写了标签加载逻辑。它不再从标签目录读取 `.txt` 文件，而是打开 COCO JSON 文件，遍历按图像分组的标注，将每个边界框从 COCO 像素格式 `[x_min, y_min, width, height]` 转换为 YOLO 归一化中心点格式 `[x_center, y_center, width, height]`。聚合标注（`iscrowd: 1`）和零面积框会被自动跳过。

`get_img_files()` 方法返回空列表，因为图像路径是在 `cache_labels()` 内部通过 JSON 中的 `file_name` 字段解析的。类别 ID 会按排序后映射为从零开始的类别索引，因此无论是从 1 开始的 ID（标准 COCO）还是非连续的 ID 方案都能正确处理。

```python
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.utils import TQDM


class COCODataset(YOLODataset):
    """直接读取 COCO JSON 标注的数据集，无需转换为 .txt 文件。"""

    def __init__(self, *args, json_file="", **kwargs):
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        """图像路径从 JSON 文件中解析，而非扫描目录。"""
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """解析 COCO JSON 并将标注转换为 YOLO 格式。结果保存到 .cache 文件。"""
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}

        # 按 ID 排序类别并映射为从 0 开始的索引
        categories = {cat["id"]: i for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                # COCO: [x, y, w, h] 左上角像素坐标 -> YOLO: [cx, cy, w, h] 中心点归一化
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2  # 左上角转中心点
                box[[0, 2]] /= w  # 归一化 x
                box[[1, 3]] /= h  # 归一化 y
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """如果 .cache 文件可用则加载缓存，否则解析 JSON 并创建缓存。"""
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]
```

解析后的标签会保存到 JSON 文件旁边的 `.cache` 文件中（例如 `instances_train.cache`）。在后续训练运行时，缓存会被直接加载，跳过 JSON 解析。如果 JSON 文件发生更改，哈希校验会失败，缓存将自动重建。

## 将数据集接入训练流程

训练器中唯一需要修改的是重写 `build_dataset()` 方法。默认的 `DetectionTrainer` 构建一个 `YOLODataset` 来扫描 `.txt` 标签文件。将其替换为 `COCODataset` 后，训练器将改为从 COCO JSON 读取数据。

JSON 文件路径从数据配置中的自定义字段 `train_json` / `val_json` 获取（参见步骤 3）。训练期间，`mode="train"` 解析为 `train_json`；验证期间，`mode="val"` 解析为 `val_json`。如果未设置 `val_json`，则回退使用 `train_json`。

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr


class COCOTrainer(DetectionTrainer):
    """使用 COCODataset 进行直接 COCO JSON 训练的训练器。"""

    def build_dataset(self, img_path, mode="train", batch=None):
        json_file = self.data["train_json"] if mode == "train" else self.data.get("val_json", self.data["train_json"])
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
```

## 为 COCO JSON 配置 dataset.yaml

`dataset.yaml` 使用标准的 `path`、`train` 和 `val` 字段来定位图像目录。另外两个字段 `train_json` 和 `val_json` 指定 `COCOTrainer` 读取的 COCO 标注文件。`nc` 和 `names` 字段定义类别数量和名称，与 JSON 中 `categories` 的排序顺序一致。

```yaml
path: /path/to/images # 根目录，包含 train/ 和 val/ 子文件夹
train: train
val: val

# COCO JSON 标注文件
train_json: /path/to/annotations/instances_train.json
val_json: /path/to/annotations/instances_val.json

nc: 80
names:
    0: person
    1: bicycle
    # ... 其余类别名称
```

期望的目录结构：

```
my_dataset/
  images/
    train/
      img_001.jpg
      ...
    val/
      img_100.jpg
      ...
  annotations/
    instances_train.json
    instances_val.json
  dataset.yaml
```

## 在 COCO JSON 上运行训练

有了数据集类、训练器类和 YAML 配置后，通过标准的 `model.train()` 调用即可开始训练。与普通训练运行的唯一区别是 `trainer=COCOTrainer` 参数，它告诉 Ultralytics 使用自定义数据集加载器而非默认加载器。

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640, trainer=COCOTrainer)
```

完整的[训练](../modes/train.md)流程会按预期运行，包括[验证](../modes/val.md)、检查点保存和指标记录。

## 完整实现

为了方便使用，下面提供了完整的实现代码，可作为单个复制粘贴脚本使用。它包含了自定义数据集、自定义训练器和训练调用。将其与 `dataset.yaml` 放在一起即可直接运行。

```python
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import TQDM, colorstr


class COCODataset(YOLODataset):
    """直接读取 COCO JSON 标注的数据集，无需转换为 .txt 文件。"""

    def __init__(self, *args, json_file="", **kwargs):
        self.json_file = json_file
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        with open(self.json_file) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        categories = {cat["id"]: i for i, cat in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [lb["im_file"] for lb in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)
        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]


class COCOTrainer(DetectionTrainer):
    """使用 COCODataset 进行直接 COCO JSON 训练的训练器。"""

    def build_dataset(self, img_path, mode="train", batch=None):
        json_file = self.data["train_json"] if mode == "train" else self.data.get("val_json", self.data["train_json"])
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


model = YOLO("yolo26n.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640, trainer=COCOTrainer)
```

关于[超参数](https://www.ultralytics.com/glossary/hyperparameter-tuning)建议，请参阅[模型训练技巧](model-training-tips.md)指南。

## 常见问题

### 这与 convert_coco() 有什么区别？

[`convert_coco()`](../reference/data/converter.md#ultralytics.data.converter.convert_coco) 将 `.txt` 标签文件作为一次性转换写入磁盘。本方法在每次训练运行开始时解析 JSON，并在内存中转换标注。如果希望使用永久的 YOLO 格式标签，请使用 `convert_coco()`；如果希望将 COCO JSON 作为唯一数据源而不生成额外文件，请使用本方法。

### YOLO 能否在不编写自定义代码的情况下直接训练 COCO JSON？

当前的 Ultralytics 流程默认期望 YOLO 格式的 `.txt` 标签，因此不能。本指南提供了所需的最小自定义代码——一个数据集类和一个训练器类。定义完成后，训练只需一个标准的 `model.train()` 调用即可。

### 是否支持分割和姿态估计？

本指南涵盖[目标检测](https://www.ultralytics.com/glossary/object-detection)。要添加[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)支持，请在每个标签字典的 `segments` 字段中包含 COCO 标注中的 `segmentation` 多边形数据。对于[姿态估计](https://www.ultralytics.com/glossary/pose-estimation)，请包含 `keypoints` 数据。`GroundingDataset` 的[源代码](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/dataset.py)提供了处理分割数据的参考实现。

### 数据增强是否适用于这个自定义数据集？

是的。`COCODataset` 继承自 `YOLODataset`，因此所有内置的[数据增强](yolo-data-augmentation.md)方法——[mosaic](yolo-data-augmentation.md#mosaic-mosaic)、[mixup](yolo-data-augmentation.md#mixup-mixup)、[copy-paste](yolo-data-augmentation.md#copy-paste-copy_paste) 等——均可无需修改直接运行。

### 类别 ID 如何映射为类别索引？

类别按 `id` 排序后映射为从 0 开始的连续索引。这可以处理从 1 开始的 ID（标准 COCO）、从 0 开始的 ID 以及非连续的 ID。`dataset.yaml` 中的 `names` 字典应遵循与 COCO `categories` 数组相同的排序顺序。

### 与预先转换的标签相比，是否有性能开销？

COCO JSON 仅在首次训练运行时解析一次。解析后的标签会保存到 `.cache` 文件中，因此后续运行可即时加载，无需重新解析。训练速度与标准 YOLO 训练完全相同，因为标注数据保存在内存中。如果 JSON 文件发生更改，缓存将自动重建。