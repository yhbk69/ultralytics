---
comments: true
description: 了解如何将 COCO JSON 标注转换为 YOLO 格式，用于目标检测、实例分割和姿态估计训练。包含分步示例、常见问题及自定义数据集的类别 ID 映射的完整指南。
keywords: COCO 转 YOLO, 转换 COCO JSON 为 YOLO, COCO JSON 格式, YOLO 标注格式, convert_coco, COCO 数据集训练, 用 COCO 训练 YOLO, 目标检测数据集, 实例分割数据集, 姿态估计数据集, 数据集转换, 标注格式, cls91to80, category_id, 边界框格式, YOLO 训练数据
---

# 如何将 COCO 标注转换为 YOLO 格式

训练 [Ultralytics YOLO](https://www.ultralytics.com/) 模型需要使用 YOLO 格式的标注，但许多流行的 [标注](https://www.ultralytics.com/glossary/data-labeling) 工具导出的却是 [COCO JSON](https://cocodataset.org/#format-data) 格式。本指南将展示如何将 COCO 标注转换为 YOLO 格式，并开始训练 [目标检测](https://www.ultralytics.com/glossary/object-detection)、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation) 和 [姿态估计](https://www.ultralytics.com/glossary/pose-estimation) 模型。

## 为什么需要从 COCO 转换为 YOLO？

COCO JSON 格式将所有标注存储在一个文件中，而 [YOLO](https://docs.ultralytics.com/datasets/detect#ultralytics-yolo-format) 则为每张图像使用一个文本文件，并使用归一化坐标。需要转换的原因如下：

- **YOLO 模型需要 `.txt` 标签文件**，每张图像对应一个文件，包含归一化坐标格式的 `class x_center y_center width height`。
- **COCO JSON 使用像素坐标**，格式为 `[x_min, y_min, width, height]`，所有图像共用一个 JSON 文件。
- **类别 ID 不同** — COCO 使用任意的 `category_id` 值，而 YOLO 需要从零开始的类别 ID。

| 特性             | COCO JSON                                  | YOLO TXT                                                  |
| ---------------- | ------------------------------------------ | --------------------------------------------------------- |
| **结构**         | 所有图像共用一个 JSON 文件                  | 每张图像一个 `.txt` 文件                                   |
| **边界框格式**   | 像素坐标 `[x_min, y_min, width, height]`   | 归一化坐标 `class x_center y_center width height`（0-1）   |
| **类别 ID**      | `category_id`（可从任意数字开始）           | 从零开始索引（从 0 开始）                                  |
| **分割**         | `segmentation` 字段中的多边形数组           | 类别 ID 后的多边形坐标                                     |
| **关键点**       | 像素坐标 `[x, y, visibility, ...]`         | 归一化坐标 `[x, y, visibility, ...]`                      |

## 快速开始

转换 COCO 标注并开始训练的最快方式：

```python
from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="path/to/annotations/",  # 包含 JSON 文件的目录
    save_dir="path/to/output/",  # 转换后标签的保存目录
    cls91to80=False,  # 重要：对于自定义数据集请设为 False
)
```

转换完成后，[整理目录结构](#3-整理目录结构)，[创建 dataset.yaml](#4-创建-datasetyaml)，然后 [开始训练](#5-训练你的-yolo-模型)。详见下方的完整 [分步指南](#分步转换指南)。

!!! warning "自定义数据集：始终使用 `cls91to80=False`"

    默认值 `cls91to80=True` **仅** 适用于包含 80 个目标类别的标准 [COCO 数据集](../datasets/detect/coco.md)，该选项会将 91 个非连续的类别 ID 映射为 80 个连续的类别 ID。对于任何自定义数据集，你**必须**设置 `cls91to80=False` — 否则你的类别 ID 将被静默地错误映射，导致模型学习到错误的类别。

## 分步转换指南

### 1. 准备你的 COCO 数据集

从标注工具导出的典型 COCO 格式数据集结构如下：

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── val/
│       ├── img_100.jpg
│       └── ...
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

每个 JSON 文件遵循 [COCO 数据格式](https://cocodataset.org/#format-data) 规范，包含三个必需字段 — `images`、`annotations` 和 `categories`：

```json
{
    "images": [{ "id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480 }],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 50, 200, 150],
            "area": 30000,
            "iscrowd": 0
        }
    ],
    "categories": [
        { "id": 1, "name": "helmet" },
        { "id": 2, "name": "vest" }
    ]
}
```

### 2. 转换标注

使用 [`convert_coco()`](../reference/data/converter.md#ultralytics.data.converter.convert_coco) 函数将 COCO JSON 标注转换为 YOLO `.txt` 格式：

!!! example "将 COCO 转换为 YOLO 格式"

    === "目标检测"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(
            labels_dir="my_dataset/annotations/",
            save_dir="my_dataset/converted/",
            cls91to80=False,
        )
        ```

    === "实例分割"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(
            labels_dir="my_dataset/annotations/",
            save_dir="my_dataset/converted/",
            use_segments=True,
            cls91to80=False,
        )
        ```

    === "姿态估计"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(
            labels_dir="my_dataset/annotations/",
            save_dir="my_dataset/converted/",
            use_keypoints=True,
            cls91to80=False,
        )
        ```

### 3. 整理目录结构

转换后，标签文件需要与图像放在一起。YOLO 期望 `labels/` 目录与 `images/` 目录结构镜像对应：

```python
import shutil
from pathlib import Path

# 路径
converted_dir = Path("my_dataset/converted/labels")
dataset_dir = Path("my_dataset")

# 将每个 split 的标签移动到图像旁边
for split in ["train", "val"]:
    src = converted_dir / split  # convert_coco 会从 JSON 文件名中去除 "instances_" 前缀
    dst = dataset_dir / "labels" / split
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.txt"):
        shutil.move(str(f), str(dst / f.name))
```

最终的 [数据集结构](../datasets/detect/index.md#ultralytics-yolo-format) 应如下所示：

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img_001.txt
│   │   └── ...
│   └── val/
│       └── ...
└── dataset.yaml
```

### 4. 创建 dataset.yaml

创建一个 `dataset.yaml` 配置文件，将 COCO 类别映射为 YOLO 类别名称。该文件告诉 YOLO 数据在哪里以及需要检测哪些类别：

```python
import json
from pathlib import Path

import yaml

# 从 COCO JSON 中读取类别
with open("my_dataset/annotations/instances_train.json") as f:
    coco = json.load(f)

# 构建与 convert_coco 输出匹配的类别名称（category_id - 1）
categories = sorted(coco["categories"], key=lambda x: x["id"])
names = {cat["id"] - 1: cat["name"] for cat in categories}
# 注意：convert_coco 将类别 ID 映射为 category_id - 1，因此 category_id 必须
# 从 1 开始。如果你的类别从 0 开始，请先将每个 ID 加 1。

# 创建 dataset.yaml
dataset = {
    "path": str(Path("my_dataset").resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": names,
}

with open("my_dataset/dataset.yaml", "w") as f:
    yaml.dump(dataset, f, default_flow_style=False)
```

生成的 YAML 文件：

```yaml
path: /absolute/path/to/my_dataset
train: images/train
val: images/val
names:
    0: helmet
    1: vest
```

关于数据集 YAML 格式的更多细节，请参阅 [数据集配置指南](../datasets/detect/index.md)。

### 5. 训练你的 YOLO 模型

准备好转换后的数据集后，训练一个 YOLO 模型：

!!! example "在转换后的 COCO 数据上训练"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # 加载预训练模型
        results = model.train(data="my_dataset/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26n.pt data=my_dataset/dataset.yaml epochs=100 imgsz=640
        ```

关于训练技巧和最佳实践，请参阅 [模型训练指南](model-training-tips.md)。

### 6. 验证你的转换结果

在训练之前，抽查几个标签文件以确认类别 ID 和坐标是否正确：

```python
from pathlib import Path

label_file = Path("my_dataset/labels/train/img_001.txt")
for line in label_file.read_text().strip().splitlines():
    parts = line.split()
    cls_id = int(parts[0])
    coords = [float(v) for v in parts[1:5]]
    assert cls_id >= 0, f"类别 ID 为负数 {cls_id} — JSON 中的 category_id 可能从 0 开始"
    assert all(0 <= v <= 1 for v in coords), f"坐标超出 [0, 1] 范围: {coords}"
```

!!! tip

    如果看到负数类别 ID，说明你的 COCO JSON 可能使用了从 0 开始的 `category_id`。在运行 `convert_coco()` 之前，将 JSON 中所有 `category_id` 值加 1，因为它的类别 ID 映射方式为 `category_id - 1`。

## 常见问题排查

### 转换后类别 ID 错误

如果你的模型可以训练，但检测到的目标类别是错误的，那么你可能在自定义数据集上使用了 `cls91to80=True`（默认值）。这会通过 COCO 91-转-80 的查找表映射你的 `category_id` 值，而这仅对标准 [COCO 数据集](../datasets/detect/coco.md) 是正确的。

**解决方案**：对于自定义数据集，始终使用 `cls91to80=False`。

### 训练时显示未找到标签

如果训练显示 `WARNING: No labels found` 或 `0 images, N backgrounds`，说明你的标签文件不在预期的目录中。`convert_coco()` 会将标签保存到一个单独的输出目录（例如 `save_dir/labels/train/`），但 YOLO 期望 `labels/` 与 `images/` 在数据集目录内平行放置。

**解决方案**：移动标签文件以匹配预期的 [目录结构](#3-整理目录结构)。确保 `labels/train/` 与 `images/train/` 同级。

### 转换过程中出现 KeyError

如果在运行 `convert_coco()` 时出现 `KeyError: 'bbox'` 或类似错误，说明你的 `labels_dir` 中可能包含了非实例标注的 JSON 文件（例如 `captions_train2017.json`），这些文件具有不同的标注结构。

**解决方案**：仅在 `labels_dir` 中放置实例标注 JSON 文件（例如 `instances_train2017.json`）。

### 转换后标签文件为空

如果转换完成但 `.txt` 文件为空或缺失，则所有标注可能都是 `iscrowd: 1`（常见于 [SAM](../models/sam.md) 生成的掩码），或者 [边界框](https://www.ultralytics.com/glossary/bounding-box) 的宽度或高度为零。

**解决方案**：检查 JSON 标注中的 `iscrowd` 值。如果使用 SAM 掩码，请预处理 JSON，将 `iscrowd` 设为 `0`。

### 转换后标签文件中类别 ID 不连续

如果标签文件中的类别 ID 不连续（例如 0, 4, 9 而不是 0, 1, 2），说明你的标注工具使用了非连续的 `category_id` 值。

**解决方案**：验证 `.txt` 文件中的类别 ID 与 `dataset.yaml` 中的 `names` 字典是否匹配。如有需要，将 ID 重新映射为连续值。

关于完整的 API 细节和参数描述，请参阅 [`convert_coco` API 参考](../reference/data/converter.md#ultralytics.data.converter.convert_coco)。

## 常见问题

### 如何将 COCO JSON 标注转换为 YOLO 格式？

使用 Ultralytics 的 `convert_coco()` 函数将 COCO JSON 标注转换为 YOLO `.txt` 格式。对于自定义数据集，设置 `cls91to80=False`：

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/annotations/", save_dir="output/", cls91to80=False)
```

转换后，重新整理标签文件使 `labels/` 与 `images/` 目录镜像对应，然后创建 `dataset.yaml` 文件。完整流程请参阅 [分步指南](#分步转换指南)。

### 为什么 COCO 转换后 YOLO 训练显示"未找到标签"？

这是因为 `convert_coco()` 将标签保存到 `save_dir/labels/` 的子目录中（例如 `save_dir/labels/train/`），而不是直接保存到数据集 `images/train/` 旁边的 `labels/train/`。YOLO 期望标签与图像平行放置 — 例如 `images/train/img.jpg` 需要对应 `labels/train/img.txt`。将转换后的标签移动到匹配此结构的位置。参见 [修正目录结构](#3-整理目录结构)。

### `convert_coco()` 中的 `cls91to80` 有什么作用？

`cls91to80` 参数控制 COCO `category_id` 值如何映射为 YOLO 类别 ID。当设为 `True`（默认值）时，它使用为标准 [COCO 数据集](../datasets/detect/coco.md) 设计的查找表，该数据集有 80 个类别，ID 非连续（1-90）。对于**自定义数据集**，始终设置 `cls91to80=False` — 这会将每个 `category_id` 减去 1 来创建从零开始的类别 ID。

### 能否直接在 COCO JSON 上训练 YOLO 而不进行转换？

目前 YOLO 训练流程不支持 — 标注必须是 YOLO `.txt` 格式，每张图像一个文件。请先使用 `convert_coco()` 转换 COCO JSON，然后按照本 [指南](#分步转换指南) 整理和训练。关于支持的格式，详见 [数据集格式](../datasets/detect/index.md)。

### 能否将 COCO 分割标注转换为 YOLO 格式？

可以，在调用 `convert_coco()` 时使用 `use_segments=True`，即可在转换后的 YOLO 标签中包含多边形分割掩码。这将生成与 [YOLO 分割模型](../tasks/segment.md) 兼容的标签文件：

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="annotations/", save_dir="output/", use_segments=True, cls91to80=False)
```

### 如何将 COCO 关键点标注转换为 YOLO 格式？

使用 `use_keypoints=True` 来转换 COCO 关键点标注，用于 [姿态估计](../tasks/pose.md) 训练：

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="annotations/", save_dir="output/", use_keypoints=True, cls91to80=False)
```

请注意，如果 `use_segments` 和 `use_keypoints` 同时设为 `True`，则只有关键点会被写入标签文件 — 分割信息将被静默忽略。