---
comments: true
description: 探索 Ultralytics YOLO 支持的数据集格式，了解如何准备和使用数据集来训练目标分割模型。
keywords: Ultralytics, YOLO, 实例分割, 数据集格式, 自动标注, COCO, 分割模型, 训练数据
---

# 实例分割数据集概述

实例分割是一种计算机视觉任务，涉及识别和勾勒图像中的各个对象。本指南概述了 Ultralytics YOLO 针对实例分割任务所支持的数据集格式，以及如何准备、转换和使用这些数据集来训练模型。

## 支持的数据集格式

### Ultralytics YOLO 格式

用于训练 YOLO 分割模型的数据集标签格式如下：

1. 每张图像对应一个文本文件：数据集中每张图像都有一个同名的文本文件，扩展名为 ".txt"。
2. 每行对应一个对象：文本文件中的每一行对应图像中的一个对象实例。
3. 每行包含对象信息：每行包含以下关于对象实例的信息：
    - 对象类别索引：表示对象类别的整数（例如，0 表示人，1 表示汽车等）。
    - 对象边界坐标：掩码区域的边界坐标，归一化到 0 到 1 之间。

分割数据集文件中单行的格式如下：

```
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

在此格式中，`<class-index>` 是对象的类别索引，`<x1> <y1> <x2> <y2> ... <xn> <yn>` 是对象分割掩码的归一化多边形坐标（相对于图像宽度和高度，值在 `[0, 1]` 范围内）。坐标之间用空格分隔。

以下是包含两个对象（一个由 3 个点组成的分割段，一个由 5 个点组成的分割段）的单张图像的 YOLO 数据集格式示例：

```
0 0.681 0.485 0.670 0.487 0.676 0.487
1 0.504 0.000 0.501 0.004 0.498 0.004 0.493 0.010 0.492 0.0104
```

!!! tip

    - 每行的长度**不必**相等。
    - 每个分割标签必须至少有 **3 个 `(x, y)` 点**：`<class-index> <x1> <y1> <x2> <y2> <x3> <y3>`

### 数据集 YAML 格式

Ultralytics 框架使用 YAML 文件格式来定义训练分割模型的数据集和模型配置。以下是用于定义分割数据集的 YAML 格式示例：

!!! example "ultralytics/cfg/datasets/coco8-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-seg.yaml"
    ```

`train` 和 `val` 字段分别指定包含训练和验证图像的目录路径。

`names` 是一个类别名称字典。名称的顺序应与 YOLO 数据集文件中对象类别索引的顺序相匹配。

## 使用方法

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

## 支持的数据集

Ultralytics YOLO 支持多种用于实例分割任务的数据集。以下是最常用的数据集列表：

- [Carparts-seg](carparts-seg.md)：专注于汽车零部件分割的专用数据集，非常适合汽车应用。包含多种车辆，带有各个汽车组件的详细标注。
- [COCO](coco.md)：用于[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和字幕的综合数据集，包含超过 20 万张各类别的标注图像。
- [COCO8-seg](coco8-seg.md)：COCO 的紧凑型 8 图像子集，专为快速测试分割模型训练而设计，适用于 `ultralytics` 仓库中的 CI 检查和工作流验证。
- [COCO128-seg](coco128-seg.md)：用于[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)任务的小型数据集，包含带有分割标注的 128 张 COCO 图像子集。
- [Crack-seg](crack-seg.md)：专为各类表面裂缝分割而定制的数据集。对基础设施维护和质量控制至关重要，提供详细图像用于训练识别结构性薄弱点的模型。
- [Package-seg](package-seg.md)：专用于不同类型包装材料和形状分割的数据集。对物流和仓库自动化特别有用，有助于开发包裹处理和分拣系统。

### 添加自己的数据集

如果您有自己的数据集并希望将其用于使用 Ultralytics YOLO 格式训练分割模型，请确保它遵循上述"Ultralytics YOLO 格式"部分中指定的格式。将标注转换为所需格式，并在 YAML 配置文件中指定路径、类别数量和类别名称。请将 `images/` 和 `labels/` 作为独立的文件夹保持在同一级别，并具有匹配的子文件夹结构；将标签 `.txt` 文件放在图像文件夹中会导致模型遗漏标签。

## 转换标签格式

### COCO 数据集格式转换为 YOLO 格式

您可以使用以下代码片段轻松地将常用 COCO 数据集格式的标签转换为 YOLO 格式：

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)
        ```

此转换工具可用于将 COCO 数据集或任何 COCO 格式的数据集转换为 Ultralytics YOLO 格式。

请务必仔细检查您要使用的数据集是否与您的模型兼容，并遵循必要的格式规范。格式正确的数据集对于训练成功的分割模型至关重要。

## 自动标注

[自动标注](https://www.ultralytics.com/annotate)是一项重要功能，允许您使用预训练的检测模型生成分割数据集。它使您能够快速准确地为大量图像添加标注，而无需手动标注，从而节省时间和精力。

### 使用检测模型生成分割数据集

要使用 Ultralytics 框架对数据集进行自动标注，您可以使用以下所示的 `auto_annotate` 函数：

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolo26x.pt", sam_model="sam_b.pt")
        ```

{% include "macros/sam-auto-annotate.md" %}

`auto_annotate` 函数接受图像路径，以及用于指定预训练检测模型（如 [YOLO26](../../models/yolo26.md)、[YOLO11](../../models/yolo11.md) 或其他[模型](../../models/index.md)）和分割模型（如 [SAM](../../models/sam.md)、[SAM2](../../models/sam-2.md) 或 [MobileSAM](../../models/mobile-sam.md)）的可选参数，以及运行模型的设备和保存标注结果的输出目录。

通过利用预训练模型的强大功能，自动标注可以显著减少创建高质量分割数据集所需的时间和精力。此功能对于处理大量图像集合的研究人员和开发人员特别有用，因为它允许他们专注于模型开发和评估，而不是手动标注。

### 可视化数据集标注

在训练模型之前，可视化数据集标注通常很有帮助，以确保它们是正确的。Ultralytics 为此提供了一个实用函数：

```python
from ultralytics.data.utils import visualize_image_annotations

label_map = {  # 定义包含所有标注类别标签的标签映射。
    0: "person",
    1: "car",
}

# 可视化
visualize_image_annotations(
    "path/to/image.jpg",  # 输入图像路径。
    "path/to/annotations.txt",  # 图像的标注文件路径。
    label_map,
)
```

此函数绘制边界框，用类别名称标记对象，并调整文本颜色以提高可读性，帮助您在训练前识别和纠正任何标注错误。

### 将分割掩码转换为 YOLO 格式

如果您有二进制格式的分割掩码，可以使用以下方法将其转换为 YOLO 分割格式：

```python
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# 对于具有 80 个类别的 COCO 等数据集
convert_segment_masks_to_yolo_seg(masks_dir="path/to/masks_dir", output_dir="path/to/output_dir", classes=80)
```

此实用程序将二进制掩码图像转换为 YOLO 分割格式，并将其保存在指定的输出目录中。

## 常见问题

### Ultralytics YOLO 支持哪些用于实例分割的数据集格式？

Ultralytics YOLO 支持多种用于实例分割的数据集格式，主要格式是其自有的 Ultralytics YOLO 格式。数据集中的每张图像都需要一个对应的文本文件，其中对象信息被分割成多行（每个对象一行），列出类别索引和归一化边界坐标。有关 YOLO 数据集格式的更详细说明，请访问[实例分割数据集概述](#实例分割数据集概述)。

### 如何将 COCO 数据集标注转换为 YOLO 格式？

使用 Ultralytics 工具将 COCO 格式标注转换为 YOLO 格式非常简单。您可以使用 `ultralytics.data.converter` 模块中的 `convert_coco` 函数：

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations/", use_segments=True)
```

此脚本将您的 COCO 数据集标注转换为所需的 YOLO 格式，使其适用于训练 YOLO 模型。更多详情，请参阅[转换标签格式](#coco-数据集格式转换为-yolo-格式)。

### 如何为训练 Ultralytics YOLO 模型准备 YAML 文件？

要为使用 Ultralytics 训练 YOLO 模型准备 YAML 文件，您需要定义数据集路径和类别名称。以下是一个 YAML 配置示例：

```yaml
--8<-- "ultralytics/cfg/datasets/coco8-seg.yaml"
```

请根据您的数据集更新路径和类别名称。更多信息，请查看[数据集 YAML 格式](#数据集-yaml-格式)部分。

### Ultralytics YOLO 中的自动标注功能是什么？

Ultralytics YOLO 中的自动标注允许您使用预训练的检测模型为数据集生成分割标注。这大大减少了手动标注的需求。您可以按以下方式使用 `auto_annotate` 函数：

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolo26x.pt", sam_model="sam_b.pt")  # 或 sam_model="mobile_sam.pt"
```

此函数自动化了标注过程，使其更快、更高效。更多详情，请探索[自动标注参考](https://docs.ultralytics.com/reference/data/annotator#ultralytics.data.annotator.auto_annotate)。
