---
comments: true
description: 了解 Ultralytics YOLO 姿态估计数据集格式、支持的格式、COCO-Pose、COCO8-Pose、Tiger-Pose 以及如何添加自己的数据集。
keywords: 姿态估计, Ultralytics, YOLO 格式, COCO-Pose, COCO8-Pose, Tiger-Pose, 数据集转换, 关键点
---

# 姿态估计数据集概述

## 支持的数据集格式

### Ultralytics YOLO 格式

用于训练 YOLO 姿态模型的数据集标签格式如下：

1. **每个图像一个文本文件**：数据集中的每个图像都有一个对应的文本文件，文件名与图像文件相同，扩展名为 ".txt"。
2. **每个对象一行**：文本文件中的每一行对应图像中的一个对象实例。
3. **每行的对象信息**：每行包含以下关于对象实例的信息：
    - **对象类别索引**：表示对象类别的整数（例如，0 表示人，1 表示汽车等）。
    - **对象中心坐标**：对象中心的 x 和 y 坐标，归一化到 0 到 1 之间。
    - **对象宽度和高度**：对象的宽度和高度，归一化到 0 到 1 之间。
    - **对象关键点坐标**：对象的关键点，归一化到 0 到 1 之间。

以下是姿态估计任务的标签格式示例：

**带 2D 关键点的格式**

```
<类别索引> <x> <y> <宽度> <高度> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
```

**带 3D 关键点的格式（包含每个点的可见性）**

```
<类别索引> <x> <y> <宽度> <高度> <px1> <py1> <p1-可见性> <px2> <py2> <p2-可见性> <pxn> <pyn> <pn-可见性>
```

在此格式中，`<类别索引>` 是对象的类别索引，`<x> <y> <宽度> <高度>` 是[边界框](https://www.ultralytics.com/glossary/bounding-box)的归一化坐标，`<px1> <py1> <px2> <py2> ... <pxn> <pyn>` 是归一化的关键点坐标。可见性通道是可选的，但对于标注遮挡的数据集很有用。

### 数据集 YAML 格式

Ultralytics 框架使用 YAML 文件格式来定义姿态估计训练的数据集和模型配置。以下是用于定义姿态数据集的 YAML 格式示例：

!!! example "ultralytics/cfg/datasets/coco8-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-pose.yaml"
    ```

`train` 和 `val` 字段分别指定包含训练和验证图像的目录路径。

`names` 是类别名称的字典。名称的顺序应与 YOLO 数据集文件中对象类别索引的顺序一致。

（可选）如果关键点是对称的，则需要 `flip_idx`，例如人类或面部的左右侧。假设我们有五个面部关键点：[左眼, 右眼, 鼻子, 左嘴角, 右嘴角]，原始索引是 [0, 1, 2, 3, 4]，那么 `flip_idx` 是 [1, 0, 2, 4, 3]（只需交换左右索引，即 0-1 和 3-4，不修改其他如鼻子的索引）。

## 使用方法

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练 *.pt 模型开始训练
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## 支持的数据集

本节概述了与 Ultralytics YOLO 格式兼容并可用于训练[姿态估计](https://docs.ultralytics.com/tasks/pose)模型的数据集：

### COCO-Pose

- **描述**：COCO-Pose 是一个大规模[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和姿态估计数据集。它是流行的 COCO 数据集的子集，专注于人体姿态估计。COCO-Pose 为每个人体实例包含多个关键点。
- **标签格式**：与上述 Ultralytics YOLO 格式相同，包含人体姿态关键点。
- **类别数量**：1（人）。
- **关键点**：17 个关键点，包括鼻子、眼睛、耳朵、肩膀、肘部、手腕、臀部、膝盖和脚踝。
- **用途**：适用于训练人体姿态估计模型。
- **附加说明**：该数据集丰富多样，包含超过 20 万张标注图像。
- [了解更多关于 COCO-Pose](coco.md)

### COCO8-Pose

- **描述**：[Ultralytics](https://www.ultralytics.com/) COCO8-Pose 是一个小型但多功能的姿态检测数据集，由 COCO train 2017 集的前 8 张图像组成，其中 4 张用于训练，4 张用于验证。
- **标签格式**：与上述 Ultralytics YOLO 格式相同，包含人体姿态关键点。
- **类别数量**：1（人）。
- **关键点**：17 个关键点，包括鼻子、眼睛、耳朵、肩膀、肘部、手腕、臀部、膝盖和脚踝。
- **用途**：适用于测试和调试目标检测模型，或尝试新的检测方法。
- **附加说明**：COCO8-Pose 非常适合完整性检查和[CI 检查](https://docs.ultralytics.com/help/CI)。
- [了解更多关于 COCO8-Pose](coco8-pose.md)

### Dog-Pose

- **描述**：Dog Pose 数据集包含 6,773 张训练图像和 1,703 张测试图像，为犬类关键点估计提供了多样且广泛的资源。
- **标签格式**：遵循 Ultralytics YOLO 格式，包含针对狗解剖结构的多个关键点标注。
- **类别数量**：1（狗）。
- **关键点**：包含 24 个针对狗姿态定制的关键点，如四肢、关节和头部位置。
- **用途**：适用于训练模型以估计各种场景下的狗姿态，从研究到[实际应用](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-for-dog-pose-estimation)。
- [了解更多关于 Dog-Pose](dog-pose.md)

### Hand Keypoints

- **描述**：手部关键点姿态数据集包含近 26K 张图像，其中 18,776 张用于训练，7,992 张用于验证。
- **标签格式**：与上述 Ultralytics YOLO 格式相同，但包含 21 个人手关键点和可见性维度。
- **类别数量**：1（手）。
- **关键点**：21 个关键点。
- **用途**：非常适合人手姿态估计和[手势识别](https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11)。
- [了解更多关于 Hand Keypoints](hand-keypoints.md)

### Tiger-Pose

- **描述**：[Ultralytics](https://www.ultralytics.com/) Tiger Pose 数据集包含从 [YouTube 视频](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0)中获取的 263 张图像，其中 210 张用于训练，53 张用于验证。
- **标签格式**：与上述 Ultralytics YOLO 格式相同，包含 12 个动物姿态关键点，无可见性维度。
- **类别数量**：1（老虎）。
- **关键点**：12 个关键点。
- **用途**：非常适合动物姿态或任何非基于人体的姿态估计。
- [了解更多关于 Tiger-Pose](tiger-pose.md)

### 添加你自己的数据集

如果你有自己的数据集，并希望使用 Ultralytics YOLO 格式训练姿态估计模型，请确保它遵循上述“Ultralytics YOLO 格式”中指定的格式。将你的标注转换为所需格式，并在 YAML 配置文件中指定路径、类别数量和类别名称。

### 转换工具

Ultralytics 提供了一个方便的转换工具，可将流行的 [COCO 数据集](https://docs.ultralytics.com/datasets/detect/coco)格式标签转换为 YOLO 格式：

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/", use_keypoints=True)
        ```

此转换工具可用于将 COCO 数据集或任何 COCO 格式的数据集转换为 Ultralytics YOLO 格式。`use_keypoints` 参数指定是否在转换后的标签中包含关键点（用于姿态估计）。

## 常见问题

### 什么是 Ultralytics YOLO 姿态估计格式？

Ultralytics YOLO 姿态估计数据集格式涉及用相应的文本文件标注每个图像。文本文件的每一行存储一个对象实例的信息：

- 对象类别索引
- 对象中心坐标（归一化的 x 和 y）
- 对象宽度和高度（归一化）
- 对象关键点坐标（归一化的 pxn 和 pyn）

对于 2D 姿态，关键点包括像素坐标。对于 3D，每个关键点还有一个可见性标志。更多详情，请参阅 [Ultralytics YOLO 格式](#ultralytics-yolo-format)。

### 如何在 Ultralytics YOLO 中使用 COCO-Pose 数据集？

要在 Ultralytics YOLO 中使用 [COCO-Pose 数据集](https://docs.ultralytics.com/datasets/pose/coco)：

1. 下载数据集，并以 YOLO 格式准备标签文件。
2. 创建一个 YAML 配置文件，指定训练和验证图像的路径、关键点形状和类别名称。
3. 使用配置文件进行训练：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n-pose.pt")  # 加载预训练模型
    results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
    ```

    更多信息，请访问 [COCO-Pose](coco.md) 和 [训练](../../modes/train.md) 部分。

### 如何在 Ultralytics YOLO 中添加自己的数据集进行姿态估计？

要添加你的数据集：

1. 将你的标注转换为 Ultralytics YOLO 格式。
2. 创建一个 YAML 配置文件，指定数据集路径、类别数量和类别名称。
3. 使用配置文件训练你的模型：

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n-pose.pt")
    results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
    ```

    完整步骤，请查看 [添加你自己的数据集](#adding-your-own-dataset) 部分。

### Ultralytics YOLO 中数据集 YAML 文件的用途是什么？

Ultralytics YOLO 中的数据集 YAML 文件定义了训练的数据集和模型配置。它指定了训练、验证和测试图像的路径、关键点形状、类别名称以及其他配置选项。这种结构化格式有助于简化数据集管理和模型训练。以下是 YAML 格式示例：

```yaml
--8<-- "ultralytics/cfg/datasets/coco8-pose.yaml"
```

有关创建 YAML 配置文件的更多信息，请参阅 [数据集 YAML 格式](#dataset-yaml-format)。

### 如何将 COCO 数据集标签转换为 Ultralytics YOLO 格式以进行姿态估计？

Ultralytics 提供了一个转换工具，可将 COCO 数据集标签转换为 YOLO 格式，包括关键点信息：

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations/", use_keypoints=True)
```

此工具有助于将 COCO 数据集无缝集成到 YOLO 项目中。详情请参阅 [转换工具](#conversion-tool) 部分和[数据预处理指南](https://docs.ultralytics.com/guides/preprocessing_annotated_data)。