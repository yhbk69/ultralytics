---
comments: true
description: 了解 Ultralytics YOLO 模型的 OBB 数据集格式。了解其结构、应用和格式转换，以增强你的目标检测训练。
keywords: 旋转边界框, OBB 数据集, YOLO, Ultralytics, 目标检测, 数据集格式
---

# 旋转边界框 (OBB) 数据集概述

使用旋转边界框 (OBB) 训练精确的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型需要一个完整的数据集。本指南介绍了与 Ultralytics YOLO 模型兼容的各种 OBB 数据集格式，提供了关于其结构、应用和格式转换方法的见解。

## 支持的 OBB 数据集格式

### YOLO OBB 格式

YOLO OBB 格式通过四个角点指定边界框，坐标归一化到 0 到 1 之间。它遵循以下格式：

```bash
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

在内部，YOLO 以 `xywhr` 格式处理损失和输出，该格式表示[边界框](https://www.ultralytics.com/glossary/bounding-box)的中心点 (xy)、宽度、高度和旋转角度。

<p align="center"><img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/obb-format-examples.avif" alt="旋转边界框标注格式示例"></p>

上述图像的 `*.txt` 标签文件示例，包含一个 OBB 格式的类别 `0` 对象，可能如下所示：

```bash
0 0.780811 0.743961 0.782371 0.74686 0.777691 0.752174 0.776131 0.749758
```

### 数据集 YAML 格式

Ultralytics 框架使用 YAML 文件格式定义用于训练 OBB 模型的数据集和模型配置。以下是用于定义 OBB 数据集的 YAML 格式示例：

!!! example "ultralytics/cfg/datasets/dota8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dota8.yaml"
    ```

## 使用方法

使用这些 OBB 格式训练模型：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从头创建新的 YOLO26n-OBB 模型
        model = YOLO("yolo26n-obb.yaml")

        # 在 DOTAv1 数据集上训练模型
        results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # 在 DOTAv1 数据集上训练新的 YOLO26n-OBB 模型
        yolo obb train data=DOTAv1.yaml model=yolo26n-obb.pt epochs=100 imgsz=1024
        ```

## 支持的数据集

目前，以下带有旋转边界框的数据集受支持：

- [DOTA-v1](dota-v2.md#dota-v10)：DOTA 数据集的第一个版本，提供一套全面的带有旋转边界框的航拍图像用于目标检测。
- [DOTA-v1.5](dota-v2.md#dota-v15)：DOTA 数据集的中间版本，在 DOTA-v1 的基础上提供了额外的标注和改进，以增强目标检测任务。
- [DOTA-v2](dota-v2.md#dota-v20)：DOTA（航拍图像目标检测大规模数据集）版本 2，强调从空中视角进行检测，包含 170 万个实例和 11,268 张图像的旋转边界框。
- [DOTA8](dota8.md)：完整 DOTA 数据集的一个小型 8 图像子集，适用于测试工作流和 `ultralytics` 仓库中 OBB 训练的持续集成 (CI) 检查。
- [DOTA128](dota128.md)：DOTA 数据集的 128 图像子集，所有图像都在训练文件夹中（同时用于训练和验证），在大小和多样性之间提供了良好的平衡，适用于测试 OBB 模型。

### 引入你自己的 OBB 数据集

对于那些希望引入自己带有旋转边界框的数据集的人，请确保与上述 "YOLO OBB 格式" 兼容。将你的标注转换为此所需格式，并在相应的 YAML 配置文件中详细说明路径、类别和类别名称。

## 转换标签格式

### DOTA 数据集格式转 YOLO OBB 格式

可以使用此脚本将标签从 DOTA 数据集格式转换为 YOLO OBB 格式：

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_dota_to_yolo_obb

        convert_dota_to_yolo_obb("path/to/DOTA")
        ```

此转换机制对于 DOTA 格式的数据集至关重要，可确保与 [Ultralytics YOLO](../../models/yolo26.md) OBB 格式对齐。

验证数据集与模型的兼容性并遵循必要的格式约定至关重要。结构良好的数据集对于训练高效的带旋转边界框目标检测模型至关重要。

## 常见问题

### 什么是旋转边界框 (OBB)，它们在 Ultralytics YOLO 模型中如何使用？

旋转边界框 (OBB) 是一种边界框标注类型，其中框可以旋转以更紧密地对齐被检测的物体，而不仅仅是轴对齐。这在物体可能与图像轴不对齐的航拍或卫星图像中特别有用。在 [Ultralytics YOLO](../../tasks/obb.md) 模型中，OBB 通过其四个角点以 YOLO OBB 格式表示。这使得目标检测更加准确，因为边界框可以旋转以更好地适应物体。

### 如何将现有的 DOTA 数据集标签转换为 YOLO OBB 格式以便与 Ultralytics YOLO26 一起使用？

你可以使用 Ultralytics 的 [`convert_dota_to_yolo_obb`](../../reference/data/converter.md) 函数将 DOTA 数据集标签转换为 YOLO OBB 格式。此转换确保与 Ultralytics YOLO 模型的兼容性，使你能够利用 OBB 功能进行增强的目标检测。以下是一个快速示例：

```python
from ultralytics.data.converter import convert_dota_to_yolo_obb

convert_dota_to_yolo_obb("path/to/DOTA")
```

此脚本将重新格式化你的 DOTA 标注为 YOLO 兼容格式。

### 如何在我的数据集上训练带有旋转边界框 (OBB) 的 YOLO26 模型？

训练带有 OBB 的 YOLO26 模型需要确保你的数据集是 YOLO OBB 格式，然后使用 [Ultralytics API](../../usage/python.md) 训练模型。以下是 Python 和 CLI 中的示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从头创建新的 YOLO26n-OBB 模型
        model = YOLO("yolo26n-obb.yaml")

        # 在自定义数据集上训练模型
        results = model.train(data="your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 在自定义数据集上训练新的 YOLO26n-OBB 模型
        yolo obb train data=your_dataset.yaml model=yolo26n-obb.yaml epochs=100 imgsz=640
        ```

这确保你的模型利用详细的 OBB 标注来提高检测[准确率](https://www.ultralytics.com/glossary/accuracy)。

### Ultralytics YOLO 模型目前支持哪些数据集用于 OBB 训练？

目前，Ultralytics 支持以下数据集用于 OBB 训练：

- [DOTA-v1](dota-v2.md)：DOTA 数据集的第一个版本，提供一套全面的带有旋转边界框的航拍图像用于目标检测。
- [DOTA-v1.5](dota-v2.md)：DOTA 数据集的中间版本，在 DOTA-v1 的基础上提供了额外的标注和改进，以增强目标检测任务。
- [DOTA-v2](dota-v2.md)：该数据集包含 170 万个实例，带有旋转边界框和 11,268 张图像，主要关注空中目标检测。
- [DOTA8](dota8.md)：DOTA 数据集的一个较小 8 图像子集，用于测试和[持续集成](../../help/CI.md) (CI) 检查。
- [DOTA128](dota128.md)：一个 128 图像子集，所有图像都在训练文件夹中（同时用于训练和验证），比 DOTA8 更具多样性，同时对于初始 OBB 模型开发和实验来说仍然易于管理。

这些数据集针对 OBB 提供显著优势的场景进行了定制，如航拍和卫星图像分析。

### 我可以使用自己的带有旋转边界框的数据集进行 YOLO26 训练吗？如果可以，如何操作？

是的，你可以使用自己的带有旋转边界框的数据集进行 YOLO26 训练。确保你的数据集标注转换为 YOLO OBB 格式，这涉及通过四个角点定义边界框。然后你可以创建一个 [YAML 配置文件](../../usage/cfg.md) 指定数据集路径、类别和其他必要细节。有关创建和配置数据集的更多信息，请参阅[支持的数据集](#支持的数据集)部分。
