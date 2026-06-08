---
comments: true
description: 了解如何使用精确指标、易用工具和自定义设置来验证您的 YOLO26 模型，以获得最佳性能。
keywords: Ultralytics, YOLO26, 模型验证, 机器学习, 目标检测, mAP 指标, Python API, CLI
---

# 使用 Ultralytics YOLO 进行模型验证

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统与集成">

## 简介

验证是[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)流程中的关键步骤，用于评估训练模型的质量。Ultralytics YOLO26 的验证模式提供了一套强大的工具和指标，用于评估您的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型性能。本指南将全面介绍如何有效使用验证模式，确保您的模型既准确又可靠。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=47"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics 模式教程：验证
</p>

## 为什么使用 Ultralytics YOLO 进行验证？

以下是使用 YOLO26 验证模式的优势：

- **精确性：** 获取准确的指标，如 mAP50、mAP75 和 mAP50-95，全面评估您的模型。
- **便捷性：** 利用内置功能，模型会自动记住训练设置，简化验证流程。
- **灵活性：** 可以使用相同或不同的数据集和图像尺寸来验证模型。
- **[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)：** 利用验证指标对模型进行微调，以获得更好的性能。

### 验证模式的主要特性

以下是 YOLO26 验证模式提供的显著功能：

- **自动设置：** 模型会记住其训练配置，使验证变得简单直接。
- **多指标支持：** 基于一系列准确率指标来评估模型。
- **CLI 和 Python API：** 根据您的偏好，可选择命令行界面或 Python API 进行验证。
- **数据兼容性：** 与训练阶段使用的数据集以及自定义数据集无缝兼容。

!!! tip

    * YOLO26 模型会自动记住其训练设置，因此您只需使用 `yolo val model=yolo26n.pt` 或 `YOLO("yolo26n.pt").val()` 即可轻松地在原始数据集上以相同图像尺寸进行验证。

## 使用示例

在 COCO8 数据集上验证训练好的 YOLO26n 模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。由于 `model` 将训练时的 `data` 和参数作为模型属性保留，因此无需额外参数。完整的验证参数列表请参见下文参数章节。

!!! warning "Windows 多进程错误"

    在 Windows 上，以脚本方式启动验证时可能会遇到 `RuntimeError`。在验证代码前添加 `if __name__ == "__main__":` 代码块即可解决。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置已记住
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # 每个类别的 mAP50-95 列表
        metrics.box.image_metrics  # 每张图片的指标字典，包含精确率、召回率、F1、TP、FP 和 FN
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo26n.pt      # 验证官方模型
        yolo detect val model=path/to/best.pt # 验证自定义模型
        ```

## YOLO 模型验证参数

在验证 YOLO 模型时，可以微调多个参数来优化评估过程。这些参数控制着输入图像尺寸、批处理和性能阈值等方面。以下是每个参数的详细说明，帮助您有效地自定义验证设置。

{% include "macros/validation-args.md" %}

这些设置在验证过程中都起着至关重要的作用，使 YOLO 模型的评估既可定制又高效。根据您的具体需求和资源调整这些参数，有助于在准确率和性能之间取得最佳平衡。

### 带参数的验证示例

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/zHxwDkYShNc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何将模型验证结果导出为 CSV、JSON、SQL、Polars DataFrame 等格式
</p>

<a href="https://github.com/ultralytics/notebooks/blob/main/notebooks/how-to-export-the-validation-results-into-dataframe-csv-sql-and-other-formats.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Google Colab 中探索模型验证和不同的导出方法"></a>

以下示例展示了在 Python 和 CLI 中使用自定义参数进行 YOLO 模型验证。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 自定义验证设置
        metrics = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.7, device="0")
        ```

    === "CLI"

        ```bash
        yolo val model=yolo26n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.7 device=0
        ```

!!! tip "导出混淆矩阵"

    您还可以使用以下代码以不同格式保存混淆矩阵结果。

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    results = model.val(data="coco8.yaml", plots=True)
    print(results.confusion_matrix.to_df())
    ```

!!! tip "每张图片的精确率、召回率和 F1"

    验证会为除分类外的所有任务存储每张图片的精确率、召回率、F1、TP、FP 和 FN 指标（IoU 阈值为 0.5）。验证完成后，可通过 `results.box.image_metrics`（检测和 OBB）、`results.seg.image_metrics`（分割）和 `results.pose.image_metrics`（姿态估计）来访问这些指标。

    ```python
    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolo26n.pt")

    # 验证并访问每张图片的指标
    results = model.val(data="coco8.yaml")

    # image_metrics 是一个字典，键为图片文件名
    print(results.box.image_metrics)
    # 输出: {'image1.jpg': {'precision': 0.85, 'recall': 0.92, 'f1': 0.88, 'tp': 17, 'fp': 3, 'fn': 1}, ...}

    # 访问特定图片的指标
    results.box.image_metrics["image1.jpg"]  # {'precision': 0.85, 'recall': 0.92, 'f1': 0.88, 'tp': 17, 'fp': 3, 'fn': 1}
    ```

    `image_metrics` 中的每个条目包含以下键：

    | 键           | 描述                                        |
    |-------------|---------------------------------------------|
    | `precision` | 该图片的精确率得分 (`tp / (tp + fp)`)。        |
    | `recall`    | 该图片的召回率得分 (`tp / (tp + fn)`)。        |
    | `f1`        | 精确率和召回率的调和平均值。                      |
    | `tp`        | 该图片的真正例数量。                             |
    | `fp`        | 该图片的假正例数量。                             |
    | `fn`        | 该图片的假负例数量。                             |

    此功能适用于检测、分割、姿态估计和 OBB 任务。

| 方法          | 返回类型                 | 描述                                         |
| ----------- | ---------------------- | -------------------------------------------- |
| `summary()` | `List[Dict[str, Any]]` | 将验证结果转换为汇总字典。                        |
| `to_df()`   | `DataFrame`            | 将验证结果返回为结构化的 Polars DataFrame。        |
| `to_csv()`  | `str`                  | 以 CSV 格式导出验证结果并返回 CSV 字符串。         |
| `to_json()` | `str`                  | 以 JSON 格式导出验证结果并返回 JSON 字符串。       |

有关更多详细信息，请参见 [`DataExportMixin` 类文档](../reference/utils/__init__.md#ultralytics.utils.__init__.DataExportMixin)。

## 常见问题

### 如何使用 Ultralytics 验证我的 YOLO26 模型？

要验证您的 YOLO26 模型，可以使用 Ultralytics 提供的验证模式。例如，使用 Python API，您可以加载模型并运行验证：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")

# 验证模型
metrics = model.val()
print(metrics.box.map)  # map50-95
```

或者，您可以使用命令行界面（CLI）：

```bash
yolo val model=yolo26n.pt
```

如需进一步自定义，您可以在 Python 和 CLI 模式中调整各种参数，如 `imgsz`、`batch` 和 `conf`。完整参数列表请参见 [YOLO 模型验证参数](#yolo-模型验证参数)章节。

### YOLO26 模型验证可以获得哪些指标？

YOLO26 模型验证提供了多个用于评估模型性能的关键指标，包括：

- mAP50（IoU 阈值为 0.5 时的平均精度均值）
- mAP75（IoU 阈值为 0.75 时的平均精度均值）
- mAP50-95（IoU 阈值从 0.5 到 0.95 的多个阈值下的平均精度均值）

使用 Python API，您可以按以下方式访问这些指标：

```python
metrics = model.val()  # 假设 `model` 已加载
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # 每个类别的 mAP50-95 列表
print(metrics.box.image_metrics)  # 每张图片的指标字典，包含精确率、召回率、F1、TP、FP 和 FN
```

要获得完整的性能评估，查看所有这些指标至关重要。更多详情请参见 [验证模式的主要特性](#验证模式的主要特性)。

### 使用 Ultralytics YOLO 进行验证有哪些优势？

使用 Ultralytics YOLO 进行验证具有以下优势：

- **[精确性](https://www.ultralytics.com/glossary/precision)：** YOLO26 提供准确的性能指标，包括 mAP50、mAP75 和 mAP50-95。
- **便捷性：** 模型会记住其训练设置，使验证变得简单直接。
- **灵活性：** 可以使用相同或不同的数据集和图像尺寸进行验证。
- **超参数调优：** 验证指标有助于微调模型以获得更好的性能。

这些优势确保您的模型得到全面评估，并可优化以获得更优结果。更多信息请参见 [为什么使用 Ultralytics YOLO 进行验证](#为什么使用-ultralytics-yolo-进行验证)章节。

### 可以使用自定义数据集验证我的 YOLO26 模型吗？

可以，您可以使用[自定义数据集](https://docs.ultralytics.com/datasets)来验证 YOLO26 模型。通过 `data` 参数指定数据集配置文件的路径，该文件应包含[验证数据](https://www.ultralytics.com/glossary/validation-data)的路径。

!!! note

    验证使用的是模型自身的类别名称，可通过 `model.names` 查看，这些名称可能与数据集配置文件中指定的名称不同。

Python 示例：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")

# 使用自定义数据集验证
metrics = model.val(data="path/to/your/custom_dataset.yaml")
print(metrics.box.map)  # map50-95
```

CLI 示例：

```bash
yolo val model=yolo26n.pt data=path/to/your/custom_dataset.yaml
```

有关验证过程中更多可自定义的选项，请参见 [带参数的验证示例](#带参数的验证示例)章节。

### 如何在 YOLO26 中将验证结果保存为 JSON 文件？

要将验证结果保存为 JSON 文件，可以在运行验证时将 `save_json` 参数设置为 `True`。这可以在 Python API 和 CLI 中实现。

Python 示例：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")

# 将验证结果保存为 JSON
metrics = model.val(save_json=True)
```

CLI 示例：

```bash
yolo val model=yolo26n.pt save_json=True
```

此功能对于进一步分析或与其他工具集成特别有用。更多详情请参见 [YOLO 模型验证参数](#yolo-模型验证参数)。
