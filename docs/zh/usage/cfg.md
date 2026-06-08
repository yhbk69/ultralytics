---
comments: true
description: 通过正确的设置和超参数优化您的 Ultralytics YOLO 模型性能。了解训练、验证和预测配置。
keywords: YOLO, 超参数, 配置, 训练, 验证, 预测, 模型设置, Ultralytics, 性能优化, 机器学习
---

# 配置

YOLO 设置和超参数在模型的性能、速度和[准确度](https://www.ultralytics.com/glossary/accuracy)中起着关键作用。这些设置会影响模型在训练、验证和预测等各个阶段的行为。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=87"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>掌握 Ultralytics YOLO：配置
</p>

Ultralytics 命令使用以下语法：

!!! example

    === "CLI"

        ```bash
        yolo TASK MODE ARGS
        ```

    === "Python"

        ```python
        from ultralytics import YOLO

        # 从预训练权重文件加载 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 使用自定义 ARGS 在 MODE 中运行模型
        MODE = "predict"
        ARGS = {"source": "image.jpg", "imgsz": 640}
        getattr(model, MODE)(**ARGS)
        ```

其中：

- `TASK`（可选）是以下之一（[detect](../tasks/detect.md)、[segment](../tasks/segment.md)、[classify](../tasks/classify.md)、[pose](../tasks/pose.md)、[obb](../tasks/obb.md)）
- `MODE`（必需）是以下之一（[train](../modes/train.md)、[val](../modes/val.md)、[predict](../modes/predict.md)、[export](../modes/export.md)、[track](../modes/track.md)、[benchmark](../modes/benchmark.md)）
- `ARGS`（可选）是 `arg=value` 对，如 `imgsz=640`，用于覆盖默认值。

默认的 `ARG` 值在此页面上定义，并来自 `cfg/default.yaml` [文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)。

## 任务

Ultralytics YOLO 模型可以执行各种计算机视觉任务，包括：

- **Detect**：[目标检测](https://docs.ultralytics.com/tasks/detect)识别并定位图像或视频中的对象。
- **Segment**：[实例分割](https://docs.ultralytics.com/tasks/segment)将图像或视频划分为对应于不同对象或类别的区域。
- **Classify**：[图像分类](https://docs.ultralytics.com/tasks/classify)预测输入图像的类别标签。
- **Pose**：[姿态估计](https://docs.ultralytics.com/tasks/pose)识别对象并估计其在图像或视频中的关键点。
- **OBB**：[定向边界框](https://docs.ultralytics.com/tasks/obb)使用旋转的边界框，适用于卫星或医学影像。

| 参数     | 默认值      | 描述                                                                                                                                                                                                                                                                                                                              |
| -------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`   | `'detect'`  | 指定 YOLO 任务：`detect` 用于[目标检测](https://www.ultralytics.com/glossary/object-detection)，`segment` 用于分割，`classify` 用于分类，`pose` 用于姿态估计，`obb` 用于定向边界框。每个任务都针对图像和视频分析中的特定输出和问题进行了定制。                                                                                     |

[任务指南](../tasks/index.md){ .md-button }

## 模式

Ultralytics YOLO 模型在不同的模式下运行，每种模式设计用于模型生命周期的特定阶段：

- **Train**：在自定义数据集上训练 YOLO 模型。
- **Val**：验证训练好的 YOLO 模型。
- **Predict**：使用训练好的 YOLO 模型对新图像或视频进行预测。
- **Export**：导出 YOLO 模型以进行部署。
- **Track**：使用 YOLO 模型实时跟踪对象。
- **Benchmark**：对 YOLO 导出（ONNX、TensorRT 等）的速度和准确度进行基准测试。

| 参数     | 默认值      | 描述                                                                                                                                                                                                                                                                                                                              |
| -------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`   | `'train'`   | 指定 YOLO 模型的操作模式：`train` 用于模型训练，`val` 用于验证，`predict` 用于推理，`export` 用于转换为部署格式，`track` 用于对象跟踪，`benchmark` 用于性能评估。每种模式支持从开发到部署的不同阶段。                                                                                                                              |

[模式指南](../modes/index.md){ .md-button }

## 训练设置

YOLO 模型的训练设置包括影响模型性能、速度和[准确度](https://www.ultralytics.com/glossary/accuracy)的超参数和配置。关键设置包括[批次大小](https://www.ultralytics.com/glossary/batch-size)、[学习率](https://www.ultralytics.com/glossary/learning-rate)、动量、权重衰减。优化器、[损失函数](https://www.ultralytics.com/glossary/loss-function)和数据集组成的选择也会影响训练。调优和实验对于获得最佳性能至关重要。更多细节请参见 [Ultralytics 入口点函数](../reference/cfg/__init__.md)。

{% include "macros/train-args.md" %}

!!! info "关于批次大小设置的说明"

    `batch` 参数提供三种配置选项：

    - **固定批次大小**：使用整数指定每个批次的图像数量（例如 `batch=16`）。
    - **自动模式（60% GPU 内存）**：使用 `batch=-1` 自动调整到大约 60% CUDA 内存使用率。
    - **带利用率分数的自动模式**：设置一个分数（例如 `batch=0.70`）以根据指定的 GPU 内存使用率进行调整。

[训练指南](../modes/train.md){ .md-button }

## 预测设置

YOLO 模型的预测设置包括在推理过程中影响性能、速度和[准确度](https://www.ultralytics.com/glossary/accuracy)的超参数和配置。关键设置包括置信度阈值、[非极大值抑制 (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) 阈值和类别数量。输入数据大小、格式以及掩码等补充功能也会影响预测。调优这些设置对于获得最佳性能至关重要。

推理参数：

{% include "macros/predict-args.md" %}

可视化参数：

{% from "macros/visualization-args.md" import param_table %} {{ param_table() }}

[预测指南](../modes/predict.md){ .md-button }

## 验证设置

YOLO 模型的验证设置涉及在[验证数据集](https://www.ultralytics.com/glossary/validation-data)上评估性能的超参数和配置。这些设置影响性能、速度和[准确度](https://www.ultralytics.com/glossary/accuracy)。常见设置包括批次大小、验证频率和性能指标。验证数据集的大小和组成以及特定任务也会影响该过程。

{% include "macros/validation-args.md" %}

仔细调优和实验对于确保最佳性能以及检测和防止[过拟合](https://www.ultralytics.com/glossary/overfitting)至关重要。

[验证指南](../modes/val.md){ .md-button }

## 导出设置

YOLO 模型的导出设置包括用于在不同环境中保存或导出模型的配置。这些设置影响性能、大小和兼容性。关键设置包括导出文件格式（例如 ONNX、TensorFlow SavedModel）、目标设备（例如 CPU、GPU）以及掩码等功能。模型的任务和目标环境的限制也会影响导出过程。

{% include "macros/export-args.md" %}

周到的配置可确保导出的模型针对其用例进行了优化，并在目标环境中有效运行。

[导出指南](../modes/export.md){ .md-button }

## 解决方案设置

Ultralytics 解决方案配置设置提供了灵活性，可以自定义模型以用于对象计数、热图创建、运动跟踪、数据分析、区域跟踪、队列管理和基于区域的计数等任务。这些选项允许轻松调整以获得准确且有用的结果，以满足特定需求。

{% from "macros/solutions-args.md" import param_table %} {{ param_table() }}

[解决方案指南](../solutions/index.md){ .md-button }

## 增强设置

[数据增强](https://www.ultralytics.com/glossary/data-augmentation)技术对于提高 YOLO 模型的鲁棒性和性能至关重要，它通过向[训练数据](https://www.ultralytics.com/glossary/training-data)引入可变性，帮助模型更好地泛化到未见过的数据。下表概述了每个增强参数的目的和效果：

{% include "macros/augmentation-args.md" %}

调整这些设置以满足数据集和任务要求。尝试不同的值有助于找到最佳增强策略，以获得最佳模型性能。

[增强指南](../guides/yolo-data-augmentation.md){ .md-button }

## 日志记录、检查点和绘图设置

训练 YOLO 模型时，日志记录、检查点、绘图和文件管理非常重要：

- **日志记录**：使用 [TensorBoard](https://docs.ultralytics.com/integrations/tensorboard) 等库或写入文件来跟踪模型的进度并诊断问题。
- **检查点**：定期保存模型以恢复训练或尝试不同的配置。
- **绘图**：使用 Matplotlib 或 TensorBoard 等库可视化性能并训练进度。
- **文件管理**：组织训练期间生成的文件，例如检查点、日志文件和绘图，以便于访问和分析。

有效管理这些方面有助于跟踪进度，并使调试和优化更加容易。

| 参数       | 默认值     | 描述                                                                                                                                                                                                                                                                                                                              |
| ---------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`  | `'runs'`   | 指定用于保存训练运行的根目录。每个运行保存在单独的 subdirectory 中。                                                                                                                                                                                                                                                              |
| `name`     | `'exp'`    | 定义实验名称。如果未指定，YOLO 会为每个运行递增此名称（例如 `exp`、`exp-2`）以避免覆盖。                                                                                                                                                                                                                                          |
| `exist_ok` | `False`    | 确定是否覆盖现有的实验目录。`True` 允许覆盖；`False` 阻止覆盖。                                                                                                                                                                                                                                                                    |
| `plots`    | `True`     | 控制训练和验证图的生成和保存。设置为 `True` 以创建损失曲线、[精确度](https://www.ultralytics.com/glossary/precision)-[召回率](https://www.ultralytics.com/glossary/recall) 曲线和示例预测等图表，用于视觉跟踪性能。                                                                                                               |
| `save`     | `True`     | 启用保存训练检查点和最终模型权重。设置为 `True` 以定期保存模型状态，允许恢复训练或模型部署。                                                                                                                                                                                                                                      |

## 自定义配置文件

加载保存的 YAML 以重用完整的参数集，而无需内联传递它们。`cfg` 参数会覆盖 `default.yaml` 中的值，而同时传递的额外参数仍具有优先权。

| 参数  | 默认值 | 描述                                                                                                                             |
| ----- | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `cfg` | `None` | YAML 文件的路径，其值替换 `default.yaml` 中的条目。有关 CLI 示例，请参见[覆盖默认配置文件](cli.md#overriding-default-config-file)。 |

## FAQ

### 如何在训练期间提高 YOLO 模型的性能？

通过调优超参数（如[批次大小](https://www.ultralytics.com/glossary/batch-size)、[学习率](https://www.ultralytics.com/glossary/learning-rate)、动量、权重衰减）来提高性能。调整[数据增强](https://www.ultralytics.com/glossary/data-augmentation)设置，选择合适的优化器，并使用早停或[混合精度](https://www.ultralytics.com/glossary/mixed-precision)等技术。详情请参见[训练指南](../modes/train.md)。

### 影响 YOLO 模型准确度的关键超参数有哪些？

影响准确度的关键超参数包括：

- **批次大小 (`batch`)**：较大的批次可以稳定训练，但需要更多内存。
- **学习率 (`lr0`)**：较小的学习率提供精细调整，但收敛较慢。
- **动量 (`momentum`)**：加速梯度向量，抑制振荡。
- **图像大小 (`imgsz`)**：较大的尺寸提高准确度，但增加计算负载。

根据您的数据集和硬件调整这些参数。更多信息请参见[训练设置](#train-settings)。

### 如何设置 YOLO 模型训练的学习率？

学习率 (`lr0`) 至关重要；对于 SGD 从 `0.01` 开始，对于 [Adam 优化器](https://www.ultralytics.com/glossary/adam-optimizer)从 `0.001` 开始。监控指标并根据需要进行调整。使用余弦学习率调度器 (`cos_lr`) 或预热 (`warmup_epochs`、`warmup_momentum`)。详情请参见[训练指南](../modes/train.md)。

### YOLO 模型的默认推理设置是什么？

默认设置包括：

- **置信度阈值 (`conf=0.25`)**：检测的最小置信度。
- **IoU 阈值 (`iou=0.7`)**：用于 [非极大值抑制 (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)。
- **图像大小 (`imgsz=640`)**：调整输入图像的大小。
- **设备 (`device=None`)**：选择 CPU、GPU、Apple MPS 或华为昇腾 NPU (`npu`)。

完整概述请参见[预测设置](#predict-settings)和[预测指南](../modes/predict.md)。

### 为什么在 YOLO 模型中使用混合精度训练？

[混合精度](https://www.ultralytics.com/glossary/mixed-precision)训练 (`amp=True`) 使用 FP16 和 FP32 减少内存使用并加速训练。这对于现代 GPU 有益，允许更大的模型和更快的计算，而不会显著损失准确度。更多信息请参见[训练指南](../modes/train.md)。