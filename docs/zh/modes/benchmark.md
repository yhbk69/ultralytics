---
comments: true
description: 了解如何使用基准测试模式评估 YOLO26 模型在真实场景中的性能。在不同导出格式中优化速度、精度和资源分配。
keywords: 模型基准测试, YOLO26, Ultralytics, 性能评估, 导出格式, ONNX, TensorRT, OpenVINO, CoreML, TensorFlow, 优化, mAP50-95, 推理时间
---

# 使用 Ultralytics YOLO 进行模型基准测试

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统与集成">

## 基准测试可视化

!!! tip "刷新浏览器"

    由于可能存在 Cookie 问题，您可能需要刷新页面才能正确查看图表。

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400"></canvas>

## 简介

模型训练和验证完成后，下一步就是评估其在各种真实场景中的表现。Ultralytics YOLO26 的基准测试模式正是为此而生，它提供了一个强大的框架，用于评估您的模型在各种导出格式下的速度和[精度](https://www.ultralytics.com/glossary/accuracy)。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/UF7pYdLSMng"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics YOLO26 模型基准测试 | 如何在不同硬件上比较模型性能？
</p>

## 为什么基准测试至关重要？

- **明智决策：** 深入了解速度与精度之间的权衡。
- **资源分配：** 了解不同导出格式在不同硬件上的表现。
- **优化：** 确定哪种导出格式能为您的特定用例提供最佳性能。
- **成本效益：** 根据基准测试结果更高效地利用硬件资源。

### 基准测试模式的关键指标

- **mAP50-95：** 用于[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和姿态估计。
- **accuracy_top5：** 用于[图像分类](https://www.ultralytics.com/glossary/image-classification)。
- **推理时间：** 每张图像处理所需的时间（毫秒）。

### 支持的导出格式

- **ONNX：** 实现最佳 CPU 性能
- **TensorRT：** 实现最高 GPU 效率
- **OpenVINO：** 针对 Intel 硬件优化
- **CoreML、TensorFlow SavedModel 等：** 满足多样化部署需求。

!!! tip

    * 导出为 ONNX 或 OpenVINO 可获得最高 3 倍的 CPU 加速。
    * 导出为 TensorRT 可获得最高 5 倍的 GPU 加速。

## 使用示例

!!! tip "推荐安装"

    在进行基准测试之前，请安装带导出依赖的 Ultralytics，以避免缺少软件包。

    ```bash
    pip install ultralytics[export]
    ```

在所有支持的导出格式（ONNX、TensorRT 等）上运行 YOLO26n 基准测试。完整的导出选项列表请参见下方的参数章节。

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在 GPU 上进行基准测试
        benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)

        # 对特定导出格式进行基准测试
        benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, format="onnx")
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolo26n.pt data='coco8.yaml' imgsz=640 half=False device=0

        # 对特定导出格式进行基准测试
        yolo benchmark model=yolo26n.pt data='coco8.yaml' imgsz=640 format=onnx
        ```

## 参数

`model`、`data`、`imgsz`、`half`、`device`、`verbose` 和 `format` 等参数使用户能够灵活地根据特定需求微调基准测试，并轻松比较不同导出格式的性能。

| 参数       | 默认值         | 描述                                                                                                                                                 |
| --------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | `None`        | 指定模型文件的路径。支持 `.pt` 和 `.yaml` 格式，例如 `"yolo26n.pt"` 用于预训练模型或配置文件。                                                                    |
| `data`    | `None`        | 定义基准测试数据集的 YAML 文件路径，通常包含[验证数据](https://www.ultralytics.com/glossary/validation-data)的路径和设置。示例：`"coco8.yaml"`。                       |
| `imgsz`   | `640`         | 模型的输入图像尺寸。可以是单个整数（表示正方形图像）或元组 `(width, height)`（表示非正方形），例如 `(640, 480)`。                                                        |
| `half`    | `False`       | 启用 FP16（半精度）推理，可减少内存使用，并可能在兼容硬件上提升速度。使用 `half=True` 启用。                                                                              |
| `int8`    | `False`       | 激活 INT8 量化以在支持的设备上进一步优化性能，尤其适用于边缘设备。使用 `int8=True` 启用。                                                                                  |
| `device`  | `None`        | 定义用于基准测试的计算设备，例如 `"cpu"` 或 `"cuda:0"`。                                                                                                        |
| `verbose` | `False`       | 控制日志输出的详细程度。使用 `verbose=True` 输出详细日志。                                                                                                        |
| `format`  | `''`          | 仅对指定的导出格式进行基准测试（例如 `format=onnx`）。留空则自动测试所有支持的格式。                                                                                       |

## 导出格式

基准测试将自动尝试在下方列出的所有可能的导出格式上运行。或者，您也可以使用 `format` 参数对特定格式进行基准测试，该参数接受下面提到的任何格式。

{% include "macros/export-table.md" %}

完整的 `export` 详情请参见[导出](../modes/export.md)页面。

## 常见问题

### 如何使用 Ultralytics 对我的 YOLO26 模型性能进行基准测试？

Ultralytics YOLO26 提供了基准测试模式，用于评估模型在不同导出格式下的性能。该模式提供关键指标洞察，如[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP50-95)、准确率和推理时间（毫秒）。您可以使用 Python 或 CLI 命令运行基准测试。例如，在 GPU 上进行基准测试：

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在 GPU 上进行基准测试
        benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolo26n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

有关基准测试参数的更多详情，请访问[参数](#arguments)章节。

### 将 YOLO26 模型导出为不同格式有什么好处？

将 YOLO26 模型导出为 [ONNX](https://docs.ultralytics.com/integrations/onnx)、[TensorRT](https://docs.ultralytics.com/integrations/tensorrt) 和 [OpenVINO](https://docs.ultralytics.com/integrations/openvino) 等不同格式，可以根据部署环境优化性能。例如：

- **ONNX：** 提供最高 3 倍的 CPU 加速。
- **TensorRT：** 提供最高 5 倍的 GPU 加速。
- **OpenVINO：** 专门针对 Intel 硬件优化。

这些格式可提升模型的速度和精度，使其在各种实际应用中更加高效。完整详情请访问[导出](../modes/export.md)页面。

### 为什么基准测试在评估 YOLO26 模型中至关重要？

对 YOLO26 模型进行基准测试至关重要，原因如下：

- **明智决策：** 了解速度与精度之间的权衡。
- **资源分配：** 评估不同硬件选项上的性能表现。
- **优化：** 确定哪种导出格式在特定用例中表现最佳。
- **成本效益：** 根据基准测试结果优化硬件使用。

mAP50-95、Top-5 准确率和推理时间等关键指标有助于做出这些评估。更多信息请参考[关键指标](#key-metrics-in-benchmark-mode)章节。

### YOLO26 支持哪些导出格式？它们各自的优势是什么？

YOLO26 支持多种导出格式，每种格式针对特定硬件和用例进行了优化：

- **ONNX：** 最适合 CPU 性能。
- **TensorRT：** 理想的 GPU 效率。
- **OpenVINO：** 针对 Intel 硬件优化。
- **CoreML 和 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)：** 适用于 iOS 和通用机器学习应用。

完整的支持格式列表及其各自优势，请参见[支持的导出格式](#supported-export-formats)章节。

### 我可以使用哪些参数来微调 YOLO26 基准测试？

运行基准测试时，可以自定义以下几个参数以满足特定需求：

- **model：** 模型文件路径（例如 `"yolo26n.pt"`）。
- **data：** 定义数据集的 YAML 文件路径（例如 `"coco8.yaml"`）。
- **imgsz：** 输入图像尺寸，可以是单个整数或元组。
- **half：** 启用 FP16 推理以获得更好性能。
- **int8：** 为边缘设备激活 INT8 量化。
- **device：** 指定计算设备（例如 `"cpu"`、`"cuda:0"`）。
- **verbose：** 控制日志详细级别。

完整的参数列表请参见[参数](#arguments)章节。
