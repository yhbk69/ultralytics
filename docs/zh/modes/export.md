---
comments: true
description: 了解如何将 YOLO26 模型导出为 ONNX、TensorRT、CoreML 等多种格式，实现最大兼容性和性能。
keywords: YOLO26, 模型导出, ONNX, TensorRT, CoreML, Ultralytics, AI, 机器学习, 推理, 部署
---

# Ultralytics YOLO 模型导出

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统与集成">

## 简介

训练模型的最终目标是将其部署到实际应用中。Ultralytics YOLO26 的导出模式提供了多种灵活的选项，可将训练好的模型导出为不同格式，使其能够在各种平台和设备上部署。本综合指南将带你深入了解模型导出的细节，展示如何实现最大兼容性和性能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/KGHYU-MKYeE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何将 Ultralytics YOLO26 导出为不同格式进行部署 | ONNX、TensorRT、CoreML 🚀
</p>

## 为什么选择 YOLO26 的导出模式？

- **多样性：** 支持导出为多种格式，包括 [ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md)、[CoreML](../integrations/coreml.md) 等。
- **性能：** 使用 TensorRT 可获得高达 5 倍的 GPU 加速，使用 ONNX 或 [OpenVINO](../integrations/openvino.md) 可获得高达 3 倍的 CPU 加速。
- **兼容性：** 使你的模型可在众多硬件和软件环境中普遍部署。
- **易用性：** 简单的 CLI 和 Python API，可快速直接地导出模型。

### 导出模式的主要特性

以下是一些突出的功能：

- **一键导出：** 使用简单的命令即可导出为不同格式。
- **批量导出：** 导出支持批量推理的模型。
- **优化推理：** 导出的模型经过优化，可缩短推理时间。
- **教程视频：** 深入的指南和教程，带来流畅的导出体验。

!!! tip

    * 导出为 [ONNX](../integrations/onnx.md) 或 [OpenVINO](../integrations/openvino.md) 可获得高达 3 倍的 CPU 加速。
    * 导出为 [TensorRT](../integrations/tensorrt.md) 可获得高达 5 倍的 GPU 加速。

## 使用示例

将 YOLO26n 模型导出为 ONNX 或 TensorRT 等不同格式。完整的导出参数列表请参见下方的参数部分。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练模型
        ```

## 参数

下表详细列出了将 YOLO 模型导出为不同格式时可用的配置和选项。这些设置对于优化导出模型的性能、大小以及跨各种平台和环境的兼容性至关重要。正确的配置可确保模型在目标应用中以最佳效率为部署做好准备。

{% include "macros/export-args.md" %}

调整这些参数可以根据具体需求（如部署环境、硬件约束和性能目标）自定义导出过程。选择合适的格式和设置对于在模型大小、速度和[准确率](https://www.ultralytics.com/glossary/accuracy)之间取得最佳平衡至关重要。

## 导出格式

可用的 YOLO26 导出格式如下表所示。你可以使用 `format` 参数导出为任意格式，例如 `format='onnx'` 或 `format='engine'`。你可以直接在导出的模型上进行预测或验证，例如 `yolo predict model=yolo26n.onnx`。导出完成后，会为你展示对应模型的使用示例。模型也可以直接在 [Ultralytics 平台](https://platform.ultralytics.com)上从浏览器导出，无需任何本地设置。

{% include "macros/export-table.md" %}

## 常见问题

### 如何将 YOLO26 模型导出为 ONNX 格式？

使用 Ultralytics 将 YOLO26 模型导出为 ONNX 格式非常简单。它提供了 Python 和 CLI 两种导出方法。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练模型
        ```

有关该过程的更多详细信息（包括处理不同输入尺寸等高级选项），请参阅 [ONNX 集成指南](../integrations/onnx.md)。

### 使用 TensorRT 进行模型导出有哪些好处？

使用 TensorRT 进行模型导出可显著提升性能。导出为 TensorRT 的 YOLO26 模型可获得高达 5 倍的 GPU 加速，非常适合实时推理应用。

- **多样性：** 针对特定硬件设置优化模型。
- **速度：** 通过高级优化实现更快的推理。
- **兼容性：** 与 NVIDIA 硬件无缝集成。

要了解更多关于 TensorRT 集成的信息，请参阅 [TensorRT 集成指南](../integrations/tensorrt.md)。

### 如何在导出 YOLO26 模型时启用 INT8 量化？

INT8 量化是压缩模型并加速推理的绝佳方式，尤其适用于边缘设备。以下是如何启用 INT8 量化：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # 加载模型
        model.export(format="engine", int8=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=engine int8=True # 使用 INT8 量化导出 TensorRT 模型
        ```

INT8 量化可应用于多种格式，如 [TensorRT](../integrations/tensorrt.md)、[OpenVINO](../integrations/openvino.md) 和 [CoreML](../integrations/coreml.md)。为获得最佳量化效果，请使用 `data` 参数提供代表性[数据集](https://docs.ultralytics.com/datasets)。

### 为什么导出模型时动态输入尺寸很重要？

动态输入尺寸允许导出模型处理不同的图像尺寸，为不同用例提供灵活性并优化处理效率。在导出为 [ONNX](../integrations/onnx.md) 或 [TensorRT](../integrations/tensorrt.md) 等格式时，启用动态输入尺寸可确保模型能无缝适应不同的输入形状。

要启用此功能，请在导出时使用 `dynamic=True` 标志：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx dynamic=True
        ```

当输入尺寸可能变化时（如视频处理或处理来自不同来源的图像），动态输入尺寸尤其有用。

### 为优化模型性能需要考虑哪些关键的导出参数？

理解和配置导出参数对于优化模型性能至关重要：

- **`format:`** 导出模型的目标格式（例如 `onnx`、`torchscript`、`tensorflow`）。
- **`imgsz:`** 模型输入的期望图像尺寸（例如 `640` 或 `(height, width)`）。
- **`half:`** 启用 FP16 量化，减小模型大小并可能加速推理。
- **`optimize:`** 针对移动或受限环境应用特定优化。
- **`int8:`** 启用 INT8 量化，对[边缘 AI](https://www.ultralytics.com/blog/deploying-computer-vision-applications-on-edge-ai-devices) 部署非常有益。

针对特定硬件平台进行部署时，请考虑使用专用的导出格式，如适用于 NVIDIA GPU 的 [TensorRT](../integrations/tensorrt.md)、适用于 Apple 设备的 [CoreML](../integrations/coreml.md)，或适用于 Google Coral 设备的 [Edge TPU](../integrations/edge-tpu.md)。

### 导出的 YOLO 模型中的输出张量代表什么含义？

将 YOLO 模型导出为 ONNX 或 TensorRT 等格式时，输出张量结构取决于模型任务。理解这些输出对于自定义推理实现非常重要。

对于**检测模型**（例如 `yolo26n.pt`），输出通常是一个形状为 `(batch_size, 4 + num_classes, num_predictions)` 的单一张量，其中通道分别表示边界框坐标和每个类别的得分，`num_predictions` 取决于导出时的输入分辨率（并且可以是动态的）。

对于**分割模型**（例如 `yolo26n-seg.pt`），通常会得到两个输出：第一个张量形状为 `(batch_size, 4 + num_classes + mask_dim, num_predictions)`（边界框、类别得分和掩码系数），第二个张量形状为 `(batch_size, mask_dim, proto_h, proto_w)`，包含掩码原型，与系数一起用于生成实例掩码。尺寸取决于导出时的输入分辨率（并且可以是动态的）。

对于**姿态模型**（例如 `yolo26n-pose.pt`），输出张量通常形状为 `(batch_size, 4 + num_classes + keypoint_dims, num_predictions)`，其中 `keypoint_dims` 取决于姿态规范（例如关键点数量以及是否包含置信度），`num_predictions` 取决于导出时的输入分辨率（并且可以是动态的）。

[ONNX 推理示例](https://github.com/ultralytics/ultralytics/tree/main/examples)中的示例演示了如何为每种模型类型处理这些输出。

### 为什么使用 `half=True` 和 `end2end=True` 导出时 `output0` 仍是 FP32？

使用 `half=True`（或 `int8=True`）导出时，大多数张量会转换为较低精度以减小模型大小并提高性能。然而，当启用 `end2end=True` 时，后处理（包括类别索引）会直接嵌入到导出的计算图中。

`output0` 张量包含类别索引，这些索引在内部表示为浮点值。由于 FP16 的尾数精度有限，无法可靠地表示大于 2048 的整数值。为了避免潜在的精度损失或类别 ID 错误，`output0` 被有意保留为 FP32。

这是预期行为，同样适用于需要保持类别索引保真度的低精度或量化导出。

如果要求完全 FP16 输出，请使用 `end2end=False` 导出并在外部执行后处理。
