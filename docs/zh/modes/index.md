---
comments: true
description: 探索 Ultralytics YOLO26 的多种模式，包括训练、验证、预测、导出、跟踪和基准测试，最大化模型性能与效率。
keywords: Ultralytics, YOLO26, 机器学习, 模型训练, 验证, 预测, 导出, 跟踪, 基准测试, 目标检测
---

# Ultralytics YOLO26 模式

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO 生态系统与集成">

## 简介

Ultralytics YOLO26 不仅仅是一个目标检测模型；它是一个多功能框架，旨在覆盖[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型的整个生命周期——从数据摄入和模型训练到验证、部署以及真实世界的跟踪。每种模式都有其特定的用途，并经过精心设计，为不同的任务和使用场景提供所需的灵活性和效率。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics 模式教程：训练、验证、预测、导出与基准测试。
</p>

### 模式概览

理解 Ultralytics YOLO26 支持的不同**模式**对于充分发挥模型潜力至关重要：

- **训练（Train）模式**：在自定义或预加载的数据集上微调模型。
- **验证（Val）模式**：训练后的检查点，用于验证模型性能。
- **预测（Predict）模式**：在真实数据上释放模型的预测能力。
- **导出（Export）模式**：使你的[模型部署](https://www.ultralytics.com/glossary/model-deployment)准备好多种格式。
- **跟踪（Track）模式**：将目标检测模型扩展到实时跟踪应用中。
- **基准测试（Benchmark）模式**：分析模型在不同部署环境中的速度和准确性。

本综合指南旨在为你提供每种模式的概述和实用见解，帮助你充分发挥 YOLO26 的潜力。

## [训练](train.md)

训练模式用于在自定义数据集上训练 YOLO26 模型。在此模式下，模型使用指定的数据集和超参数进行训练。训练过程涉及优化模型参数，使其能够准确预测图像中物体的类别和位置。训练对于创建能够识别特定应用相关物体的模型至关重要。

[训练示例](train.md){ .md-button }

## [验证](val.md)

验证模式用于在训练后验证 YOLO26 模型。在此模式下，模型在验证集上进行评估，以衡量其准确性和泛化性能。验证有助于识别[过拟合](https://www.ultralytics.com/glossary/overfitting)等潜在问题，并提供[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）等指标来量化模型性能。该模式对于调整超参数和提升整体模型效果至关重要。

[验证示例](val.md){ .md-button }

## [预测](predict.md)

预测模式用于使用训练好的 YOLO26 模型对新图像或视频进行预测。在此模式下，模型从检查点文件加载，用户可以提供图像或视频来执行推理。模型在输入媒体中识别并定位物体，使其准备好用于实际应用。预测模式是将训练好的模型应用于解决实际问题的入口。

[预测示例](predict.md){ .md-button }

## [导出](export.md)

导出模式用于将 YOLO26 模型转换为适合在不同平台和设备上部署的格式。此模式将 PyTorch 模型转换为优化格式，如 ONNX、TensorRT 或 CoreML，使其能够在生产环境中部署。导出对于将模型集成到各种软件应用或硬件设备中至关重要，通常能带来显著的性能提升。

[导出示例](export.md){ .md-button }

## [跟踪](track.md)

跟踪模式扩展了 YOLO26 的目标检测能力，使其能够跨视频帧或实时流跟踪物体。该模式对于需要持久化物体识别的应用特别有价值，例如[监控系统](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai)或[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)。跟踪模式实现了 ByteTrack 等复杂算法，即使在物体暂时从视野中消失时也能保持其跨帧身份。

[跟踪示例](track.md){ .md-button }

## [基准测试](benchmark.md)

基准测试模式分析 YOLO26 各种导出格式的速度和准确性。此模式提供了关于模型大小、准确性（检测任务的 mAP50-95 或分类任务的 accuracy_top5）以及不同格式（如 ONNX、[OpenVINO](https://docs.ultralytics.com/integrations/openvino) 和 TensorRT）下推理时间的全面指标。基准测试帮助你根据部署环境中对速度和准确性的具体要求，选择最佳的导出格式。

[基准测试示例](benchmark.md){ .md-button }

## 常见问题

### 如何使用 Ultralytics YOLO26 训练自定义[目标检测](https://www.ultralytics.com/glossary/object-detection)模型？

使用 Ultralytics YOLO26 训练自定义目标检测模型需要用到训练模式。你需要一个以 [YOLO 格式](../datasets/detect/index.md#ultralytics-yolo-format)组织的数据集，包含图像和对应的标注文件。使用以下命令启动训练过程：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO 模型（可选择 n、s、m、l 或 x 版本）
        model = YOLO("yolo26n.pt")

        # 在自定义数据集上开始训练
        model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从命令行训练 YOLO 模型
        yolo detect train data=path/to/dataset.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

如需更详细的说明，可参阅 [Ultralytics 训练指南](../modes/train.md)。

### Ultralytics YOLO26 使用哪些指标来验证模型性能？

Ultralytics YOLO26 在验证过程中使用多种指标来评估模型性能，包括：

- **mAP（平均精度均值）**：评估目标检测的准确性。
- **IoU（交并比）**：衡量预测边界框与真实边界框之间的重叠程度。
- **[精确率](https://www.ultralytics.com/glossary/precision)和[召回率](https://www.ultralytics.com/glossary/recall)**：精确率衡量真正例检测数占总正例检测数的比例，召回率衡量真正例检测数占实际正例总数的比例。

你可以运行以下命令开始验证：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练或自定义的 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 在数据集上运行验证
        model.val(data="path/to/validation.yaml")
        ```

    === "CLI"

        ```bash
        # 从命令行验证 YOLO 模型
        yolo val model=yolo26n.pt data=path/to/validation.yaml
        ```

更多详情请参阅[验证指南](../modes/val.md)。

### 如何导出我的 YOLO26 模型以便部署？

Ultralytics YOLO26 提供导出功能，可将训练好的模型转换为多种部署格式，如 ONNX、TensorRT、CoreML 等。使用以下示例导出模型：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载训练好的 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 将模型导出为 ONNX 格式（可根据需要指定其他格式）
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        # 从命令行将 YOLO 模型导出为 ONNX 格式
        yolo export model=yolo26n.pt format=onnx
        ```

每种导出格式的详细步骤可在[导出指南](../modes/export.md)中找到。

### Ultralytics YOLO26 中基准测试模式的目的是什么？

Ultralytics YOLO26 中的基准测试模式用于分析各种导出格式（如 ONNX、TensorRT 和 OpenVINO）的速度和[准确性](https://www.ultralytics.com/glossary/accuracy)。它提供模型大小、目标检测的 `mAP50-95` 以及不同硬件设置下的推理时间等指标，帮助你选择最适合部署需求的格式。

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在 GPU（设备 0）上运行基准测试
        # 你可以根据需要调整模型、数据集、图像大小和精度等参数
        benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        # 从命令行对 YOLO 模型进行基准测试
        # 根据你的具体使用场景调整参数
        yolo benchmark model=yolo26n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

更多详情请参阅[基准测试指南](../modes/benchmark.md)。

### 如何使用 Ultralytics YOLO26 进行实时目标跟踪？

使用 Ultralytics YOLO26 的跟踪模式可以实现实时目标跟踪。该模式扩展了目标检测能力，使其能够跨视频帧或实时源跟踪物体。使用以下示例启用跟踪：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 开始跟踪视频中的物体
        # 你也可以使用实时视频流或摄像头输入
        model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # 从命令行对视频进行目标跟踪
        # 你可以指定不同的源，如摄像头（0）或 RTSP 流
        yolo track model=yolo26n.pt source=path/to/video.mp4
        ```

深入说明请访问[跟踪指南](../modes/track.md)。
