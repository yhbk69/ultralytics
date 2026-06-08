---
comments: true
description: 了解 YOLO26 的目标检测。探索预训练模型、训练、验证、预测和导出细节，实现高效物体识别。
keywords: 目标检测, YOLO26, 预训练模型, 训练, 验证, 预测, 导出, 机器学习, 计算机视觉
---

# 目标检测（Object Detection）

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/object-detection-examples.avif" alt="YOLO 目标检测（带边界框）">

[目标检测](https://www.ultralytics.com/glossary/object-detection)是一项涉及识别图像或视频流中物体位置和类别的任务。

目标检测器的输出是一组包围图像中物体的边界框，以及每个框的类别标签和置信度分数。当你需要识别场景中感兴趣的物体，但不需要确切知道物体所在位置或其精确形状时，目标检测是一个不错的选择。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用预训练 Ultralytics YOLO 模型进行目标检测。
</p>

!!! tip

    YOLO26 Detect 模型是默认的 YOLO26 模型，即 `yolo26n.pt`，并在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上进行了预训练。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

此处展示了 YOLO26 预训练的 Detect 模型。Detect、Segment 和 Pose 模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上预训练，而 Classify 模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)自动下载。

{% include "macros/yolo-det-perf.md" %}

- **mAP<sup>val</sup>** 值针对单模型单尺度在 [COCO val2017](https://cocodataset.org/) 数据集上测得。<br>可通过 `yolo val detect data=coco.yaml device=0` 复现。
- **速度** 使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例在 COCO 验证集图像上取平均值。<br>可通过 `yolo val detect data=coco.yaml batch=1 device=0|cpu` 复现。
- **参数量（Params）** 和 **FLOPs** 值针对 `model.fuse()` 之后的融合模型，该操作会合并 Conv 和 BatchNorm 层，对于端到端模型则移除辅助的一对多检测头。预训练检查点保留完整的训练架构，可能显示更高的数值。

## 训练（Train）

在 COCO8 数据集上以图像尺寸 640 训练 YOLO26n 模型，共 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)。完整的可用参数列表请参见[配置](../usage/cfg.md)页面。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # 从 YAML 构建并迁移权重

        # 训练模型
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo detect train data=coco8.yaml model=yolo26n.yaml epochs=100 imgsz=640

        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，迁移预训练权重并开始训练
        yolo detect train data=coco8.yaml model=yolo26n.yaml pretrained=yolo26n.pt epochs=100 imgsz=640
        ```

完整的 `train` 模式详情请参见[训练](../modes/train.md)页面。检测模型也可以通过 [Ultralytics Platform](https://platform.ultralytics.com) 在云端 GPU 上进行训练。

### 数据集格式

YOLO 检测数据集格式的详细信息可在[数据集指南](../datasets/detect/index.md)中查看。如需将现有数据集从其他格式（如 COCO 等）转换为 YOLO 格式，请使用 Ultralytics 提供的 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 工具。你也可以通过 [Ultralytics Platform](https://platform.ultralytics.com) 使用 AI 辅助标注工具直接标注和管理检测数据集。

## 验证（Val）

在 COCO8 数据集上验证已训练的 YOLO26n 模型的[准确率](https://www.ultralytics.com/glossary/accuracy)。无需指定参数，因为 `model` 会将其训练 `data` 和参数作为模型属性保留。

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
        metrics.box.maps  # 包含每个类别 mAP50-95 的列表
        metrics.box.image_metrics  # 每张图像的指标字典，包含精确率、召回率、F1、TP、FP 和 FN
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo26n.pt      # 验证官方模型
        yolo detect val model=path/to/best.pt # 验证自定义模型
        ```

## 预测（Predict）

使用已训练的 YOLO26n 模型对图像进行预测。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测

        # 访问结果
        for result in results:
            xywh = result.boxes.xywh  # center-x, center-y, width, height
            xywhn = result.boxes.xywhn  # 归一化
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            xyxyn = result.boxes.xyxyn  # 归一化
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # 每个框的类别名称
            confs = result.boxes.conf  # 每个框的置信度分数
        ```

    === "CLI"

        ```bash
        yolo detect predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'      # 使用官方模型预测
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # 使用自定义模型预测
        ```

完整的 `predict` 模式详情请参见[预测](../modes/predict.md)页面。

## 导出（Export）

将 YOLO26n 模型导出为不同格式，如 ONNX、CoreML 等。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练的模型
        ```

YOLO26 可用的导出格式如下表所示。你可以使用 `format` 参数导出为任何格式，例如 `format='onnx'` 或 `format='engine'`。你也可以直接对导出的模型进行预测或验证，例如 `yolo predict model=yolo26n.onnx`。导出完成后会显示模型的使用示例。

{% include "macros/export-table.md" %}

完整的 `export` 详情请参见[导出](../modes/export.md)页面。

## 常见问题

### 我可以在不编写代码的情况下训练和部署检测模型吗？

可以。[Ultralytics Platform](https://platform.ultralytics.com) 提供了基于浏览器的工作流程，用于标注数据集、在云端 GPU 上训练检测模型，并将其部署到推理端点。请参阅[平台快速入门](../platform/quickstart.md)开始使用。

### 如何在自定义数据集上训练 YOLO26 模型？

在自定义数据集上训练 YOLO26 模型包含以下几个步骤：

1. **准备数据集**：确保你的数据集符合 YOLO 格式。有关指导，请参阅我们的[数据集指南](../datasets/detect/index.md)。
2. **加载模型**：使用 Ultralytics YOLO 库加载预训练模型或从 YAML 文件创建新模型。
3. **训练模型**：在 Python 中执行 `train` 方法，或在 CLI 中执行 `yolo detect train` 命令。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n.pt")

        # 在自定义数据集上训练模型
        model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=my_custom_dataset.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

详细的配置选项请访问[配置](../usage/cfg.md)页面。

### YOLO26 提供哪些预训练模型？

Ultralytics YOLO26 为目标检测、分割和姿态估计提供了多种预训练模型。这些模型在 COCO 数据集或 ImageNet（用于分类任务）上进行了预训练。以下是一些可用的模型：

- [YOLO26n](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)
- [YOLO26s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt)
- [YOLO26m](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt)
- [YOLO26l](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt)
- [YOLO26x](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt)

详细列表和性能指标请参见[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)部分。

### 如何验证我已训练的 YOLO 模型的准确率？

要验证已训练的 YOLO26 模型的准确率，你可以使用 Python 中的 `.val()` 方法或 CLI 中的 `yolo detect val` 命令。这将提供 mAP50-95、mAP50 等指标。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")

        # 验证模型
        metrics = model.val()
        print(metrics.box.map)  # mAP50-95
        ```

    === "CLI"

        ```bash
        yolo detect val model=path/to/best.pt
        ```

更多验证详情请访问[验证](../modes/val.md)页面。

### YOLO26 模型可以导出为哪些格式？

Ultralytics YOLO26 支持将模型导出为多种格式，如 [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange)、[TensorRT](https://www.ultralytics.com/glossary/tensorrt)、[CoreML](https://docs.ultralytics.com/integrations/coreml) 等，以确保跨不同平台和设备的兼容性。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 将模型导出为 ONNX 格式
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx
        ```

支持格式的完整列表和说明请查看[导出](../modes/export.md)页面。

### 为什么应该使用 Ultralytics YOLO26 进行目标检测？

Ultralytics YOLO26 旨在为目标检测、分割和姿态估计提供最先进的性能。以下是一些关键优势：

1. **预训练模型**：利用在 [COCO](https://docs.ultralytics.com/datasets/detect/coco) 和 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet) 等流行数据集上预训练的模型，加快开发速度。
2. **高准确率**：实现令人印象深刻的 mAP 分数，确保可靠的目标检测。
3. **速度快**：针对[实时推理](https://www.ultralytics.com/glossary/real-time-inference)进行了优化，非常适合需要快速处理的应用。
4. **灵活性**：可将模型导出为 ONNX 和 TensorRT 等多种格式，以便在多个平台上部署。

请浏览我们的[博客](https://www.ultralytics.com/blog)，了解展示 YOLO26 实际应用的使用案例和成功案例。
