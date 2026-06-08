---
comments: true
description: 掌握 YOLO26 实例分割。通过详细指南和示例学习如何检测、分割并勾勒图像中的物体轮廓。
keywords: 实例分割, YOLO26, 物体检测, 图像分割, 机器学习, 深度学习, 计算机视觉, COCO 数据集, Ultralytics
model_name: yolo26n-seg
---

# 实例分割

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/instance-segmentation-examples.avif" alt="实例分割示例">

[实例分割](https://www.ultralytics.com/glossary/instance-segmentation) 比物体检测更进一步，涉及识别图像中的每个独立物体并将其与图像的其余部分分割开来。

实例分割模型的输出是一组掩码或轮廓，勾勒出图像中每个物体的形状，同时附有每个物体的类别标签和置信度分数。当你不仅需要知道物体在图像中的位置，还需要知道它们的确切形状时，实例分割非常有用。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 使用预训练 Ultralytics YOLO 模型在 Python 中运行分割。
</p>

!!! tip

    YOLO26 Segment 模型使用 `-seg` 后缀，即 `yolo26n-seg.pt`，并在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 上进行了预训练。

## [模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

此处展示了 YOLO26 预训练 Segment 模型。Detect、Segment 和 Pose 模型在 [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 数据集上预训练，而 Classify 模型在 [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) 数据集上预训练。

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) 在首次使用时从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases) 自动下载。

{% include "macros/yolo-seg-perf.md" %}

- **mAP<sup>val</sup>** 数值基于单模型单尺度在 [COCO val2017](https://cocodataset.org/) 数据集上的结果。<br>通过 `yolo val segment data=coco.yaml device=0` 复现。
- **速度** 基于 COCO 验证图像在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上的平均耗时。<br>通过 `yolo val segment data=coco.yaml batch=1 device=0|cpu` 复现。
- **参数量** 和 **FLOPs** 数值基于 `model.fuse()` 融合后的模型，此操作会合并 Conv 和 BatchNorm 层，对于端到端模型还会移除辅助的一对多检测头。预训练检查点保留了完整的训练架构，可能会显示更高的数值。

## 训练

在 COCO8-seg 数据集上以图像尺寸 640 训练 YOLO26n-seg 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)。完整参数列表请参见 [配置](../usage/cfg.md) 页面。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.yaml")  # 从 YAML 构建新模型
        model = YOLO("yolo26n-seg.pt")  # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo26n-seg.yaml").load("yolo26n-seg.pt")  # 从 YAML 构建并迁移权重

        # 训练模型
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从 YAML 构建新模型并从头开始训练
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.yaml epochs=100 imgsz=640

        # 从预训练 *.pt 模型开始训练
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640

        # 从 YAML 构建新模型，迁移预训练权重并开始训练
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.yaml pretrained=yolo26n-seg.pt epochs=100 imgsz=640
        ```

完整 `train` 模式详情请参见 [训练](../modes/train.md) 页面。分割模型也可以通过 [Ultralytics 平台](https://platform.ultralytics.com) 在云端 GPU 上训练。

### 数据集格式

YOLO 分割数据集格式的详细信息可在 [数据集指南](../datasets/segment/index.md) 中找到。要将现有数据集从其他格式（如 COCO 等）转换为 YOLO 格式，请使用 Ultralytics 的 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 工具。你也可以在 [Ultralytics 平台](https://platform.ultralytics.com) 上使用多边形工具和基于 SAM 的智能标注来创建分割掩码。

## 验证

在 COCO8-seg 数据集上验证训练好的 YOLO26n-seg 模型[准确率](https://www.ultralytics.com/glossary/accuracy)。无需传入参数，因为 `model` 会将其训练 `data` 和参数作为模型属性保留。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置已记住
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # 包含每个类别的 mAP50-95(B) 列表
        metrics.box.image_metrics  # 检测任务的逐图像指标字典，包含精确率、召回率、F1、TP、FP 和 FN
        metrics.seg.map  # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps  # 包含每个类别的 mAP50-95(M) 列表
        metrics.seg.image_metrics  # 分割任务的逐图像指标字典，包含精确率、召回率、F1、TP、FP 和 FN
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo26n-seg.pt  # 验证官方模型
        yolo segment val model=path/to/best.pt # 验证自定义模型
        ```

## 预测

使用训练好的 YOLO26n-seg 模型对图像进行预测。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 使用模型进行预测
        results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测

        # 访问结果
        for result in results:
            xy = result.masks.xy  # 多边形格式的掩码
            xyn = result.masks.xyn  # 归一化格式
            masks = result.masks.data  # 矩阵格式的掩码 (num_objects x H x W)
        ```

    === "CLI"

        ```bash
        yolo segment predict model=yolo26n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # 使用官方模型预测
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # 使用自定义模型预测
        ```

完整 `predict` 模式详情请参见 [预测](../modes/predict.md) 页面。

## 导出

将 YOLO26n-seg 模型导出为不同格式，如 ONNX、CoreML 等。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练模型

        # 导出模型
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-seg.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx # 导出自定义训练模型
        ```

YOLO26-seg 可用的导出格式见下表。你可以使用 `format` 参数导出为任意格式，例如 `format='onnx'` 或 `format='engine'`。你可以直接对导出的模型进行预测或验证，例如 `yolo predict model=yolo26n-seg.onnx`。导出完成后会显示模型的使用示例。

{% include "macros/export-table.md" %}

完整 `export` 详情请参见 [导出](../modes/export.md) 页面。

## 常见问题

### 如何在自定义数据集上训练 YOLO26 分割模型？

要在自定义数据集上训练 YOLO26 分割模型，首先需要按照 YOLO 分割格式准备数据集。你可以使用 [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) 等工具从其他格式转换数据集。数据集准备就绪后，可以使用 Python 或 CLI 命令训练模型：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练 YOLO26 分割模型
        model = YOLO("yolo26n-seg.pt")

        # 训练模型
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment train data=path/to/your_dataset.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

更多可用参数请参见 [配置](../usage/cfg.md) 页面。

### YOLO26 中[物体检测](https://www.ultralytics.com/glossary/object-detection)与实例分割的区别是什么？

物体检测通过在物体周围绘制边界框来识别和定位图像中的物体，而实例分割不仅识别边界框，还能勾勒出每个物体的精确形状。YOLO26 实例分割模型提供勾勒每个检测物体的掩码或轮廓，这在需要了解物体精确形状的任务中特别有用，例如医学影像或自动驾驶。

### 为什么使用 YOLO26 进行实例分割？

Ultralytics YOLO26 是一种以高准确率和实时性能著称的先进模型，非常适合实例分割任务。YOLO26 Segment 模型在 [COCO 数据集](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) 上进行了预训练，确保在各种物体上具有稳健的性能。此外，YOLO 支持训练、验证、预测和导出功能，并实现无缝集成，使其在研究和工业应用中都非常通用。

### 如何加载并验证预训练的 YOLO 分割模型？

加载并验证预训练的 YOLO 分割模型非常简单。以下是使用 Python 和 CLI 的方法：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n-seg.pt")

        # 验证模型
        metrics = model.val()
        print("边界框平均精度:", metrics.box.map)
        print("掩码平均精度:", metrics.seg.map)
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo26n-seg.pt
        ```

这些步骤将为你提供验证指标，例如[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP），这对评估模型性能至关重要。

### 如何将 YOLO 分割模型导出为 ONNX 格式？

将 YOLO 分割模型导出为 ONNX 格式很简单，可以使用 Python 或 CLI 命令完成：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n-seg.pt")

        # 将模型导出为 ONNX 格式
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-seg.pt format=onnx
        ```

有关导出为各种格式的更多详情，请参见 [导出](../modes/export.md) 页面。
