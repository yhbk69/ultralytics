---
comments: true
description: 探索 ImageNette 数据集，这是 ImageNet 的一个子集，包含 10 个类别，用于高效训练和评估图像分类模型。非常适合 ML 和 CV 项目。
keywords: ImageNette 数据集, ImageNet 子集, 图像分类, 机器学习, 深度学习, YOLO, 卷积神经网络, ML 数据集, 教育, 训练
---

# ImageNette 数据集

[ImageNette](https://github.com/fastai/imagenette) 数据集是更大的 [ImageNet](https://www.image-net.org/) 数据集的一个子集，但仅包含 10 个易于区分的类别。它的创建是为了为软件开发和教学提供一个更快速、更易用的 ImageNet 版本。

## 关键特性

- ImageNette 包含来自 10 个不同类别的图像，如丁鱥、英国史宾格犬、卡带播放器、链锯、教堂、法国号、垃圾车、加油站、高尔夫球、降落伞。
- 数据集由不同尺寸的彩色图像组成。
- ImageNette 广泛用于机器学习领域的训练和测试，特别是图像分类任务。

## 数据集结构

ImageNette 数据集分为两个子集：

1. **训练集**：该子集包含数千张图像，用于训练机器学习模型。每个类别的具体数量不同。
2. **验证集**：该子集包含数百张图像，用于验证和基准测试训练后的模型。同样，每个类别的具体数量不同。

## 应用

ImageNette 数据集广泛用于训练和评估[图像分类](https://www.ultralytics.com/glossary/image-classification)任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）以及各种其他机器学习算法。该数据集简单的格式和精心选择的类别使其成为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域初学者和经验丰富的从业者的便捷资源。

## 使用方法

要在 ImageNette 数据集上训练模型 100 个 epoch，标准图像大小为 224x224，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenette", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenette model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

## 示例图像与标注

ImageNette 数据集包含各种对象和场景的彩色图像，为图像分类任务提供了多样化的数据集。以下是数据集中图像的一些示例：

![ImageNette 分类数据集样本图像](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagenette-sample-image.avif)

该示例展示了 ImageNette 数据集中图像的多样性和复杂性，强调了多样化数据集对于训练鲁棒图像分类模型的重要性。

## ImageNette160 和 ImageNette320

为了更快的原型设计和训练，ImageNette 数据集还提供两种缩小尺寸的版本：[ImageNette160](https://github.com/fastai/imagenette) 和 [ImageNette320](https://github.com/fastai/imagenette)。这些数据集保持与完整 ImageNette 数据集相同的类别和结构，但图像被调整到更小的尺寸。因此，这些版本的数据集特别适用于初步模型测试，或计算资源有限的情况。

要使用这些数据集，只需在训练命令中将 'imagenette' 替换为 'imagenette160' 或 'imagenette320'。以下代码片段说明了这一点：

!!! example "使用 ImageNette160 的训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model with ImageNette160
        results = model.train(data="imagenette160", epochs=100, imgsz=160)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model with ImageNette160
        yolo classify train data=imagenette160 model=yolo26n-cls.pt epochs=100 imgsz=160
        ```

!!! example "使用 ImageNette320 的训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model with ImageNette320
        results = model.train(data="imagenette320", epochs=100, imgsz=320)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model with ImageNette320
        yolo classify train data=imagenette320 model=yolo26n-cls.pt epochs=100 imgsz=320
        ```

这些较小版本的数据集允许在开发过程中快速迭代，同时仍提供有价值且现实的图像分类任务。

## 引用与致谢

如果您在研究或开发工作中使用了 ImageNette 数据集，请适当致谢。有关 ImageNette 数据集的更多信息，请访问 [ImageNette 数据集 GitHub 页面](https://github.com/fastai/imagenette)。

## 常见问题

### ImageNette 数据集是什么？

[ImageNette 数据集](https://github.com/fastai/imagenette)是更大的 [ImageNet 数据集](https://www.image-net.org/)的一个简化子集，仅包含 10 个易于区分的类别，如丁鱥、英国史宾格犬和法国号。它的创建是为了提供一个更易于管理的数据集，用于高效训练和评估图像分类模型。该数据集特别适用于[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉中的快速软件开发和教学目的。

### 如何使用 ImageNette 数据集训练 YOLO 模型？

要在 ImageNette 数据集上训练 YOLO 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下命令。请确保已设置好 Ultralytics YOLO 环境。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenette", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenette model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

更多详情，请参阅[训练](../../modes/train.md)文档页面。

### 为什么应该使用 ImageNette 进行图像分类任务？

ImageNette 数据集具有以下几个优势：

- **快速简单**：仅包含 10 个类别，与更大的数据集相比，复杂度更低，耗时更少。
- **教学用途**：由于需要较少的计算能力和时间，非常适合学习和教授图像分类的基础知识。
- **多功能性**：广泛用于训练和基准测试各种机器学习模型，特别是图像分类。

有关模型训练和数据集管理的更多详情，请探索[数据集结构](#数据集结构)部分。

### ImageNette 数据集可以使用不同的图像尺寸吗？

可以，ImageNette 数据集还提供两种调整尺寸的版本：ImageNette160 和 ImageNette320。这些版本有助于更快的原型设计，在计算资源有限时特别有用。

!!! example "使用 ImageNette160 的训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")

        # Train the model with ImageNette160
        results = model.train(data="imagenette160", epochs=100, imgsz=160)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model with ImageNette160
        yolo classify train data=imagenette160 model=yolo26n-cls.pt epochs=100 imgsz=160
        ```

更多信息，请参阅[使用 ImageNette160 和 ImageNette320 训练](#imagenette160-和-imagenette320)。

### ImageNette 数据集有哪些实际应用？

ImageNette 数据集广泛用于：

- **教学环境**：教育机器学习初学者和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)入门。
- **软件开发**：用于图像分类模型的快速原型设计和开发。
- **深度学习研究**：评估和基准测试各种深度学习模型的性能，特别是卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN）。

探索[应用](#应用)部分以获取详细用例。