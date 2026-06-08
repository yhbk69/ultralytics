---
comments: true
description: 探索 ImageWoof 数据集，ImageNet 中聚焦于 10 种犬种的高难度子集，旨在增强图像分类模型。在 Ultralytics 文档了解更多。
keywords: ImageWoof 数据集, ImageNet 子集, 犬种, 图像分类, 深度学习, 机器学习, Ultralytics, 训练数据集, 噪声标签
---

# ImageWoof 数据集

[ImageWoof](https://github.com/fastai/imagenette) 数据集是 [ImageNet](imagenet.md) 的一个子集，包含 10 个难以分类的类别，因为它们都是犬种。它的创建是为[图像分类](https://www.ultralytics.com/glossary/image-classification)算法提供一个更困难的任务，旨在鼓励开发更先进的模型。

## 关键特性

- ImageWoof 包含 10 种不同犬种的图像：澳大利亚梗、边境梗、萨摩耶、比格犬、西施犬、英国猎狐犬、罗得西亚脊背犬、澳洲野犬、金毛寻回犬和古英国牧羊犬。
- 数据集提供多种分辨率的图像（全尺寸、320px、160px），适应不同的计算能力和研究需求。
- 它还包含一个带噪声标签的版本，提供标签可能并不总是可靠的更真实场景。

## 数据集结构

ImageWoof 数据集结构基于犬种类别，每个品种都有自己独立的图像目录。与其他分类数据集类似，它遵循训练集和验证集分开目录的分割目录格式。

## 应用

ImageWoof 数据集广泛用于训练和评估图像分类任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，尤其是在涉及更复杂和相似类别时。该数据集的挑战在于犬种之间的细微差异，推动模型性能和泛化能力的极限。它特别适用于：

- 对细粒度类别的分类模型性能进行基准测试
- 测试模型对相似外观类别的鲁棒性
- 开发能够区分细微视觉差异的算法
- 评估从通用领域到特定领域的迁移学习能力

## 使用方法

要在 ImageWoof 数据集上训练 CNN 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 224x224，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagewoof model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

## 数据集变体

ImageWoof 数据集提供三种不同尺寸，以适应各种研究需求和计算能力：

1. **全尺寸 (imagewoof)**：这是 ImageWoof 数据集的原始版本。包含全尺寸图像，非常适合最终训练和性能基准测试。

2. **中等尺寸 (imagewoof320)**：此版本包含调整后最大边长为 320 像素的图像。适合更快的训练，同时不会显著牺牲模型性能。

3. **小尺寸 (imagewoof160)**：此版本包含调整后最大边长为 160 像素的图像。专为训练速度优先的快速原型设计和实验而设计。

要在训练中使用这些变体，只需将数据集参数中的 'imagewoof' 替换为 'imagewoof320' 或 'imagewoof160'。例如：

!!! example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # For medium-sized dataset
        model.train(data="imagewoof320", epochs=100, imgsz=224)

        # For small-sized dataset
        model.train(data="imagewoof160", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Load a pretrained model and train on the medium-sized dataset
        yolo classify train model=yolo26n-cls.pt data=imagewoof320 epochs=100 imgsz=224
        ```

需要注意的是，使用较小的图像可能会导致分类准确率方面的性能降低。然而，这是在模型开发和原型设计的早期阶段快速迭代的绝佳方式。

## 示例图像与标注

ImageWoof 数据集包含各种犬种的彩色图像，为图像分类任务提供了具有挑战性的数据集。以下是数据集中图像的一些示例：

![ImageWoof 犬种分类数据集样本](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagewoof-dataset-sample.avif)

该示例展示了 ImageWoof 数据集中不同犬种之间的细微差异和相似之处，突显了分类任务的复杂性和难度。

## 引用与致谢

如果您在研究或开发工作中使用了 ImageWoof 数据集，请确保通过链接到[官方数据集仓库](https://github.com/fastai/imagenette)来致谢数据集的创建者。

我们要感谢 [FastAI](https://www.fast.ai/) 团队创建并维护 ImageWoof 数据集，为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)研究社区提供了宝贵的资源。有关 ImageWoof 数据集的更多信息，请访问 [ImageWoof 数据集仓库](https://github.com/fastai/imagenette)。

## 常见问题

### Ultralytics 中的 ImageWoof 数据集是什么？

[ImageWoof](https://github.com/fastai/imagenette) 数据集是 ImageNet 的一个高难度子集，聚焦于 10 个特定犬种。它的创建是为了推动图像分类模型的极限，包含比格犬、西施犬和金毛寻回犬等犬种。数据集包括多种分辨率的图像（全尺寸、320px、160px），甚至还有噪声标签，用于更真实的训练场景。这种复杂性使 ImageWoof 成为开发更先进深度学习模型的理想选择。

### 如何使用 Ultralytics YOLO 在 ImageWoof 数据集上训练模型？

要使用 Ultralytics YOLO 在 ImageWoof 数据集上训练[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）模型 100 个 epoch，图像大小为 224x224，可以使用以下代码：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n-cls.pt")  # Load a pretrained model
        results = model.train(data="imagewoof", epochs=100, imgsz=224)
        ```


    === "CLI"

        ```bash
        yolo classify train data=imagewoof model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

有关可用训练参数的更多详情，请参阅[训练](../../modes/train.md)页面。

### ImageWoof 数据集有哪些版本可用？

ImageWoof 数据集提供三种尺寸：

1. **全尺寸 (imagewoof)**：包含全尺寸图像，非常适合最终训练和基准测试。
2. **中等尺寸 (imagewoof320)**：调整后最大边长为 320 像素的图像，适合更快的训练。
3. **小尺寸 (imagewoof160)**：调整后最大边长为 160 像素的图像，非常适合快速原型设计。

通过相应地替换数据集参数中的 'imagewoof' 来使用这些版本。但请注意，较小的图像可能会导致分类[准确率](https://www.ultralytics.com/glossary/accuracy)降低，但对于更快的迭代非常有用。

### ImageWoof 数据集中的噪声标签如何有益于训练？

ImageWoof 数据集中的噪声标签模拟了标签可能并不总是准确的实际条件。使用此数据训练模型有助于在图像分类任务中发展鲁棒性和泛化能力。这使模型能够有效处理在实际应用中经常遇到的模糊或错误标记的数据。

### 使用 ImageWoof 数据集的主要挑战是什么？

ImageWoof 数据集的主要挑战在于其所包含犬种之间的细微差异。由于它聚焦于 10 个密切相关的品种，区分它们需要更先进和精细调优的图像分类模型。这使得 ImageWoof 成为测试[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型能力和改进的绝佳基准。