---
comments: true
description: 探索 MNIST 数据集，机器学习中手写数字识别的基石。了解其结构、特性和应用。
keywords: MNIST, 数据集, 手写数字, 图像分类, 深度学习, 机器学习, 训练集, 测试集, NIST
---

# MNIST 数据集

[MNIST](https://en.wikipedia.org/wiki/MNIST_database)（Modified National Institute of Standards and Technology，改进的美国国家标准与技术研究院）数据集是一个大型手写数字数据库，通常用于训练各种图像处理系统和机器学习模型。它通过"重新混合"NIST 原始数据集中的样本创建，已成为评估[图像分类](https://www.ultralytics.com/glossary/image-classification)算法性能的基准。

## 关键特性

- MNIST 包含 60000 张训练图像和 10000 张测试图像，均为手写数字。
- 数据集由 28×28 像素的灰度图像组成。
- 图像被归一化以适应 28×28 像素的[边界框](https://www.ultralytics.com/glossary/bounding-box)并进行抗锯齿处理，引入了灰度级别。
- MNIST 广泛用于机器学习领域的训练和测试，特别是图像分类任务。

## 数据集结构

MNIST 数据集分为两个子集：

1. **训练集**：该子集包含 60000 张手写数字图像，用于训练机器学习模型。
2. **测试集**：该子集包含 10000 张图像，用于测试和基准测试训练后的模型。

## 数据集访问

- **原始文件**：如果希望直接控制预处理，可以从原始 MNIST 归档下载 gzip 压缩包。
- **Ultralytics 加载器**：在命令中使用 `data="mnist"`（或使用 `data="mnist160"` 获取下面的子集），数据集将自动下载、转换为 PNG 并缓存。

数据集中的每张图像都标注了对应的数字（0-9），使其成为适合分类任务的有监督学习数据集。

## Extended MNIST (EMNIST)

Extended MNIST (EMNIST) 是 NIST 开发和发布的较新数据集，旨在成为 MNIST 的后继者。MNIST 仅包含手写数字图像，而 EMNIST 包含来自 NIST 特殊数据库 19 的所有图像，这是一个大型的手写大小写字母和数字数据库。EMNIST 中的图像通过相同的处理流程转换为相同的 28×28 像素格式。因此，适用于旧版、较小 MNIST 数据集的工具很可能无需修改即可与 EMNIST 配合使用。

## 应用

MNIST 数据集广泛用于训练和评估图像分类任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。该数据集简单且结构良好的格式使其成为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究人员和从业者的重要资源。

一些常见应用包括：

- 基准测试新的分类算法
- 用于教授机器学习概念的教学目的
- 原型设计图像识别系统
- 测试模型优化技术

## 使用方法

要在 MNIST 数据集上训练 CNN 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 28×28，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="mnist", epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=mnist model=yolo26n-cls.pt epochs=100 imgsz=28
        ```

## 示例图像与标注

MNIST 数据集包含手写数字的灰度图像，为图像分类任务提供了结构良好的数据集。以下是数据集中图像的一些示例：

![MNIST 手写数字分类数据集样本](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

该示例展示了 MNIST 数据集中手写数字的多样性和复杂性，强调了多样化数据集对于训练鲁棒图像分类模型的重要性。

## 引用与致谢

如果您在研究或开发工作中使用了 MNIST 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{lecun2010mnist,
                 title={MNIST handwritten digit database},
                 author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
                 journal={ATT Labs [Online]},
                 volume={2},
                 year={2010}
        }
        ```

我们要感谢 Yann LeCun、Corinna Cortes 和 Christopher J.C. Burges 创建并维护 MNIST 数据集，为机器学习和计算机视觉研究社区提供了宝贵的资源。有关 MNIST 数据集及其创建者的更多信息，请访问 [MNIST 数据集网站](https://en.wikipedia.org/wiki/MNIST_database)。

## MNIST160 快速测试

需要闪电般的回归测试？Ultralytics 还提供了 `data="mnist160"`，这是一个包含 160 张图像的切片，包含每个数字类别的前 8 个样本。它镜像了 MNIST 的目录结构，因此您可以在不更改任何其他参数的情况下切换数据集：

!!! example "使用 MNIST160 的训练示例"

    === "CLI"

        ```bash
        yolo classify train data=mnist160 model=yolo26n-cls.pt epochs=5 imgsz=28
        ```

在提交到完整的 70000 张图像数据集之前，使用此子集进行 CI 流水线或健全性检查。

## 常见问题

### MNIST 数据集是什么？为什么它在机器学习中很重要？

[MNIST](https://en.wikipedia.org/wiki/MNIST_database) 数据集（Modified National Institute of Standards and Technology）是一个广泛使用的手写数字集合，用于训练和测试图像分类系统。它包含 60000 张训练图像和 10000 张测试图像，所有图像均为 28×28 像素的灰度图。该数据集的重要性在于它作为评估图像分类算法的标准基准，帮助研究人员和工程师比较方法并追踪领域进展。

### 如何使用 Ultralytics YOLO 在 MNIST 数据集上训练模型？

要使用 Ultralytics YOLO 在 MNIST 数据集上训练模型，可以按照以下步骤操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="mnist", epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=mnist model=yolo26n-cls.pt epochs=100 imgsz=28
        ```

有关可用训练参数的详细列表，请参阅[训练](../../modes/train.md)页面。

### MNIST 和 EMNIST 数据集有什么区别？

MNIST 数据集仅包含手写数字，而 Extended MNIST (EMNIST) 数据集包含数字以及大小写字母。EMNIST 作为 MNIST 的后继者开发，使用相同的 28×28 像素格式处理图像，使其与为原始 MNIST 数据集设计的工具和模型兼容。EMNIST 中更广泛的字符范围使其适用于更多种类的机器学习应用。

### 可以使用 Ultralytics Platform 在 MNIST 等自定义数据集上训练模型吗？

可以，您可以使用 [Ultralytics Platform](https://docs.ultralytics.com/platform) 在 MNIST 等自定义数据集上训练模型。Ultralytics Platform 提供了用户友好的界面，用于上传数据集、训练模型和管理项目，无需大量编码知识。有关如何开始的更多详情，请查看 [Ultralytics Platform 快速入门](https://docs.ultralytics.com/platform/quickstart)页面。

### MNIST 与其他图像分类数据集相比如何？

MNIST 比许多现代数据集（如 [CIFAR-10](../classify/cifar10.md) 或 [ImageNet](../classify/imagenet.md)）更简单，非常适合初学者和快速实验。虽然更复杂的数据集提供了彩色图像和多样化对象类别的更大挑战，但 MNIST 因其简单性、小文件体积以及在机器学习算法开发中的历史意义仍然非常有价值。对于更高级的分类任务，可以考虑使用 [Fashion-MNIST](../classify/fashion-mnist.md)，它保持相同的结构但使用服装物品而非数字。