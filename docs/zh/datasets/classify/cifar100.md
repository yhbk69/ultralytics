---
comments: true
description: 探索 CIFAR-100 数据集，包含 100 个类别 60000 张 32x32 彩色图像。非常适合机器学习和计算机视觉任务。
keywords: CIFAR-100, 数据集, 机器学习, 计算机视觉, 图像分类, 深度学习, YOLO, 训练, 测试, Alex Krizhevsky
---

# CIFAR-100 数据集

[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)（Canadian Institute For Advanced Research）数据集是 CIFAR-10 数据集的一个重要扩展，由 60000 张 32x32 彩色图像组成，涵盖 100 个不同类别。它由 CIFAR 研究所的研究人员开发，为更复杂的机器学习和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务提供了更具挑战性的数据集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/6bZeCs0xwO4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 CIFAR-100 上训练 Ultralytics YOLO26 | 逐步图像分类教程 🚀
</p>

## 关键特性

- CIFAR-100 数据集包含 60000 张图像，分为 100 个类别。
- 每个类别包含 600 张图像，其中 500 张用于训练，100 张用于测试。
- 图像为彩色，尺寸为 32x32 像素。
- 100 个不同类别被归入 20 个粗粒度类别，用于更高层次的分类。
- CIFAR-100 通常用于机器学习和计算机视觉领域的训练和测试。

## 数据集结构

CIFAR-100 数据集分为两个子集：

1. **训练集**：该子集包含 50000 张图像，用于训练机器学习模型。
2. **测试集**：该子集包含 10000 张图像，用于测试和基准测试训练后的模型。

## 应用

CIFAR-100 数据集广泛用于训练和评估[图像分类](https://www.ultralytics.com/glossary/image-classification)任务中的深度学习模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。数据集的类别多样性和彩色图像使其成为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉领域研究和开发的更具挑战性和全面的数据集。

## 使用方法

要在 CIFAR-100 数据集上训练 YOLO 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 32x32，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar100", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar100 model=yolo26n-cls.pt epochs=100 imgsz=32
        ```

## 示例图像与标注

CIFAR-100 数据集包含各种对象的彩色图像，为图像分类任务提供了结构良好的数据集。以下是数据集中图像的一些示例：

![CIFAR-100 图像分类数据集样本](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/cifar100-sample-image.avif)

该示例展示了 CIFAR-100 数据集中对象的多样性和复杂性，强调了多样化数据集对于训练鲁棒图像分类模型的重要性。

## 引用与致谢

如果您在研究或开发工作中使用了 CIFAR-100 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @TECHREPORT{Krizhevsky09learningmultiple,
                    author={Alex Krizhevsky},
                    title={Learning multiple layers of features from tiny images},
                    institution={},
                    year={2009}
        }
        ```

我们要感谢 Alex Krizhevsky 创建并维护 CIFAR-100 数据集，为机器学习和计算机视觉研究社区提供了宝贵的资源。有关 CIFAR-100 数据集及其创建者的更多信息，请访问 [CIFAR-100 数据集网站](https://www.cs.toronto.edu/~kriz/cifar.html)。

## 常见问题

### CIFAR-100 数据集是什么？为什么它很重要？

[CIFAR-100 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)是一个包含 60000 张 32x32 彩色图像的大型集合，分为 100 个类别。由加拿大高等研究院（CIFAR）开发，它为复杂的机器学习和计算机视觉任务提供了一个具有挑战性的理想数据集。其重要性在于类别的多样性和图像的小尺寸，使其成为使用 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26) 等框架训练和测试[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型（如卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn) CNN）的宝贵资源。

### 如何在 CIFAR-100 数据集上训练 YOLO 模型？

您可以使用 Python 或 CLI 命令在 CIFAR-100 数据集上训练 YOLO 模型。方法如下：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar100", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar100 model=yolo26n-cls.pt epochs=100 imgsz=32
        ```

有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### CIFAR-100 数据集的主要应用是什么？

CIFAR-100 数据集广泛用于训练和评估图像分类的深度学习模型。其包含 100 个类别（归入 20 个粗粒度类别）的多样化集合，为测试卷积神经网络（CNN）、支持向量机（SVM）以及各种其他机器学习方法等算法提供了具有挑战性的环境。该数据集是机器学习和计算机视觉领域研究与开发的关键资源，特别适用于[目标识别](https://docs.ultralytics.com/tasks/classify)和分类任务。

### CIFAR-100 数据集的结构如何？

CIFAR-100 数据集分为两个主要子集：

1. **训练集**：包含 50000 张图像，用于训练机器学习模型。
2. **测试集**：包含 10000 张图像，用于测试和基准测试训练后的模型。

100 个类别中每个类别包含 600 张图像，其中 500 张用于训练，100 张用于测试，使其特别适合严格的学术和工业研究。

### 在哪里可以找到 CIFAR-100 数据集的示例图像和标注？

CIFAR-100 数据集包含各种对象的彩色图像，是图像分类任务的结构化数据集。您可以参考文档页面查看[示例图像和标注](#示例图像与标注)。这些示例突出了数据集的多样性和复杂性，对于训练鲁棒的图像分类模型非常重要。有关更多适合分类任务的数据集，请查看 [Ultralytics 分类数据集概览](https://docs.ultralytics.com/datasets/classify)。