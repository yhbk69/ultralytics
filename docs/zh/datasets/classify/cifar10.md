---
comments: true
description: 探索 CIFAR-10 数据集，包含 10 个类别 60000 张彩色图像。了解其结构、应用以及如何使用 YOLO 训练模型。
keywords: CIFAR-10, 数据集, 机器学习, 计算机视觉, 图像分类, YOLO, 深度学习, 神经网络
---

# CIFAR-10 数据集

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)（Canadian Institute For Advanced Research）数据集是一个广泛用于[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉算法的图像集合。它由 CIFAR 研究所的研究人员开发，包含 60000 张 32x32 彩色图像，分为 10 个不同类别。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/fLBbyhPbWzY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 通过 CIFAR-10 数据集训练<a href="https://www.ultralytics.com/glossary/image-classification">图像分类</a>模型
</p>

## 关键特性

- CIFAR-10 数据集包含 60000 张图像，分为 10 个类别。
- 每个类别包含 6000 张图像，其中 5000 张用于训练，1000 张用于测试。
- 图像为彩色，尺寸为 32x32 像素。
- 10 个不同类别分别代表飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。
- CIFAR-10 通常用于机器学习和计算机视觉领域的训练和测试。

## 数据集结构

CIFAR-10 数据集分为两个子集：

1. **训练集**：该子集包含 50000 张图像，用于训练机器学习模型。
2. **测试集**：该子集包含 10000 张图像，用于测试和基准测试训练后的模型。

## 应用

CIFAR-10 数据集广泛用于训练和评估图像分类任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。数据集的类别多样性和彩色图像的存在使其成为机器学习和计算机视觉领域研究和开发的全面数据集。

## 使用方法

要在 CIFAR-10 数据集上训练 YOLO 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 32x32，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar10", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar10 model=yolo26n-cls.pt epochs=100 imgsz=32
        ```

## 示例图像与标注

CIFAR-10 数据集包含各种对象的彩色图像，为图像分类任务提供了结构良好的数据集。以下是数据集中图像的一些示例：

![CIFAR-10 图像分类数据集样本](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/cifar10-sample-image.avif)

该示例展示了 CIFAR-10 数据集中对象的多样性和复杂性，强调了多样化数据集对于训练鲁棒图像分类模型的重要性。

## 引用与致谢

如果您在研究或开发工作中使用了 CIFAR-10 数据集，请引用以下论文：

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

我们要感谢 Alex Krizhevsky 创建并维护 CIFAR-10 数据集，为机器学习和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)研究社区提供了宝贵的资源。有关 CIFAR-10 数据集及其创建者的更多信息，请访问 [CIFAR-10 数据集网站](https://www.cs.toronto.edu/~kriz/cifar.html)。

## 常见问题

### 如何在 CIFAR-10 数据集上训练 YOLO 模型？

要使用 Ultralytics 在 CIFAR-10 数据集上训练 YOLO 模型，可以按照 Python 和 CLI 的示例进行。以下是训练模型 100 个 epoch、图像大小为 32x32 像素的基本示例：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="cifar10", epochs=100, imgsz=32)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=cifar10 model=yolo26n-cls.pt epochs=100 imgsz=32
        ```

更多详情，请参阅模型[训练](../../modes/train.md)页面。

### CIFAR-10 数据集的关键特性是什么？

CIFAR-10 数据集包含 60000 张彩色图像，分为 10 个类别。每个类别包含 6000 张图像，其中 5000 张用于训练，1000 张用于测试。图像尺寸为 32x32 像素，涵盖以下类别：

- 飞机
- 汽车
- 鸟
- 猫
- 鹿
- 狗
- 青蛙
- 马
- 船
- 卡车

这个多样化的数据集对于在机器学习和计算机视觉领域训练图像分类模型至关重要。更多信息，请访问 CIFAR-10 的[数据集结构](#数据集结构)和[应用](#应用)部分。

### 为什么使用 CIFAR-10 数据集进行图像分类任务？

CIFAR-10 数据集因其多样性和结构而成为图像分类的优秀基准。它包含 10 个不同类别的 60000 张标注图像的均衡混合，有助于训练鲁棒且泛化的模型。它广泛用于评估深度学习模型，包括卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN）和其他机器学习算法。该数据集相对较小，适合快速实验和算法开发。在[应用](#应用)部分探索其众多应用。

### CIFAR-10 数据集的结构如何？

CIFAR-10 数据集分为两个主要子集：

1. **训练集**：包含 50000 张图像，用于训练机器学习模型。
2. **测试集**：包含 10000 张图像，用于测试和基准测试训练后的模型。

每个子集包含分类为 10 个类别的图像，标注随时可用于模型训练和评估。更多详细信息，请参阅[数据集结构](#数据集结构)部分。

### 如何在研究中引用 CIFAR-10 数据集？

如果您在研究或开发项目中使用 CIFAR-10 数据集，请务必引用以下论文：

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

感谢数据集的创建者有助于支持该领域持续的研究和开发。更多详情，请参阅[引用与致谢](#引用与致谢)部分。

### 使用 CIFAR-10 数据集有哪些实际示例？

CIFAR-10 数据集通常用于训练图像分类模型，例如卷积神经网络（CNN）和支持向量机（SVM）。这些模型可用于各种计算机视觉任务，包括[目标检测](https://www.ultralytics.com/glossary/object-detection)、[图像识别](https://www.ultralytics.com/glossary/image-recognition)和自动标记。要查看一些实际示例，请参阅[使用方法](#使用方法)部分中的代码片段。