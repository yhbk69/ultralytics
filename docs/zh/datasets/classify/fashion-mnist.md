---
comments: true
description: 探索 Fashion-MNIST 数据集，MNIST 的现代替代品，包含 70000 张 Zalando 商品图像。非常适合基准测试机器学习模型。
keywords: Fashion-MNIST, 图像分类, Zalando 数据集, 机器学习, 深度学习, CNN, 数据集概览
---

# Fashion-MNIST 数据集

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 数据集是一个 Zalando 商品图像数据库——包含 60000 个训练样本和 10000 个测试样本。每个样本是一张 28x28 的灰度图像，关联一个来自 10 个类别的标签。Fashion-MNIST 旨在作为原始 MNIST 数据集的直接替代品，用于基准测试[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)算法。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/eX5ad6udQ9Q"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 在 Fashion MNIST 数据集上进行<a href="https://www.ultralytics.com/glossary/image-classification">图像分类</a>
</p>

## 关键特性

- Fashion-MNIST 包含 60000 张训练图像和 10000 张测试图像，均为 Zalando 商品图像。
- 数据集由 28x28 像素的灰度图像组成。
- 每个像素有一个关联的像素值，表示该像素的明暗程度，数值越高表示越暗。该像素值是 0 到 255 之间的整数。
- Fashion-MNIST 广泛用于机器学习领域的训练和测试，特别是图像分类任务。

## 数据集结构

Fashion-MNIST 数据集分为两个子集：

1. **训练集**：该子集包含 60000 张图像，用于训练机器学习模型。
2. **测试集**：该子集包含 10000 张图像，用于测试和基准测试训练后的模型。

## 标签

每个训练和测试样本被分配到以下标签之一：

```
0. T恤/上衣
1. 裤子
2. 套头衫
3. 连衣裙
4. 外套
5. 凉鞋
6. 衬衫
7. 运动鞋
8. 包
9. 短靴
```

## 应用

Fashion-MNIST 数据集广泛用于训练和评估图像分类任务中的深度学习模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。该数据集简单且结构良好的格式使其成为机器学习和计算机视觉领域研究人员和从业者的重要资源。

## 使用方法

要在 Fashion-MNIST 数据集上训练 CNN 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 28x28，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="fashion-mnist", epochs=100, imgsz=28)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=fashion-mnist model=yolo26n-cls.pt epochs=100 imgsz=28
        ```

## 示例图像与标注

Fashion-MNIST 数据集包含 Zalando 商品图像的灰度图像，为图像分类任务提供了结构良好的数据集。以下是数据集中图像的一些示例：

![Fashion-MNIST 服装分类数据集样本](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/fashion-mnist-sample.avif)

该示例展示了 Fashion-MNIST 数据集中图像的多样性和复杂性，强调了多样化数据集对于训练鲁棒图像分类模型的重要性。

## 致谢

如果您在研究或开发工作中使用了 Fashion-MNIST 数据集，请通过链接到 [GitHub 仓库](https://github.com/zalandoresearch/fashion-mnist)来致谢该数据集。该数据集由 Zalando Research 提供。

## 常见问题

### Fashion-MNIST 数据集是什么？它与 MNIST 有何不同？

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 数据集是一个包含 70000 张 Zalando 商品灰度图像的集合，旨在作为原始 MNIST 数据集的现代替代品。它作为图像分类任务中机器学习模型的基准。与包含手写数字的 MNIST 不同，Fashion-MNIST 由 28x28 像素的图像组成，分为 10 个与时尚相关的类别，如 T恤/上衣、裤子和短靴。

### 如何在 Fashion-MNIST 数据集上训练 YOLO 模型？

要在 Fashion-MNIST 数据集上训练 Ultralytics YOLO 模型，可以使用 Python 和 CLI 命令。以下是一个快速入门示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-cls.pt")

        # Train the model on Fashion-MNIST
        results = model.train(data="fashion-mnist", epochs=100, imgsz=28)
        ```


    === "CLI"

        ```bash
        yolo classify train data=fashion-mnist model=yolo26n-cls.pt epochs=100 imgsz=28
        ```

有关更详细的训练参数，请参阅[训练页面](../../modes/train.md)。

### 为什么应该使用 Fashion-MNIST 数据集来基准测试我的机器学习模型？

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 数据集在[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)社区中被广泛认为是 MNIST 的鲁棒替代品。它提供了更复杂和多样化的图像集，使其成为基准测试图像分类模型的绝佳选择。该数据集的结构包含 60000 张训练图像和 10000 张测试图像，每张图像标注了 10 个类别之一，非常适合在更具挑战性的环境中评估不同机器学习算法的性能。

### 可以使用 Ultralytics YOLO 进行像 Fashion-MNIST 这样的图像分类任务吗？

可以，Ultralytics YOLO 模型可用于图像分类任务，包括涉及 Fashion-MNIST 数据集的任务。例如，YOLO26 支持各种视觉任务，如检测、分割和分类。要开始图像分类任务，请参阅[分类页面](https://docs.ultralytics.com/tasks/classify)。

### Fashion-MNIST 数据集的关键特性和结构是什么？

Fashion-MNIST 数据集分为两个主要子集：60000 张训练图像和 10000 张测试图像。每张图像是一张 28x28 像素的灰度图片，代表 10 个与时尚相关的类别之一。其简单且结构良好的格式使其成为训练和评估机器学习和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务模型的理想选择。有关数据集结构的更多详细信息，请参阅[数据集结构部分](#数据集结构)。

### 如何在研究中致谢 Fashion-MNIST 数据集的使用？

如果您在研究或开发项目中使用 Fashion-MNIST 数据集，重要的是通过链接到 [GitHub 仓库](https://github.com/zalandoresearch/fashion-mnist)来致谢。这有助于将数据归属于 Zalando Research，他们将该数据集公开供公众使用。