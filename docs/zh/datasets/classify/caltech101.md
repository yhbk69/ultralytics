---
comments: true
description: 探索广泛使用的 Caltech-101 数据集，包含 101 个类别约 9000 张图像。非常适合机器学习和计算机视觉中的目标识别任务。
keywords: Caltech-101, 数据集, 目标识别, 机器学习, 计算机视觉, YOLO, 深度学习, 研究, AI
---

# Caltech-101 数据集

[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 数据集是一个广泛用于目标识别任务的数据集，包含来自 101 个对象类别的约 9000 张图像。这些类别的选择反映了各种真实世界的对象，图像本身经过精心挑选和标注，为目标识别算法提供了一个具有挑战性的基准。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/isc06_9qnM0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics Platform 通过 Caltech-256 数据集训练<a href="https://www.ultralytics.com/glossary/image-classification">图像分类</a>模型
</p>

!!! note "自动数据分割"

    提供的 Caltech-101 数据集没有预定义的训练/验证分割。但是，当您使用下面使用示例中的训练命令时，Ultralytics 框架将自动为您分割数据集。默认分割为训练集 80%，验证集 20%。

## 关键特性

- Caltech-101 数据集包含约 9000 张彩色图像，分为 101 个类别。
- 类别涵盖各种对象，包括动物、车辆、家居用品和人物。
- 每个类别的图像数量不等，每类约 40 到 800 张图像。
- 图像尺寸各异，大多数图像为中等分辨率。
- Caltech-101 广泛用于机器学习领域的训练和测试，特别是目标识别任务。

## 数据集结构

与许多其他数据集不同，Caltech-101 数据集没有正式划分为训练集和测试集。用户通常根据自身需求自行创建分割。然而，常见的做法是随机选择一部分图像用于训练（例如每类 30 张图像），其余图像用于测试。

## 应用

Caltech-101 数据集广泛用于训练和评估目标识别任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，例如[卷积神经网络](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn)（CNN）、[支持向量机](https://www.ultralytics.com/glossary/support-vector-machine-svm)（SVM）以及各种其他机器学习算法。其多样的类别和高质量的图像使其成为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究和开发的优秀数据集。

## 使用方法

要在 Caltech-101 数据集上训练 YOLO 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech101 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

## 示例图像与标注

Caltech-101 数据集包含各种对象的高质量彩色图像，为[图像分类](https://www.ultralytics.com/glossary/image-classification)任务提供了结构良好的数据集。以下是数据集中图像的一些示例：

![Caltech-101 图像分类数据集样本](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/caltech101-sample-image.avif)

该示例展示了 Caltech-101 数据集中对象的多样性和复杂性，强调了多样化数据集对于训练鲁棒目标识别模型的重要性。

## 引用与致谢

如果您在研究或开发工作中使用了 Caltech-101 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

我们要感谢 Li Fei-Fei、Rob Fergus 和 Pietro Perona 创建并维护 Caltech-101 数据集，为机器学习和计算机视觉研究社区提供了宝贵的资源。有关 Caltech-101 数据集及其创建者的更多信息，请访问 [Caltech-101 数据集网站](https://data.caltech.edu/records/mzrjq-6wc02)。

## 常见问题

### Caltech-101 数据集在机器学习中用于什么？

[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 数据集在机器学习中广泛用于目标识别任务。它包含 101 个类别的约 9000 张图像，为评估目标识别算法提供了具有挑战性的基准。研究人员利用它来训练和测试模型，特别是计算机视觉中的卷积[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)（CNN）和支持向量机（SVM）。

### 如何在 Caltech-101 数据集上训练 Ultralytics YOLO 模型？

要在 Caltech-101 数据集上训练 Ultralytics YOLO 模型，可以使用提供的代码片段。例如，训练 100 个 epoch：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech101 model=yolo26n-cls.pt epochs=100 imgsz=416
        ```

有关更详细的参数和选项，请参阅模型[训练](../../modes/train.md)页面。

### Caltech-101 数据集的关键特性是什么？

Caltech-101 数据集包括：

- 101 个类别约 9000 张彩色图像。
- 类别涵盖各种对象，包括动物、车辆和家居用品。
- 每个类别的图像数量不等，通常在 40 到 800 之间。
- 图像尺寸各异，大多数为中等分辨率。

这些特性使其成为训练和评估机器学习和计算机视觉中目标识别模型的绝佳选择。

### 为什么应该在我的研究中引用 Caltech-101 数据集？

在研究中引用 Caltech-101 数据集是对创建者贡献的认可，并为可能使用该数据集的其他人提供参考。推荐引用如下：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

引用有助于维护学术工作的完整性，并帮助同行定位原始资源。

### 可以使用 Ultralytics Platform 在 Caltech-101 数据集上训练模型吗？

可以，您可以使用 [Ultralytics Platform](https://platform.ultralytics.com) 在 Caltech-101 数据集上训练模型。Ultralytics Platform 提供了一个直观的平台，用于管理数据集、训练模型和部署模型，无需大量编码。有关详细指南，请参阅[如何使用 Ultralytics Platform 训练自定义模型](https://www.ultralytics.com/blog/how-to-train-your-custom-models-with-ultralytics-hub)博客文章。