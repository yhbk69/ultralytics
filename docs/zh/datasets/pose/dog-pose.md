---
comments: true
description: 探索用于姿态检测的 Dog-Pose 数据集。包含 6,773 张训练图像和 1,703 张测试图像，是训练 YOLO26 模型的稳健数据集。
keywords: Dog-Pose, Ultralytics, 姿态检测数据集, YOLO26, 机器学习, 计算机视觉, 训练数据
---

# Dog-Pose 数据集

## 介绍

[Ultralytics](https://www.ultralytics.com/) Dog-Pose 数据集是一个高质量、广泛的数据集，专门为狗关键点估计而策划。该数据集包含 6,773 张训练图像和 1,703 张测试图像，为训练稳健的姿态估计模型提供了坚实的基础。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ZhjO32tZUek"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何在 Stanford Dog Pose Estimation 数据集上训练 Ultralytics YOLO26 | 逐步教程
</p>

每张标注图像包含 24 个关键点，每个关键点有 3 个维度（x, y, 可见性），使其成为计算机视觉高级研究和开发的宝贵资源。

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-dogs.avif" alt="Ultralytics Dog-pose 展示图像" width="800">

此数据集适用于 [Ultralytics 平台](https://platform.ultralytics.com/) 和 [YOLO26](https://github.com/ultralytics/ultralytics)。

## 数据集结构

- **划分**：6,773 张训练 / 1,703 张测试图像，配有匹配的 YOLO 格式标签文件。
- **关键点**：每只狗 24 个关键点，带 `(x, y, visibility)` 三元组。
- **布局**：

    ```
    datasets/dog-pose/
    ├── images/{train,test}
    └── labels/{train,test}
    ```

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包括路径、关键点详情和其他相关信息。对于 Dog-pose 数据集，`dog-pose.yaml` 文件可在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml) 获取。

!!! example "ultralytics/cfg/datasets/dog-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dog-pose.yaml"
    ```

## 使用方法

要在 Dog-pose 数据集上以 640 的图像尺寸训练 YOLO26n-pose 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="dog-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练 *.pt 模型开始训练
        yolo pose train data=dog-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

以下是一些来自 Dog-pose 数据集的图像示例及其对应标注：

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-2-dog-pose.avif" alt="Dog 姿态估计数据集马赛克训练批次" width="800">

- **马赛克图像**：此图像展示了一个由马赛克数据集图像组成的训练批次。马赛克是一种训练中使用的技术，将多张图像组合成一张图像，以增加每个训练批次中对象和场景的多样性。这有助于提高模型对不同对象大小、宽高比和上下文的泛化能力。

该示例展示了 Dog-pose 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 引用与致谢

如果你在研究或开发工作中使用 Dog-pose 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{khosla2011fgvc,
          title={Novel dataset for Fine-Grained Image Categorization},
          author={Aditya Khosla and Nityananda Jayadevaprakash and Bangpeng Yao and Li Fei-Fei},
          booktitle={First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2011}
        }
        @inproceedings{deng2009imagenet,
          title={ImageNet: A Large-Scale Hierarchical Image Database},
          author={Jia Deng and Wei Dong and Richard Socher and Li-Jia Li and Kai Li and Li Fei-Fei},
          booktitle={IEEE Computer Vision and Pattern Recognition (CVPR)},
          year={2009}
        }
        ```

我们感谢 Stanford 团队为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。有关 Dog-pose 数据集及其创建者的更多信息，请访问 [Stanford Dogs Dataset 网站](http://vision.stanford.edu/aditya86/ImageNetDogs/)。

## 常见问题

### 什么是 Dog-pose 数据集，如何与 Ultralytics YOLO26 一起使用？

Dog-Pose 数据集包含 6,773 张训练图像和 1,703 张测试图像，标注了 24 个狗姿态关键点。它专为使用 [Ultralytics YOLO26](../../models/yolo26.md) 训练和验证模型而设计，支持动物行为分析、宠物监控和兽医研究等应用。该数据集全面的标注使其成为开发精确的犬类姿态估计模型的理想选择。

### 如何在 Ultralytics 中使用 Dog-pose 数据集训练 YOLO26 模型？

要在 Dog-pose 数据集上以 640 的图像尺寸训练 YOLO26n-pose 模型 100 个 epoch，请按照以下示例操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")

        # 训练模型
        results = model.train(data="dog-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo pose train data=dog-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

有关训练参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### 使用 Dog-pose 数据集有哪些好处？

Dog-pose 数据集提供以下好处：

**大型且多样化的数据集**：拥有超过 8,400 张图像，提供了大量涵盖各种狗姿态、品种和上下文的数据，支持稳健的模型训练和评估。

**详细的关键点标注**：每张图像包含 24 个关键点，每个关键点有 3 个维度（x, y, 可见性），为训练精确的姿态检测模型提供精确标注。

**真实世界场景**：包含来自不同环境的图像，增强模型对[宠物监控](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-for-dog-pose-estimation)和行为分析等实际应用的泛化能力。

**迁移学习优势**：该数据集与[迁移学习](https://www.ultralytics.com/blog/understanding-few-shot-zero-shot-and-transfer-learning)技术配合良好，允许在人体姿态数据集上预训练的模型适应狗特定特征。

有关其功能和使用方法的更多信息，请参阅[数据集介绍](#介绍)部分。

### 马赛克技术在使用 Dog-pose 数据集训练 YOLO26 时有何益处？

如 Dog-pose 数据集示例图像所示，马赛克技术将多张图像合并为一张合成图像，丰富了每个训练批次中对象和场景的多样性。此技术提供以下好处：

- 增加每个批次中狗姿态、大小和背景的多样性
- 提高模型在不同上下文和尺度下检测狗的能力
- 通过暴露模型于更多样的视觉模式来增强泛化能力
- 通过创建训练样本的新组合来减少过拟合

此方法可产生更稳健的模型，在真实场景中表现更好。示例图像请参阅[示例图像与标注](#示例图像与标注)部分。

### 在哪里可以找到 Dog-pose 数据集的 YAML 文件，如何使用？

Dog-pose 数据集 YAML 文件可在 <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dog-pose.yaml> 找到。此文件定义了数据集配置，包括路径、类别、关键点详情和其他相关信息。YAML 指定了 24 个关键点，每个关键点有 3 个维度，适合详细的姿态估计任务。

要在 YOLO26 训练脚本中使用此文件，只需在训练命令中引用它，如[使用方法](#使用方法)部分所示。数据集会在首次使用时自动下载，设置简单直接。

更多常见问题和详细文档，请访问 [Ultralytics 文档](https://docs.ultralytics.com/)。