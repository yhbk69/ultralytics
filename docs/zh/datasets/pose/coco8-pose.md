---
comments: true
description: 探索紧凑多功能的 COCO8-Pose 数据集，用于测试和调试目标检测模型。非常适合使用 YOLO26 进行快速实验。
keywords: COCO8-Pose, Ultralytics, 姿态检测数据集, 目标检测, YOLO26, 机器学习, 计算机视觉, 训练数据
---

# COCO8-Pose 数据集

## 介绍

[Ultralytics](https://www.ultralytics.com/) COCO8-Pose 是一个小型但多功能的姿态检测数据集，由 COCO train 2017 集的前 8 张图像组成，其中 4 张用于训练，4 张用于验证。此数据集非常适合测试和调试[目标检测](https://www.ultralytics.com/glossary/object-detection)模型，或尝试新的检测方法。仅凭 8 张图像，它足够小易于管理，同时又足够多样化，可以测试训练管道是否有错误，并在训练更大数据集之前作为完整性检查。

## 数据集结构

- **图像总数**：8 张（4 张训练 / 4 张验证）。
- **类别**：1 个（人），每个标注包含 17 个关键点。
- **推荐目录布局**：`datasets/coco8-pose/images/{train,val}` 和 `datasets/coco8-pose/labels/{train,val}`，YOLO 格式关键点存储为 `.txt` 文件。

此数据集适用于 [Ultralytics 平台](https://platform.ultralytics.com/) 和 [YOLO26](https://github.com/ultralytics/ultralytics)。

## 数据集 YAML

YAML（Yet Another Markup Language）文件用于定义数据集配置。它包含数据集的路径、类别和其他相关信息。对于 COCO8-Pose 数据集，`coco8-pose.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml)。

!!! example "ultralytics/cfg/datasets/coco8-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-pose.yaml"
    ```

## 使用方法

要在 COCO8-Pose 数据集上以 640 的图像尺寸训练 YOLO26n-pose 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练 *.pt 模型开始训练
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

以下是一些来自 COCO8-Pose 数据集的图像示例及其对应标注：

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-5.avif" alt="COCO8-pose 关键点估计数据集马赛克" width="800">

- **马赛克图像**：此图像展示了一个由马赛克数据集图像组成的训练批次。马赛克是一种训练中使用的技术，将多张图像组合成一张图像，以增加每个训练批次中对象和场景的多样性。这有助于提高模型对不同对象大小、宽高比和上下文的泛化能力。

该示例展示了 COCO8-Pose 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克的好处。

## 引用与致谢

如果你在研究或开发工作中使用 COCO 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们感谢 COCO 联盟为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。有关 COCO 数据集及其创建者的更多信息，请访问 [COCO 数据集网站](https://cocodataset.org/#home)。

## 常见问题

### 什么是 COCO8-Pose 数据集，如何与 Ultralytics YOLO26 一起使用？

COCO8-Pose 数据集是一个小型、多功能的姿态检测数据集，包含来自 COCO train 2017 集的前 8 张图像，其中 4 张用于训练，4 张用于验证。它专为测试和调试目标检测模型以及尝试新的检测方法而设计。此数据集非常适合使用 [Ultralytics YOLO26](../../models/yolo26.md) 进行快速实验。有关数据集配置的更多详情，请查看[数据集 YAML 文件](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml)。

### 如何在 Ultralytics 中使用 COCO8-Pose 数据集训练 YOLO26 模型？

要在 COCO8-Pose 数据集上以 640 的图像尺寸训练 YOLO26n-pose 模型 100 个 epoch，请按照以下示例操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-pose.pt")

        # 训练模型
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

有关训练参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### 使用 COCO8-Pose 数据集有哪些好处？

COCO8-Pose 数据集提供以下好处：

- **紧凑大小**：仅 8 张图像，易于管理，非常适合快速实验。
- **多样化数据**：尽管规模小，但包含多种场景，适用于彻底的管道测试。
- **错误调试**：非常适合在扩展到更大数据集之前识别训练错误并执行完整性检查。

有关其功能和使用方法的更多信息，请参阅[数据集介绍](#介绍)部分。

### 马赛克技术在使用 COCO8-Pose 数据集训练 YOLO26 时有何益处？

COCO8-Pose 数据集示例图像中展示的马赛克技术将多张图像合成为一张，增加了每个训练批次中对象和场景的多样性。此技术有助于提高模型在不同对象大小、宽高比和上下文中的泛化能力，最终提升模型性能。请参阅[示例图像与标注](#示例图像与标注)部分查看示例图像。

### 在哪里可以找到 COCO8-Pose 数据集的 YAML 文件，如何使用？

COCO8-Pose 数据集 YAML 文件可在 <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-pose.yaml> 找到。此文件定义了数据集配置，包括路径、类别和其他相关信息。在 YOLO26 训练脚本中使用此文件，如[训练示例](#如何在-ultralytics-中使用-coco8-pose-数据集训练-yolo26-模型)部分所述。

更多常见问题和详细文档，请访问 [Ultralytics 文档](https://docs.ultralytics.com/)。