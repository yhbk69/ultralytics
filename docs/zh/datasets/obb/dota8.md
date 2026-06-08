---
comments: true
description: 探索 DOTA8 数据集——一个适用于使用 Ultralytics YOLO26 测试和调试目标检测模型的小型多功能旋转目标检测数据集。
keywords: DOTA8 数据集, Ultralytics, YOLO26, 目标检测, 调试, 训练模型, 旋转目标检测, 数据集 YAML
---

# DOTA8 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) DOTA8 是一个小型但多功能的旋转[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集，由 DOTAv1 拆分集的前 8 张图像组成，4 张用于训练，4 张用于验证。该数据集非常适合测试和调试目标检测模型，或尝试新的检测方法。凭借 8 张图像，它小到易于管理，但又足够多样化，可以测试训练流程中的错误，并在训练更大数据集之前充当健全性检查。

## 数据集结构

- **图像**：来自 DOTAv1 的 8 张航拍瓦片（4 张训练，4 张验证）。
- **类别**：继承 DOTAv1 的 15 个类别，如飞机、船和大型车辆。
- **标签**：YOLO 格式的旋转边界框，以 `.txt` 文件保存在每张图像旁边。
- **推荐布局**：

    ```
    datasets/dota8/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    ```

该数据集适用于 [Ultralytics 平台](https://platform.ultralytics.com/)和 [YOLO26](https://github.com/ultralytics/ultralytics)。

## 数据集 YAML

YAML 文件用于定义数据集配置，包含数据集路径、类别和其他相关信息。DOTA8 数据集的 `dota8.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml)。

!!! example "ultralytics/cfg/datasets/dota8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/dota8.yaml"
    ```

## 使用方法

要在 DOTA8 数据集上以 640 的图像大小训练 YOLO26n-obb 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-obb.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

以下是 DOTA8 数据集中的一些图像示例及其对应标注：

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch.avif" alt="DOTA8 旋转边界框数据集训练马赛克" width="800">

- **马赛克图像**：该图像展示了一个由马赛克数据集图像组成的训练批次。马赛克是训练中使用的一种技术，将多张图像合并为一张，以增加每个训练批次中物体和场景的多样性。这有助于提高模型在不同物体大小、宽高比和上下文中的泛化能力。

该示例展示了 DOTA8 数据集中图像的多样性和复杂性，以及在训练过程中使用马赛克技术的好处。

## 引用与致谢

如果你在研究或开发工作中使用 DOTA 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

特别感谢 DOTA 数据集背后的团队在整理此数据集方面所做的值得称赞的努力。要全面了解数据集及其细微差别，请访问 [DOTA 官方网站](https://captain-whu.github.io/DOTA/index.html)。

## 常见问题

### DOTA8 数据集是什么，如何使用？

DOTA8 数据集是一个小型多功能的旋转目标检测数据集，由 DOTAv1 拆分集的前 8 张图像组成，其中 4 张用于训练，4 张用于验证。它非常适合测试和调试像 Ultralytics YOLO26 这样的目标检测模型。由于其可管理的大小和多样性，它有助于识别流程错误并在部署更大数据集之前运行健全性检查。了解更多关于 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 的目标检测。

### 如何使用 DOTA8 数据集训练 YOLO26 模型？

要在 DOTA8 数据集上以 640 的图像大小训练 YOLO26n-obb 模型 100 个 epoch，可以使用以下代码片段。有关全面的参数选项，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-obb.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo obb train data=dota8.yaml model=yolo26n-obb.pt epochs=100 imgsz=640
        ```

### DOTA 数据集的主要特性是什么，在哪里可以访问 YAML 文件？

DOTA 数据集以其大规模基准和航拍图像目标检测的挑战而闻名。DOTA8 子集是一个较小、易于管理的数据集，非常适合初始测试。你可以在 [GitHub 链接](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/dota8.yaml) 访问包含路径、类别和配置详情的 `dota8.yaml` 文件。

### 马赛克如何增强 DOTA8 数据集的模型训练？

马赛克在训练期间将多张图像合并为一张，增加了每个批次中物体和上下文的多样性。这提高了模型对不同物体大小、宽高比和场景的泛化能力。此技术可以通过由马赛克 DOTA8 数据集图像组成的训练批次来直观展示，有助于稳健的模型开发。在我们的[训练](../../modes/train.md)页面上了解更多关于马赛克和训练技术的信息。

### 为什么我应该使用 Ultralytics YOLO26 进行目标检测任务？

Ultralytics YOLO26 提供了最先进的实时目标检测能力，包括旋转边界框 (OBB)、[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)以及高度通用的训练流程等功能。它适用于各种应用，并提供预训练模型以实现高效微调。在 [Ultralytics YOLO26 文档](https://github.com/ultralytics/ultralytics)中进一步探索其优势和使用方法。
