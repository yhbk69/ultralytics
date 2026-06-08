---
comments: true
description: 探索 Ultralytics COCO8 数据集，一个由 8 张图像组成的通用且易于管理的数据集，非常适合测试目标检测模型和训练流程。
keywords: COCO8, Ultralytics, 数据集, 目标检测, YOLO26, 训练, 验证, 机器学习, 计算机视觉
---

# COCO8 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO8 数据集是一个紧凑而强大的[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集，由 COCO train 2017 集的前 8 张图像组成 —— 4 张用于训练，4 张用于验证。该数据集专为使用 [YOLO](https://docs.ultralytics.com/models/yolo26) 模型和训练流程进行快速测试、调试和实验而设计。其小尺寸使其非常易于管理，而其多样性确保它在扩展到更大数据集之前作为有效的健全性检查。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/uDrn9QZJ2lk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics COCO 数据集概览
</p>

COCO8 完全兼容 [Ultralytics Platform](https://platform.ultralytics.com/) 和 [YOLO26](../../models/yolo26.md)，能够无缝集成到您的计算机视觉工作流程中。

## 数据集 YAML

COCO8 数据集配置在 YAML（Yet Another Markup Language）文件中定义，该文件指定了数据集路径、类别名称和其他基本元数据。您可以在 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml)中查看官方的 `coco8.yaml` 文件。

!!! example "ultralytics/cfg/datasets/coco8.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8.yaml"
    ```

## 使用方法

要在 COCO8 数据集上训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 640，请使用以下示例。有关完整训练选项列表，请参阅 [YOLO 训练文档](../../modes/train.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Train the model on COCO8
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train YOLO26n on COCO8 using the command line
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

以下是 COCO8 数据集中拼接训练批次的示例：

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mosaiced-training-batch-1.avif" alt="COCO8 目标检测数据集拼接训练批次" width="800">

- **拼接图像**：此图像展示了一个训练批次，其中通过拼接增强将多张数据集图像组合在一起。拼接增强增加了每个批次中对象和场景的多样性，帮助模型更好地泛化到各种对象大小、宽高比和背景。

这种技术对 COCO8 等小数据集尤其有用，因为它最大化了训练期间每张图像的价值。

## 引用与致谢

如果您在研究或开发中使用 COCO 数据集，请引用以下论文：

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

特别感谢 [COCO 联盟](https://cocodataset.org/#home) 对[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区的持续贡献。

## 常见问题

### Ultralytics COCO8 数据集有什么用途？

Ultralytics COCO8 数据集专为[目标检测](https://www.ultralytics.com/glossary/object-detection)模型的快速测试和调试而设计。仅有 8 张图像（4 张训练，4 张验证），非常适合验证您的 [YOLO](https://docs.ultralytics.com/models/yolo26) 训练流程，并确保在扩展到更大数据集之前一切正常运行。有关更多详细信息，请查看 [COCO8 YAML 配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml)。

### 如何使用 COCO8 数据集训练 YOLO26 模型？

您可以使用 Python 或 CLI 在 COCO8 上训练 YOLO26 模型：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Train the model on COCO8
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关其他训练选项，请参阅 [YOLO 训练文档](../../modes/train.md)。

### 为什么应该使用 Ultralytics Platform 管理 COCO8 训练？

[Ultralytics Platform](https://platform.ultralytics.com/) 简化了 [YOLO](https://docs.ultralytics.com/models/yolo26) 模型（包括 COCO8）的数据集管理、训练和部署。通过云训练、实时监控和直观的数据集处理等功能，HUB 使您能够一键启动实验，消除了手动设置的麻烦。了解更多关于 [Ultralytics Platform](https://platform.ultralytics.com/) 以及它如何加速您的计算机视觉项目。

### 在 COCO8 数据集训练中使用拼接增强有什么好处？

COCO8 训练中使用的拼接增强在每批次中将多张图像合并为一张。这增加了对象和背景的多样性，帮助您的 [YOLO](https://docs.ultralytics.com/models/yolo26) 模型更好地泛化到新场景。拼接增强对小数据集尤其有价值，因为它最大化了每个训练步骤中可用的信息。更多信息请参见[训练指南](#使用方法)。

### 如何验证在 COCO8 数据集上训练的 YOLO26 模型？

要在 COCO8 上训练后验证您的 YOLO26 模型，请使用 Python 或 CLI 中的模型验证命令。这将使用标准指标评估模型的性能。有关逐步说明，请访问 [YOLO 验证文档](../../modes/val.md)。