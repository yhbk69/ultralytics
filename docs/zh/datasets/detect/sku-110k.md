---
comments: true
description: 探索 SKU-110k 数据集，包含密集排列的零售货架图像，非常适合训练和评估目标检测任务中的深度学习模型。
keywords: SKU-110k, 数据集, 目标检测, 零售货架图像, 深度学习, 计算机视觉, 模型训练
---

# SKU-110k 数据集

[SKU-110k](https://github.com/eg4000/SKU110K_CVPR19) 数据集是一个密集排列的零售货架图像集合，旨在支持[目标检测](https://www.ultralytics.com/glossary/object-detection)任务的研究。该数据集由 Eran Goldman 等人开发，包含超过 110,000 个独特的库存单位（SKU）类别，物体密集排列，通常外观相似甚至完全相同，且位置接近。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_gRqR-miFPE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 在 SKU-110k 数据集上训练 YOLOv10 | 零售数据集
</p>

![SKU-110K 数据集密集排列零售货架检测](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/densely-packed-retail-shelf.avif)

## 主要特性

- SKU-110k 包含来自世界各地的商店货架图像，具有密集排列的物体，对最先进的目标检测器构成挑战。
- 数据集包含超过 110,000 个独特的 SKU 类别，提供了多样化的物体外观。
- 标注包括物体的边界框和 SKU 类别标签。

## 数据集结构

SKU-110k 数据集分为三个主要子集：

1. **训练集**：该子集包含 8,219 张图像和标注，用于训练目标检测模型。
2. **验证集**：该子集包含 588 张图像和标注，用于训练期间的模型验证。
3. **测试集**：该子集包含 2,936 张图像，用于训练好的目标检测模型的最终评估。

## 应用场景

SKU-110k 数据集广泛用于训练和评估目标检测任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型，特别是在零售货架展示等密集排列场景中。其应用包括：

- 零售库存管理和自动化
- 电商平台中的产品识别
- 货架图合规性验证
- 商店自助结账系统
- 仓库中的机器人拣选和分拣

数据集多样化的 SKU 类别和密集排列的物体布局使其成为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域研究人员和实践者的宝贵资源。

## 数据集 YAML

YAML 文件用于定义数据集配置，包含数据集路径、类别和其他相关信息。SKU-110K 数据集的 `SKU-110K.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml)。

!!! example "ultralytics/cfg/datasets/SKU-110K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/SKU-110K.yaml"
    ```

## 使用方法

要在 SKU-110K 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=SKU-110K.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例数据与标注

SKU-110k 数据集包含多样化的零售货架图像，物体密集排列，为目标检测任务提供了丰富的上下文。以下是数据集中的一些数据示例及其对应标注：

![SKU-110K 商店货架零售产品检测](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/densely-packed-retail-shelf-1.avif)

- **密集排列的零售货架图像**：该图像展示了零售货架环境中密集排列物体的示例。物体以边界框和 SKU 类别标签标注。

该示例展示了 SKU-110k 数据集中数据的多样性和复杂性，并突显了高质量数据对目标检测任务的重要性。产品的密集排列为检测算法带来了独特的挑战，使该数据集对于开发鲁棒的零售导向计算机视觉解决方案特别有价值。

## 引用与致谢

如果你在研究或开发工作中使用了 SKU-110k 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{goldman2019dense,
          author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
          title     = {Precise Detection in Densely Packed Scenes},
          booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
          year      = {2019}
        }
        ```

我们感谢 Eran Goldman 等人创建和维护 SKU-110k 数据集，该数据集是计算机视觉研究社区的宝贵资源。有关 SKU-110k 数据集及其创建者的更多信息，请访问 [SKU-110k 数据集 GitHub 仓库](https://github.com/eg4000/SKU110K_CVPR19)。

## 常见问题

### SKU-110k 数据集是什么，为什么它对目标检测很重要？

SKU-110k 数据集由密集排列的零售货架图像组成，旨在辅助目标检测任务的研究。它由 Eran Goldman 等人开发，包含超过 110,000 个独特的 SKU 类别。其重要性在于能够以多样化的物体外观和接近度挑战最先进的目标检测器，使其成为计算机视觉领域研究人员和实践者的宝贵资源。在我们的 [SKU-110k 数据集](#sku-110k-dataset)部分了解更多关于数据集结构和应用的信息。

### 如何使用 SKU-110k 数据集训练 YOLO26 模型？

使用 SKU-110k 数据集训练 YOLO26 模型非常简单。以下是训练 YOLO26n 模型 100 个 epoch、图像大小为 640 的示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="SKU-110K.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=SKU-110K.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### SKU-110k 数据集的主要子集有哪些？

SKU-110k 数据集分为三个主要子集：

1. **训练集**：包含 8,219 张图像和标注，用于训练目标检测模型。
2. **验证集**：包含 588 张图像和标注，用于训练期间的模型验证。
3. **测试集**：包含 2,936 张图像，用于训练好的目标检测模型的最终评估。

更多详情请参阅[数据集结构](#dataset-structure)部分。

### 如何配置 SKU-110k 数据集进行训练？

SKU-110k 数据集配置在 YAML 文件中定义，包含数据集路径、类别和其他相关信息。`SKU-110K.yaml` 文件维护在 [SKU-110K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/SKU-110K.yaml)。例如，你可以使用此配置训练模型，如我们的[使用方法](#usage)部分所示。

### 在[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)背景下，SKU-110k 数据集的主要特性是什么？

SKU-110k 数据集包含来自世界各地的商店货架图像，展示了密集排列的物体，对目标检测器构成重大挑战：

- 超过 110,000 个独特的 SKU 类别
- 多样化的物体外观
- 标注包括边界框和 SKU 类别标签

这些特性使 SKU-110k 数据集对于训练和评估目标检测任务中的深度学习模型特别有价值。更多详情请参阅[主要特性](#key-features)部分。

### 如何在研究中引用 SKU-110k 数据集？

如果你在研究或开发工作中使用 SKU-110k 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{goldman2019dense,
          author    = {Eran Goldman and Roei Herzig and Aviv Eisenschtat and Jacob Goldberger and Tal Hassner},
          title     = {Precise Detection in Densely Packed Scenes},
          booktitle = {Proc. Conf. Comput. Vision Pattern Recognition (CVPR)},
          year      = {2019}
        }
        ```

有关数据集的更多信息可在[引用与致谢](#citations-and-acknowledgments)部分找到。
