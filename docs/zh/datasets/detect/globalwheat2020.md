---
comments: true
description: 探索 Global Wheat Head 数据集，用于开发精准的麦穗检测模型。包含训练图像、标注以及作物管理的使用方法。
keywords: Global Wheat Head 数据集, 麦穗检测, 小麦表型分析, 作物管理, 深度学习, 目标检测, 训练数据集
---

# Global Wheat Head 数据集

[Global Wheat Head 数据集](https://www.global-wheat.com/) 是一组图像集合，旨在支持开发精准的麦穗检测模型，用于小麦表型分析和作物管理。麦穗是小麦植株上结谷粒的部分。准确估算麦穗密度和大小对于评估作物健康、成熟度和产量潜力至关重要。该数据集由来自七个国家的九个研究机构合作创建，覆盖多个种植区域，确保模型在不同环境下具有良好的泛化能力。

## 主要特性

- 数据集包含来自欧洲（法国、英国、瑞士）和北美（加拿大）的 3,000 多张训练图像。
- 包含来自澳大利亚、日本和中国的约 1,000 张测试图像。
- 图像均为户外田间照片，捕捉了麦穗外观的自然变化。
- 标注包含麦穗边界框，支持[目标检测](https://docs.ultralytics.com/tasks/detect)任务。

## 数据集结构

Global Wheat Head 数据集分为两个主要子集：

1. **训练集**：该子集包含来自欧洲和北美的 3,000 多张图像。图像标注了麦穗边界框，为训练目标检测模型提供真实标签。
2. **测试集**：该子集包含来自澳大利亚、日本和中国的约 1,000 张图像。这些图像用于评估训练好的模型在未见过的基因型、环境和观测条件下的表现。

## 应用场景

Global Wheat Head 数据集广泛用于训练和评估麦穗检测任务中的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型。该数据集多样化的图像集合，捕捉了各种外观、环境和条件，使其成为[植物表型分析](https://www.ultralytics.com/blog/computer-vision-in-agriculture-transforming-fruit-detection-and-precision-farming)和作物管理领域研究人员和实践者的宝贵资源。

## 数据集 YAML

YAML 文件用于定义数据集配置，包含数据集路径、类别和其他相关信息。Global Wheat Head 数据集的 `GlobalWheat2020.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml)。

!!! example "ultralytics/cfg/datasets/GlobalWheat2020.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/GlobalWheat2020.yaml"
    ```

## 使用方法

要在 Global Wheat Head 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=GlobalWheat2020.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例数据与标注

Global Wheat Head 数据集包含多样化的户外田间图像，捕捉了麦穗外观、环境和条件的自然变化。以下是数据集中的一些数据示例及其对应标注：

![Global Wheat 数据集麦穗检测示例](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/wheat-head-detection-sample.avif)

- **麦穗检测**：该图像展示了麦穗检测的示例，麦穗用边界框标注。数据集提供了各种图像，以促进用于此任务的模型开发。

该示例展示了 Global Wheat Head 数据集中数据的多样性和复杂性，并突显了精准麦穗检测在小麦表型分析和作物管理应用中的重要性。

## 引用与致谢

如果你在研究或开发工作中使用了 Global Wheat Head 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{david2020global,
                 title={Global Wheat Head Detection (GWHD) Dataset: A Large and Diverse Dataset of High-Resolution RGB-Labelled Images to Develop and Benchmark Wheat Head Detection Methods},
                 author={David, Etienne and Madec, Simon and Sadeghi-Tehran, Pouria and Aasen, Helge and Zheng, Bangyou and Liu, Shouyang and Kirchgessner, Norbert and Ishikawa, Goro and Nagasawa, Koichi and Badhon, Minhajul and others},
                 journal={arXiv preprint arXiv:2005.02162},
                 year={2020}
        }
        ```

我们感谢为创建和维护 Global Wheat Head 数据集做出贡献的研究人员和机构，该数据集是植物表型分析和作物管理研究社区的宝贵资源。有关数据集及其创建者的更多信息，请访问 [Global Wheat Head 数据集网站](https://www.global-wheat.com/)。

## 常见问题

### Global Wheat Head 数据集用于什么？

Global Wheat Head 数据集主要用于开发和训练深度学习模型，以进行麦穗检测。这对于[小麦表型分析](https://www.ultralytics.com/blog/from-farm-to-table-how-ai-drives-innovation-in-agriculture)和作物管理至关重要，可以更准确地估算麦穗密度、大小和整体作物产量潜力。精准的检测方法有助于评估作物健康和成熟度，对高效作物管理至关重要。

### 如何在 Global Wheat Head 数据集上训练 YOLO26n 模型？

要在 Global Wheat Head 数据集上训练 YOLO26n 模型，可以使用以下代码片段。确保你拥有指定数据集路径和类别的 `GlobalWheat2020.yaml` 配置文件：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型（推荐用于训练）
        model = YOLO("yolo26n.pt")

        # 训练模型
        results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=GlobalWheat2020.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

### Global Wheat Head 数据集的主要特性是什么？

Global Wheat Head 数据集的主要特性包括：

- 来自欧洲（法国、英国、瑞士）和北美（加拿大）的 3,000 多张训练图像。
- 来自澳大利亚、日本和中国的约 1,000 张测试图像。
- 由于不同种植环境导致麦穗外观高度可变。
- 详细的麦穗边界框标注，辅助[目标检测](https://www.ultralytics.com/glossary/object-detection)模型。

这些特性有助于开发能够在多个区域泛化的鲁棒模型。

### 在哪里可以找到 Global Wheat Head 数据集的 YAML 配置文件？

Global Wheat Head 数据集的 YAML 配置文件名为 `GlobalWheat2020.yaml`，可在 GitHub 上获取：<https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml>。该文件包含 [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo26) 模型训练所需的数据集路径、类别和其他配置信息。

### 为什么麦穗检测在作物管理中很重要？

麦穗检测在作物管理中至关重要，因为它能够准确估算麦穗密度和大小，这些是评估作物健康、成熟度和产量潜力的关键指标。通过利用在 Global Wheat Head 数据集上训练的[深度学习模型](https://docs.ultralytics.com/models)，农民和研究人员可以更好地监测和管理作物，提高生产力并优化农业实践中的资源利用。这一技术进步支持[可持续农业](https://www.ultralytics.com/blog/real-time-crop-health-monitoring-with-ultralytics-yolo11)和粮食安全倡议。

有关 AI 在农业中应用的更多信息，请访问 [AI 在农业中的应用](https://www.ultralytics.com/solutions/ai-in-agriculture)。
