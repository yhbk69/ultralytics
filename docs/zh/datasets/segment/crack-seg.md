---
comments: true
description: 探索用于交通安全、基础设施维护和自动驾驶汽车模型开发的大型裂缝分割数据集，使用 Ultralytics YOLO。
keywords: 裂缝分割数据集, Ultralytics, 交通安全, 公共安全, 自动驾驶汽车, 计算机视觉, 道路安全, 基础设施维护, 数据集, YOLO, 分割, 深度学习
---

# 裂缝分割数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-crack-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开裂缝分割数据集"></a>

裂缝分割数据集是一个为交通和公共安全研究人员设计的大型资源。它同样适用于开发[自动驾驶汽车](https://www.ultralytics.com/blog/ai-in-self-driving-cars)模型或探索各种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用。该数据集是 Ultralytics [数据集中心](../../datasets/index.md)上更广泛集合的一部分。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GAFlmuk0fZI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 训练裂缝分割模型 | 建筑 AI 🎉
</p>

该数据集包含从多种道路和墙面场景拍摄的 4029 张静态图像，是裂缝分割任务的宝贵资产。无论您是研究交通基础设施还是旨在提高自动驾驶系统的[精度](https://www.ultralytics.com/glossary/accuracy)，该数据集都为训练[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型提供了丰富的图像集合。

## 数据集结构

裂缝分割数据集分为三个子集：

- **训练集**：3717 张图像及对应标注。
- **测试集**：112 张图像及对应标注。
- **验证集**：200 张图像及对应标注。

## 应用场景

裂缝分割在[基础设施维护](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation)中有实际应用，有助于识别和评估建筑物、桥梁和道路的结构损坏。它还通过使自动系统能够检测路面裂缝以进行及时修复，在提高[道路安全](https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries)方面发挥着关键作用。

在工业环境中，使用 [Ultralytics YOLO26](../../models/yolo26.md) 等深度学习模型进行裂缝检测有助于确保建筑施工中的建筑完整性，防止[制造业](https://www.ultralytics.com/solutions/ai-in-manufacturing)中昂贵的停工损失，并使道路检查更安全、更有效。自动识别和分类裂缝使维护团队能够有效地优先安排维修，从而获得更好的[模型评估洞察](../../guides/model-evaluation-insights.md)。

## 数据集 YAML

[YAML](https://www.ultralytics.com/glossary/yaml)（Yet Another Markup Language）文件定义数据集配置，包含数据集路径、类别及其他相关信息。裂缝分割数据集的 `crack-seg.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/crack-seg.yaml)。

!!! example "ultralytics/cfg/datasets/crack-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/crack-seg.yaml"
    ```

## 使用方法

要在裂缝分割数据集上以图像大小 640 训练 Ultralytics YOLO26n-seg 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，请使用以下 [Python](https://www.python.org/) 或 CLI 代码片段。参阅模型[训练](../../modes/train.md)文档页面获取可用参数和配置的完整列表，例如[超参数调整](../../guides/hyperparameter-tuning.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        # 建议使用 yolo26n-seg.pt 等预训练模型以加快收敛速度
        model = YOLO("yolo26n-seg.pt")

        # 在裂缝分割数据集上训练模型
        # 确保 'crack-seg.yaml' 可访问或提供完整路径
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)

        # 训练后，模型可用于预测或导出
        # results = model.predict(source='path/to/your/images')
        ```

    === "CLI"

        ```bash
        # 使用命令行界面从预训练的 *.pt 模型开始训练
        # 确保数据集 YAML 文件 'crack-seg.yaml' 已正确配置且可访问
        yolo segment train data=crack-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

## 示例数据和标注

裂缝分割数据集包含从不同角度拍摄的多样化图像集合，展示了道路和墙面上不同类型的裂缝。以下是一些示例：

![用于基础设施检测的裂缝分割数据集示例](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/crack-segmentation-sample.avif)

- 该图像展示了[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)，包含标注的[边界框](https://www.ultralytics.com/glossary/bounding-box)，带有标示已识别裂缝的掩码。数据集包含来自不同地点和环境的图像，使其成为开发此类任务鲁棒模型的全面资源。[数据增强](https://www.ultralytics.com/glossary/data-augmentation)等技术可以进一步提升数据集的多样性。在我们的[指南](../../guides/instance-segmentation-and-tracking.md)中了解更多关于实例分割和跟踪的内容。

- 该示例突出了裂缝分割数据集的多样性，强调了高质量数据对训练有效计算机视觉模型的重要性。

## 引用和致谢

如果您在研究或开发工作中使用了裂缝分割数据集，请适当引用来源：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ crack-bphdr_dataset,
            title = { crack Dataset },
            type = { Open Source Dataset },
            author = { University },
            url = { https://universe.roboflow.com/university-bswxt/crack-bphdr },
            year = { 2022 },
            month = { dec },
            note = { visited on 2024-01-23 },
        }
        ```

我们感谢 Roboflow 团队提供裂缝分割数据集，为计算机视觉社区提供了宝贵资源，特别是对道路安全和基础设施评估相关项目。

## 常见问题

### 裂缝分割数据集是什么？

裂缝分割数据集是一个包含 4029 张静态图像的集合，专为交通和公共安全研究而设计。它适用于[自动驾驶汽车](https://www.ultralytics.com/blog/ai-in-self-driving-cars)模型开发和[基础设施维护](https://www.ultralytics.com/blog/using-ai-for-crack-detection-and-segmentation)等任务。它包含用于裂缝检测和[分割](../../tasks/segment.md)任务的训练、测试和验证集。

### 如何使用 Ultralytics YOLO26 在裂缝分割数据集上训练模型？

要在此数据集上训练 [Ultralytics YOLO26](../../models/yolo26.md) 模型，请使用提供的 Python 或 CLI 示例。详细说明和参数请参阅模型[训练](../../modes/train.md)页面。您可以使用 [Ultralytics 平台](https://platform.ultralytics.com)等工具管理训练过程。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型（推荐）
        model = YOLO("yolo26n-seg.pt")

        # 训练模型
        results = model.train(data="crack-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 通过 CLI 从预训练模型开始训练
        yolo segment train data=crack-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

### 为什么在自动驾驶汽车项目中使用裂缝分割数据集？

该数据集对自动驾驶汽车项目非常有价值，因为它包含了涵盖各种真实场景的多样化道路和墙面图像。这种多样性提高了针对裂缝检测训练的模型的鲁棒性，这对道路安全和基础设施评估至关重要。详细的标注有助于[开发](../../guides/model-training-tips.md)能够准确识别潜在道路危险的模型。

### Ultralytics YOLO 为裂缝分割提供了哪些功能？

Ultralytics YOLO 提供实时[目标检测](https://www.ultralytics.com/glossary/object-detection)、分割和分类功能，非常适合裂缝分割任务。它能够高效处理大型数据集和复杂场景。该框架包含用于[训练](../../modes/train.md)、[预测](../../modes/predict.md)和[导出](../../modes/export.md)模型的全面模式。YOLO 的[无锚点检测](https://www.ultralytics.com/blog/benefits-ultralytics-yolo11-being-anchor-free-detector)方法可以提高对裂缝等不规则形状的检测性能，并且可以使用标准[指标](../../guides/yolo-performance-metrics.md)测量性能。

### 如何引用裂缝分割数据集？

如果在工作中使用此数据集，请使用上方提供的 BibTeX 条目进行引用，以适当致谢创建者。
