---
comments: true
description: 探索包裹分割数据集。利用精选的包裹识别和分拣图像优化物流并增强视觉模型。
keywords: 包裹分割数据集, 计算机视觉, 包裹识别, 物流, 仓库自动化, 分割模型, 训练数据, Ultralytics YOLO
---

# 包裹分割数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-package-segmentation-dataset.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开包裹分割数据集"></a>

包裹分割数据集是专为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域中包裹分割相关任务定制的精选图像集合。该数据集旨在协助从事包裹识别、分拣和处理项目的研究人员、开发人员和爱好者，主要专注于[图像分割](https://www.ultralytics.com/glossary/image-segmentation)任务。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/im7xBCnPURg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO26 训练包裹分割模型 | 工业包裹 🎉
</p>

该数据集包含在不同背景和环境中展示各种包裹的多样化图像集，是训练和评估分割模型的宝贵资源。无论您从事物流、仓库自动化，还是需要精确包裹分析的任何应用，包裹分割数据集都提供了一套有针对性的全面图像，以提升您的计算机视觉算法性能。在我们的[数据集概述页面](https://docs.ultralytics.com/datasets/segment)探索更多用于分割任务的数据集。

## 数据集结构

包裹分割数据集的数据分布如下：

- **训练集**：包含 1920 张图像及对应标注。
- **测试集**：包含 89 张图像，每张图像均与其对应标注配对。
- **验证集**：包含 188 张图像，每张图像都有对应标注。

## 应用场景

由包裹分割数据集促成的包裹分割对于优化物流、增强最后一公里配送、改善制造质量控制以及为智慧城市解决方案做出贡献至关重要。从电商到安全应用，该数据集是关键资源，推动了计算机视觉在多样化、高效包裹分析应用中的创新。

### 智能仓库与物流

在现代仓库中，[视觉 AI 解决方案](https://www.ultralytics.com/solutions)可通过自动化包裹识别和分拣来简化操作。基于该数据集训练的计算机视觉模型可以实时快速检测和分割包裹，即使在光线昏暗或空间杂乱等具有挑战性的环境中也不例外。这带来了更快的处理速度、更少的错误以及[物流操作](https://www.ultralytics.com/blog/ultralytics-yolo11-the-key-to-computer-vision-in-logistics)整体效率的提升。

### 质量控制与损坏检测

包裹分割模型可通过分析形状和外观来识别损坏的包裹。通过检测包裹轮廓的不规则性或变形，这些模型有助于确保只有完整的包裹才能通过供应链，从而减少客户投诉和退货率。这是[制造业质量控制](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision)的关键方面，对于维护产品完整性至关重要。

## 数据集 YAML

YAML（Yet Another Markup Language）文件定义数据集配置，包括路径、类别和其他重要细节。包裹分割数据集的 `package-seg.yaml` 文件维护在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml)。

!!! example "ultralytics/cfg/datasets/package-seg.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/package-seg.yaml"
    ```

## 使用方法

要在包裹分割数据集上以图像大小 640 训练 [Ultralytics YOLO26n](https://docs.ultralytics.com/models/yolo26) 模型 100 个[轮次](https://www.ultralytics.com/glossary/epoch)，您可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练页面](../../modes/train.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n-seg.pt")  # 加载预训练分割模型（推荐用于训练）

        # 在包裹分割数据集上训练模型
        results = model.train(data="package-seg.yaml", epochs=100, imgsz=640)

        # 验证模型
        results = model.val()

        # 对图像执行推理
        results = model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # 加载预训练分割模型并开始训练
        yolo segment train data=package-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640

        # 从上一个检查点恢复训练
        yolo segment train data=package-seg.yaml model=path/to/last.pt resume=True

        # 验证已训练的模型
        yolo segment val data=package-seg.yaml model=path/to/best.pt

        # 使用已训练的模型执行推理
        yolo segment predict model=path/to/best.pt source=path/to/image.jpg
        ```

## 示例数据和标注

包裹分割数据集包含从多个角度拍摄的各种图像集合。以下是数据集中的数据实例及其对应的分割掩码：

![用于物流的包裹分割数据集示例](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/package-seg-sample.avif)

- 该图像展示了包裹分割的一个实例，包含标示已识别包裹对象的标注掩码。数据集包含在不同地点、环境和密度下拍摄的多样化图像集合，是开发专门用于此[分割任务](https://docs.ultralytics.com/tasks/segment)的模型的全面资源。
- 该示例强调了数据集中存在的多样性和复杂性，凸显了高质量数据对于涉及包裹分割的计算机视觉任务的重要性。

## 使用 YOLO26 进行包裹分割的优势

[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) 为包裹分割任务提供了以下几大优势：

1. **速度与精度的平衡**：YOLO26 实现了高精度和高效率，非常适合在快节奏物流环境中进行[实时推理](https://www.ultralytics.com/glossary/real-time-inference)。与 [YOLOv8](https://docs.ultralytics.com/models/yolov8) 等模型相比，它提供了强大的平衡性。

2. **适应性**：使用 YOLO26 训练的模型可以适应各种仓库条件，从昏暗的灯光到杂乱的空间，确保鲁棒性能。

3. **可扩展性**：在节假日旺季等高峰期，YOLO26 模型可以高效扩展以处理增加的包裹量，而不影响性能或[精度](https://www.ultralytics.com/glossary/accuracy)。

4. **集成能力**：YOLO26 可以轻松与现有仓库管理系统集成，并使用 [ONNX](https://docs.ultralytics.com/integrations/onnx) 或 [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) 等格式跨各种平台部署，便于实现端到端的自动化解决方案。

## 引用和致谢

如果您在研究或开发项目中使用了包裹分割数据集，请适当引用来源：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{ factory_package_dataset,
            title = { factory_package Dataset },
            type = { Open Source Dataset },
            author = { factorypackage },
            url = { https://universe.roboflow.com/factorypackage/factory_package },
            year = { 2024 },
            month = { jan },
            note = { visited on 2024-01-24 },
        }
        ```

我们向包裹分割数据集的创建者表示感谢，感谢他们对计算机视觉社区做出的贡献。如需进一步探索数据集和模型训练，请访问我们的 [Ultralytics 数据集](https://docs.ultralytics.com/datasets)页面和[模型训练技巧](https://docs.ultralytics.com/guides/model-training-tips)指南。

## 常见问题

### 包裹分割数据集是什么，它如何助力计算机视觉项目？

包裹分割数据集是专为包裹[图像分割](https://www.ultralytics.com/glossary/image-segmentation)任务定制的精选图像集合。它包含各种背景下包裹的多样化图像，对于训练和评估分割模型极为宝贵。该数据集特别适用于物流、仓库自动化以及需要精确包裹分析的任何项目。

### 如何在包裹分割数据集上训练 Ultralytics YOLO26 模型？

您可以使用 Python 和 CLI 方法训练 [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) 模型。使用[使用方法](#使用方法)部分提供的代码片段。参阅模型[训练页面](../../modes/train.md)了解有关参数和配置的更多详情。

### 包裹分割数据集由哪些部分组成，结构是怎样的？

数据集分为三个主要部分：
- **训练集**：包含 1920 张带标注的图像。
- **测试集**：包含 89 张带对应标注的图像。
- **验证集**：包含 188 张带标注的图像。

这种结构确保了充分的模型训练、验证和测试的均衡数据集，遵循[模型评估指南](https://docs.ultralytics.com/guides/model-evaluation-insights)中概述的最佳实践。

### 为什么要将 Ultralytics YOLO26 与包裹分割数据集配合使用？

Ultralytics YOLO26 为实时[目标检测](https://www.ultralytics.com/glossary/object-detection)和分割任务提供了最先进的[精度](https://www.ultralytics.com/glossary/accuracy)和速度。将其与包裹分割数据集配合使用，可以充分利用 YOLO26 的能力进行精确的包裹分割，这对于[物流](https://www.ultralytics.com/blog/ultralytics-yolo11-the-key-to-computer-vision-in-logistics)和仓库自动化等行业尤为有益。

### 如何访问和使用包裹分割数据集的 package-seg.yaml 文件？

`package-seg.yaml` 文件托管在 Ultralytics 的 GitHub 仓库上，包含有关数据集路径、类别和配置的重要信息。您可以在 <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/package-seg.yaml> 查看或下载该文件。此文件对于配置模型以高效使用数据集至关重要。更多见解和实际示例，请探索我们的 [Python 用法](https://docs.ultralytics.com/usage/python)部分。
