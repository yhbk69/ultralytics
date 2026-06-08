---
comments: true
description: 了解用于训练模型以识别和验证各种文档中手写签名的 Signature Detection 数据集。非常适合文档验证和欺诈预防。
keywords: Signature Detection 数据集, 文档验证, 欺诈检测, 计算机视觉, YOLO26, Ultralytics, 标注签名, 训练数据集
---

# Signature Detection 数据集

该数据集专注于检测文档中的手写签名。它包含各种文档类型及标注的签名，为文档验证和欺诈检测应用提供了宝贵的洞察。该数据集对于训练[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)算法至关重要，有助于在各种文档格式中识别签名，支持文档分析领域的研究和实际应用。

## 数据集结构

signature detection 数据集分为两个子集：

- **训练集**：包含 143 张图像，每张都有相应的标注。
- **验证集**：包含 35 张图像，每张都有配对的标注。

## 应用场景

该数据集可应用于各种计算机视觉任务，如[目标检测](https://www.ultralytics.com/glossary/object-detection)、[目标追踪](https://docs.ultralytics.com/modes/track)和文档分析。具体来说，它可用于训练和评估识别文档中签名的模型，在以下方面具有重要应用：

- **文档验证**：自动化法律和金融文档的验证流程
- **欺诈检测**：识别潜在的伪造或未经授权的签名
- **数字文档处理**：简化行政和法律部门的工作流程
- **银行与金融**：增强支票处理和贷款文件验证的安全性
- **档案研究**：支持历史文档分析和编目

此外，它还是教育目的的宝贵资源，使学生和研究人员能够研究不同文档类型中的签名特征。

## 数据集 YAML

YAML 文件定义了数据集配置，包括路径和类别信息。signature detection 数据集的 `signature.yaml` 文件位于 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml)。

!!! example "ultralytics/cfg/datasets/signature.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/signature.yaml"
    ```

## 使用方法

要在 signature detection 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，请使用提供的代码示例。有关可用参数的完整列表，请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=signature.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载签名检测微调模型

        # 使用模型进行推理
        results = model.predict("https://ultralytics.com/assets/signature-s.mp4", conf=0.75)
        ```

    === "CLI"

        ```bash
        # 使用微调后的 *.pt 模型开始预测
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/signature-s.mp4" conf=0.75
        ```

## 示例图像与标注

signature detection 数据集包含各种展示不同文档类型和标注签名的图像。以下是数据集中的图像示例，每张都配有相应的标注。

![Signature detection 数据集示例图像](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/signature-detection-mosaiced-sample.avif)

- **马赛克图像**：这里展示的是一个由马赛克数据集图像组成的训练批次。马赛克是一种训练技术，将多张图像合并为一张，丰富了批次多样性。这种方法有助于增强模型在不同签名大小、宽高比和上下文中的泛化能力。

该示例展示了 Signature Detection 数据集中图像的多样性和复杂性，强调了在训练过程中包含马赛克技术的好处。

## 引用与致谢

该数据集采用 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)发布。

## 常见问题

### Signature Detection 数据集是什么，如何使用？

Signature Detection 数据集是一个标注图像集合，旨在检测各种文档类型中的手写签名。它可应用于[目标检测](https://www.ultralytics.com/glossary/object-detection)和追踪等计算机视觉任务，主要用于文档验证、欺诈检测和档案研究。该数据集有助于训练模型在不同上下文中识别签名，对[智能文档分析](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-smart-document-analysis)的研究和实际应用都很有价值。

### 如何在 Signature Detection 数据集上训练 YOLO26n 模型？

要在 Signature Detection 数据集上训练 YOLO26n 模型，请按以下步骤操作：

1. 从 [signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml) 下载 `signature.yaml` 数据集配置文件。
2. 使用以下 Python 脚本或 CLI 命令开始训练：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO("yolo26n.pt")

        # 训练模型
        results = model.train(data="signature.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=signature.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

更多详情请参阅[训练](../../modes/train.md)页面。

### Signature Detection 数据集的主要应用有哪些？

Signature Detection 数据集可用于：

1. **文档验证**：自动验证文档中手写签名的存在和真实性。
2. **欺诈检测**：识别法律和金融文档中的伪造或欺诈签名。
3. **档案研究**：协助历史学家和档案管理员对历史文档进行数字分析和编目。
4. **教育**：支持计算机视觉和[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)领域的学术研究和教学。
5. **金融服务**：通过验证签名真实性增强银行交易和贷款处理的安全性。

### 如何使用在 Signature Detection 数据集上训练的模型进行推理？

要使用在 Signature Detection 数据集上训练的模型进行推理，请按以下步骤操作：

1. 加载你的微调模型。
2. 使用以下 Python 脚本或 CLI 命令进行推理：

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载微调模型
        model = YOLO("path/to/best.pt")

        # 进行推理
        results = model.predict("https://ultralytics.com/assets/signature-s.mp4", conf=0.75)
        ```

    === "CLI"

        ```bash
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/signature-s.mp4" conf=0.75
        ```

### Signature Detection 数据集的结构是怎样的，在哪里可以找到更多信息？

Signature Detection 数据集分为两个子集：

- **训练集**：包含 143 张带标注的图像。
- **验证集**：包含 35 张带标注的图像。

详细信息请参阅[数据集结构](#dataset-structure)部分。此外，可在 [signature.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/signature.yaml) 查看完整的数据集配置。
