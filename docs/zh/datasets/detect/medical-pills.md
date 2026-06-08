---
comments: true
description: 探索包含标注图像的 medical-pills 检测数据集。对于训练 AI 模型以进行药物识别和自动化至关重要。
keywords: medical-pills 数据集, 药片检测, 药物成像, 医疗 AI, 计算机视觉, 目标检测, 医疗自动化, 训练数据集
---

# Medical Pills 数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-medical-pills-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 Medical Pills 数据集"></a>

medical-pills 检测数据集是一个概念验证（POC）数据集，经精心策划以展示 AI 在制药应用中的潜力。它包含专门为训练[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)[模型](https://docs.ultralytics.com/models)识别药片而设计的标注图像。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/8gePl_Zcs5c"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 <a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-medical-pills-dataset.ipynb">Google Colab</a> 中的 Medical Pills 检测数据集上训练 Ultralytics YOLO26 模型
</p>

该数据集为自动化基本[任务](https://docs.ultralytics.com/tasks)提供了基础资源，如质量控制、包装自动化和制药工作流中的高效分拣。通过将该数据集集成到项目中，研究人员和开发者可以探索创新的[解决方案](https://docs.ultralytics.com/solutions)，提高[准确度](https://www.ultralytics.com/glossary/accuracy)，简化操作，并最终改善医疗保健成果。

## 数据集结构

medical-pills 数据集分为两个子集：

- **训练集**：包含 92 张图像，每张标注为 `pill` 类别。
- **验证集**：包含 23 张图像及相应的标注。

## 应用场景

使用计算机视觉进行药片检测可以在制药行业实现自动化，支持以下任务：

- **药物分拣**：根据大小、形状或颜色自动分拣药片，提高生产效率。
- **AI 研究与开发**：作为开发和测试制药用例中计算机视觉算法的基准。
- **数字库存系统**：通过集成自动药片识别，为智能库存解决方案提供动力，实现实时库存监控和补货规划。
- **质量控制**：通过识别缺陷、不规则或污染，确保药片生产的一致性。
- **假冒检测**：通过分析视觉特征与已知标准的对比，帮助识别潜在的假冒药物。

## 数据集 YAML

提供 YAML 配置文件以定义数据集的结构，包括路径和类别。medical-pills 数据集的 `medical-pills.yaml` 文件可在 [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml) 访问。

!!! example "ultralytics/cfg/datasets/medical-pills.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/medical-pills.yaml"
    ```

## 使用方法

要在 medical-pills 数据集上以 640 的图像大小训练 YOLO26n 模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，请使用以下示例。详细参数请参阅模型的[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

        # 训练模型
        results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从预训练的 *.pt 模型开始训练
        yolo detect train data=medical-pills.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

!!! example "推理示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("path/to/best.pt")  # 加载微调后的模型

        # 使用模型进行推理
        results = model.predict("https://ultralytics.com/assets/medical-pills-sample.jpg")
        ```

    === "CLI"

        ```bash
        # 使用微调后的 *.pt 模型开始预测
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/medical-pills-sample.jpg"
        ```

## 示例图像与标注

medical-pills 数据集包含展示药片多样性的标注图像。以下是数据集中的标注图像示例：

![Medical-pills 数据集示例图像](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/medical-pills-dataset-sample-image.avif)

- **马赛克图像**：展示的是一个由马赛克数据集图像组成的训练批次。马赛克通过将多张图像合并为一张来增强训练多样性，从而提高模型的泛化能力。

## 与其他数据集的集成

要进行更全面的药物分析，可考虑将 medical-pills 数据集与其他相关数据集结合使用，如用于包装识别的 [package-seg](../segment/package-seg.md)，或像 [brain-tumor](brain-tumor.md) 这样的医学成像数据集，以开发端到端的医疗 AI 解决方案。

## 引用与致谢

该数据集采用 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)。

如果你在研究或开发工作中使用 Medical-pills 数据集，请使用以下详细信息引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Jocher_Ultralytics_Datasets_2024,
            author = {Jocher, Glenn and Rizwan, Muhammad},
            license = {AGPL-3.0},
            month = {Dec},
            title = {Ultralytics Datasets: Medical-pills Detection Dataset},
            url = {https://docs.ultralytics.com/datasets/detect/medical-pills/},
            version = {1.0.0},
            year = {2024}
        }
        ```

## 常见问题

### medical-pills 数据集的结构是怎样的？

该数据集包含 92 张训练图像和 23 张验证图像。每张图像标注为 `pill` 类别，可有效训练和评估用于制药应用的模型。

### 如何在 medical-pills 数据集上训练 YOLO26 模型？

你可以使用提供的 Python 或 CLI 方法，以 640px 的图像大小训练 YOLO26 模型 100 个 epoch。详细说明请参阅[训练示例](#usage)部分，有关模型功能的更多信息，请查看 [YOLO26 文档](../../models/yolo26.md)。

### 在 AI 项目中使用 medical-pills 数据集有哪些好处？

该数据集实现了药片检测的自动化，有助于防止假冒、保证质量和优化制药流程。它也是开发可提高用药安全和供应链效率的 AI 解决方案的宝贵资源。

### 如何在 medical-pills 数据集上进行推理？

可以使用 Python 或 CLI 方法使用微调后的 YOLO26 模型进行推理。代码片段请参阅[推理示例](#usage)部分，其他选项请参阅[预测模式文档](../../modes/predict.md)。

### 在哪里可以找到 medical-pills 数据集的 YAML 配置文件？

YAML 文件位于 [medical-pills.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medical-pills.yaml)，包含在该数据集上训练模型所必需的数据集路径、类别和其他配置详情。
