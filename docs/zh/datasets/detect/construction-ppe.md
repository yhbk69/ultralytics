---
comments: true
description: 探索 Construction-PPE 数据集，这是一个用于检测现实建筑工地中头盔、背心、手套、靴子和护目镜的专用数据集。包含合规和不合规场景，适用于 AI 驱动的安全监控。
keywords: Construction-PPE, PPE 数据集, 安全合规, 建筑工人, 目标检测, YOLO26, 工作场所安全, 计算机视觉
---

# Construction-PPE 数据集

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-construction-ppe-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 Construction-PPE 数据集"></a>

Construction-PPE 数据集旨在通过检测头盔、背心、手套、靴子和护目镜等基本防护装备以及缺失设备的标注，提高建筑工地的安全合规性。该数据集从真实的建筑环境中整理，包含合规和不合规案例，是训练监控工作场所安全的 AI 模型的宝贵资源。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/lFaVnrhMmaE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在个人防护装备数据集上训练 Ultralytics YOLO | 建筑中的 VisionAI 👷
</p>

## 数据集结构

Construction-PPE 数据集分为三个主要子集：

- **训练集**：带标注的建筑图像主要集合，包含工人佩戴完整和部分 PPE 的情况。
- **验证集**：用于在 PPE 检测和合规监控期间微调和评估模型性能的指定子集。
- **测试集**：用于评估最终模型在检测 PPE 和识别合规问题方面有效性的独立子集。

每张图像均以 [Ultralytics YOLO](../detect/index.md#ultralytics-yolo-数据集格式是什么以及如何结构化) 格式标注，确保与最先进的[目标检测](../../tasks/detect.md)和[跟踪](../../modes/track.md)流程兼容。

该数据集提供 **11 个类别**，分为阳性（佩戴 PPE）和阴性（缺失 PPE）类别。这种双阳性/阴性结构使模型能够检测正确佩戴的装备**并**识别安全违规行为。

## 商业价值

- 建筑业仍然是世界上最危险的行业之一，2023/2024 年英国 123 起与工作相关的**致命伤害**中有 51 起发生在建筑业。然而，问题已不再是缺乏监管，42% 的建筑工人承认并不总是遵守流程。
- 建筑业已经受到广泛的健康与安全（HSE）标准框架的监管，但 HSE 团队在执行一致性方面面临挑战。HSE 团队通常人手不足，需要在文书工作和审计之间取得平衡，并且缺乏实时监控繁忙且不断变化环境中每个角落的能力。
- 这就是基于计算机视觉的个人防护装备（PPE）检测变得无价的地方。通过自动检查工人是否佩戴**头盔、背心和其他个人防护装备**，您可以确保 HSE 规则不仅存在，而且能在所有工地上得到有效执行。除了合规性，计算机视觉通过揭示工作人员遵守安全实践的程度，提供风险的前导指标，使组织能够发现合规性下降趋势，并在事故发生前预防。
- 作为额外的好处，个人防护装备检测也已知能识别未经授权的现场入侵者，因为**那些没有配备适当安全装备**的人会首先触发通知。最终，PPE 检测是一个简单而强大的计算机视觉用例，提供全面监督、可操作的见解和标准化报告，使建筑公司能够降低风险、保护工人并保障其项目。

## 应用

Construction-PPE 支持各种以安全为重点的计算机视觉应用：

- **自动合规监控**：训练 AI 模型即时检查工人是否佩戴所需的安全装备，如头盔、背心或手套，降低现场风险。
- **工作场所安全分析**：跟踪 PPE 使用情况随时间的变化，发现频繁违规行为，并生成改善安全文化的见解。
- **智能监控系统**：将检测模型与摄像头连接，在 PPE 缺失时发送实时警报，防止事故发生。
- **机器人与自主系统**：使无人机或机器人能够在大型工地上执行 PPE 检查，支持更快、更安全的检查。
- **研究与教育**：为探索工作场所安全和人与对象交互的学生和研究人员提供真实世界的数据集。

## 数据集 YAML

Construction-PPE 数据集包含一个 YAML 配置文件，定义了训练和验证图像路径以及完整的对象类别列表。您可以直接在 Ultralytics 仓库中访问 `construction-ppe.yaml` 文件：[https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml)

!!! example "ultralytics/cfg/datasets/construction-ppe.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/construction-ppe.yaml"
    ```

## 使用方法

您可以在 Construction-PPE 数据集上训练 YOLO26n 模型 100 个 epoch，图像大小为 640。以下示例展示了如何快速开始。有关更多选项和高级配置，请参阅[训练指南](../../modes/train.md)。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained model
        model = YOLO("yolo26n.pt")

        # Train the model on Construction-PPE dataset
        model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=construction-ppe.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## 示例图像与标注

该数据集捕捉了不同环境、光照条件和姿势下的建筑工人。包含**合规**和**不合规**案例。

![Construction-PPE 数据集样本（安全装备检测）](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/construction-ppe-dataset-sample.avif)

## 许可与归属

Construction-PPE 在 [AGPL-3.0 许可证](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)下开发和发布，支持开源研究和具有适当归属的商业应用。

如果您在研究中使用此数据集，请引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Dalvi_Construction_PPE_Dataset_2025,
            author = {Mrunmayee Dalvi and Niyati Singh and Sahil Bhingarde and Ketaki Chalke},
            title = {Construction-PPE: Personal Protective Equipment Detection Dataset},
            month = {January},
            year = {2025},
            version = {1.0.0},
            license = {AGPL-3.0},
            url = {https://docs.ultralytics.com/datasets/detect/construction-ppe/},
            publisher = {Ultralytics}
        }
        ```

## 常见问题

### Construction-PPE 数据集的独特之处是什么？

与通用建筑数据集不同，Construction-PPE 明确包含**缺失装备类别**。这种双标签方法使模型不仅能够检测 PPE，还能实时标记违规行为。

### 包含哪些对象类别？

该数据集涵盖头盔、背心、手套、靴子、护目镜和工人，以及它们的“缺失 PPE”对应类别。这确保了全面的合规覆盖。

### 如何使用 Construction-PPE 数据集训练 YOLO 模型？

要使用 Construction-PPE 数据集训练 YOLO26 模型，可以使用以下代码片段：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=construction-ppe.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

### 该数据集是否适合实际应用？

是的。图像从真实建筑工地在多样化条件下整理而来。这使其对于构建可部署的工作场所安全监控系统非常有效。

### 在 AI 项目中使用 Construction-PPE 数据集有什么好处？

该数据集支持个人防护装备的实时检测，有助于监控建筑工地上的工人安全。通过包含已佩戴和缺失装备的类别，它支持能够自动标记安全违规、生成合规见解并降低风险的 AI 系统。它还为工作场所安全、机器人和学术研究中的计算机视觉解决方案开发提供了实用资源。