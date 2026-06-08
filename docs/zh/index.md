---
comments: true
description: 探索 Ultralytics YOLO - 实时目标检测和图像分割的最新进展。了解其功能并在您的项目中最大化其潜力。
keywords: Ultralytics, YOLO, YOLO26, YOLO11, 目标检测, 图像分割, 深度学习, 计算机视觉, AI, 机器学习, 文档, 教程
---

<div align="center">
<br><br>
<a href="https://platform.ultralytics.com/ultralytics/yolo26?utm_source=docs&utm_medium=referral&utm_campaign=platform_launch&utm_content=banner&utm_term=ultralytics_docs" target="_blank"><img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
<br><br>
</div>

<p align="center">
<a href="https://docs.ultralytics.com/zh">中文</a> ·
<a href="https://docs.ultralytics.com/ko">한국어</a> ·
<a href="https://docs.ultralytics.com/ja">日本語</a> ·
<a href="https://docs.ultralytics.com/ru">Русский</a> ·
<a href="https://docs.ultralytics.com/de">Deutsch</a> ·
<a href="https://docs.ultralytics.com/fr">Français</a> ·
<a href="https://docs.ultralytics.com/es">Español</a> ·
<a href="https://docs.ultralytics.com/pt">Português</a> ·
<a href="https://docs.ultralytics.com/tr">Türkçe</a> ·
<a href="https://docs.ultralytics.com/vi">Tiếng Việt</a> ·
<a href="https://docs.ultralytics.com/ar">العربية</a>
</p>

<div align="center">
<br>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://clickpy.clickhouse.com/dashboard/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo26"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
<br><br>
</div>

# 首页

欢迎了解 Ultralytics [YOLO26](models/yolo26.md)，这是备受赞誉的实时目标检测和图像分割模型的最新版本。YOLO26 基于[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)和[计算机视觉](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025)的进步构建，具有端到端无需 NMS 的推理和优化的边缘部署功能。其简洁的设计使其适用于各种应用，并易于适应不同的硬件平台，从边缘设备到云 API。对于稳定的生产工作负载，建议同时使用 YOLO26 和 [YOLO11](models/yolo11.md)。

探索 Ultralytics 文档，这是一个全面的资源，旨在帮助您理解并利用其功能和能力。无论您是经验丰富的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)从业者还是该领域的新手，这个中心都旨在最大化 YOLO 在您项目中的潜力。

如需商业用途，请在 [Ultralytics 许可](https://www.ultralytics.com/license?utm_source=docs.ultralytics.com&utm_medium=referral&utm_content=license_inline_link) 申请企业许可证。

<div align="center">
  <br>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

## 从何处开始

<div class="grid cards" markdown>

- :material-clock-fast:{ .lg .middle } &nbsp; **快速开始**

    ***

    使用 pip 安装 `ultralytics`，几分钟内即可启动并运行，训练一个 YOLO 模型

    ***

    [:octicons-arrow-right-24: 快速开始](quickstart.md)

- :material-image:{ .lg .middle } &nbsp; **预测**

    ***

    使用 YOLO 对新图像、视频和流进行预测 <br /> &nbsp;

    ***

    [:octicons-arrow-right-24: 了解更多](modes/predict.md)

- :fontawesome-solid-brain:{ .lg .middle } &nbsp; **训练模型**

    ***

    在您自己的自定义数据集上从头开始训练一个新的 YOLO 模型，或加载预训练模型并在此基础上训练

    ***

    [:octicons-arrow-right-24: 了解更多](modes/train.md)

- :material-magnify-expand:{ .lg .middle } &nbsp; **探索计算机视觉任务**

    ***

    探索 YOLO 任务，如检测、分割、分类、姿态估计、定向边界框和跟踪 <br /> &nbsp;

    ***

    [:octicons-arrow-right-24: 探索任务](tasks/index.md)

- :rocket:{ .lg .middle } &nbsp; **探索 YOLO26 🚀 新**

    ***

    探索 Ultralytics 最新的 YOLO26 模型，具有无需 NMS 的推理和边缘优化功能 <br /> &nbsp;

    ***

    [:octicons-arrow-right-24: YOLO26 模型 🚀](models/yolo26.md)

- :material-select-all:{ .lg .middle } &nbsp; **SAM 3：基于概念的分割一切 🚀 新**

    ***

    Meta 最新的 SAM 3 具有可提示的概念分割功能 - 使用文本或图像示例分割所有实例

    ***

    [:octicons-arrow-right-24: SAM 3 模型](models/sam-3.md)

- :material-scale-balance:{ .lg .middle } &nbsp; **开源，AGPL-3.0**

    ***

    Ultralytics 提供两种 YOLO 许可证：AGPL-3.0 和企业版。在 [GitHub](https://github.com/ultralytics/ultralytics) 上探索 YOLO。

    ***

    [:octicons-arrow-right-24: YOLO 许可证](https://www.ultralytics.com/license)

</div>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/7lZa3Yi2kbo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何在 <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">Google Colab</a> 中在您的自定义数据集上训练 YOLO26 模型。
</p>

## YOLO：简史

[YOLO](models/index.md)（You Only Look Once）是一种流行的[目标检测](https://www.ultralytics.com/glossary/object-detection)和[图像分割](https://www.ultralytics.com/glossary/image-segmentation)模型，由华盛顿大学的 Joseph Redmon 和 Ali Farhadi 开发。于 2015 年推出，YOLO 因其高速和高精度而广受欢迎。

- [YOLOv2](models/index.md) 于 2016 年发布，通过引入批量归一化、锚框和维度聚类改进了原始模型。
- [YOLOv3](models/yolov3.md) 于 2018 年推出，通过使用更高效的骨干网络、多个锚框和空间金字塔池化进一步提升了模型性能。
- [YOLOv4](models/yolov4.md) 于 2020 年发布，引入了 Mosaic [数据增强](https://www.ultralytics.com/glossary/data-augmentation)、新的无锚框检测头和新的[损失函数](https://www.ultralytics.com/glossary/loss-function)等创新。
- [YOLOv5](models/yolov5.md) 进一步提升了模型性能，并增加了超参数优化、集成实验跟踪和自动导出到流行导出格式等新功能。
- [YOLOv6](models/yolov6.md) 由[美团](https://www.meituan.com/)于 2022 年开源，并用于该公司许多自主配送机器人中。
- [YOLOv7](models/yolov7.md) 增加了在 COCO 关键点数据集上进行姿态估计等额外任务。
- [YOLOv8](models/yolov8.md) 由 Ultralytics 于 2023 年发布，引入了新功能和改进，以提升性能、灵活性和效率，支持全方位的视觉 AI 任务。
- [YOLOv9](models/yolov9.md) 引入了可编程梯度信息（PGI）和广义高效层聚合网络（GELAN）等创新方法。
- [YOLOv10](models/yolov10.md) 由[清华大学](https://www.tsinghua.edu.cn/en/)的研究人员使用 [Ultralytics](https://www.ultralytics.com/) [Python 包](https://pypi.org/project/ultralytics/)创建，通过引入消除非极大值抑制（NMS）需求的端到端头部，提供了实时[目标检测](tasks/detect.md)的进步。
- **[YOLO11](models/yolo11.md)**：于 2024 年 9 月发布，YOLO11 在多个任务上提供出色的性能，包括[目标检测](tasks/detect.md)、[分割](tasks/segment.md)、[姿态估计](tasks/pose.md)、[跟踪](modes/track.md)和[分类](tasks/classify.md)，可在各种 AI 应用和领域中部署。
- **[YOLO26](models/yolo26.md) 🚀**：Ultralytics 的新一代 YOLO 模型，针对边缘部署进行了优化，具有端到端无需 NMS 的推理功能。

## YOLO 许可证：Ultralytics YOLO 如何授权？

<a href="https://www.ultralytics.com/license?utm_source=docs.ultralytics.com&utm_medium=referral&utm_content=license_banner" target="_blank" rel="noopener noreferrer">
<img width="100%" style="border-radius:.4rem" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-license.avif" alt="Ultralytics Enterprise License banner"></a>

Ultralytics 提供两种许可选项以适应不同的使用场景：

- **AGPL-3.0 许可证**：这种[OSI 批准](https://opensource.org/license/agpl-3.0)的开源许可证非常适合学生和爱好者，促进开放协作和知识共享。有关详细信息，请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。
- **企业许可证**：专为商业用途设计，此许可证允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品和服务中，绕过 AGPL-3.0 的开源要求。如果您的场景涉及将我们的解决方案嵌入到商业产品中，请通过 [Ultralytics 许可](https://www.ultralytics.com/license)联系我们。

我们的许可策略旨在确保对我们开源项目的任何改进都会回馈给社区。我们相信开源，我们的使命是确保我们的贡献能够以有益于每个人的方式使用和扩展。

## 目标检测的演进

目标检测多年来经历了显著的发展，从传统的计算机视觉技术到先进的深度学习模型。[YOLO 系列模型](https://www.ultralytics.com/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models)一直处于这一演进的前沿，不断推动实时目标检测的可能性边界。

YOLO 的独特方法将目标检测视为一个单一的回归问题，在一次评估中直接从完整图像预测[边界框](https://www.ultralytics.com/glossary/bounding-box)和类别概率。这种革命性的方法使 YOLO 模型比之前的两阶段检测器显著更快，同时保持高精度。

随着每个新版本的发布，YOLO 都引入了架构改进和创新技术，提升了各种指标的性能。YOLO26 延续了这一传统，融合了计算机视觉研究的最新进展，具有端到端无需 NMS 的推理和优化的边缘部署功能，适用于实际应用。

## 常见问题

### 什么是 Ultralytics YOLO，它如何改进目标检测？

Ultralytics YOLO 是备受赞誉的 YOLO（You Only Look Once）系列，用于实时目标检测和图像分割。最新模型 [YOLO26](models/yolo26.md) 在先前版本的基础上构建，引入了端到端无需 NMS 的推理和优化的边缘部署。YOLO 支持各种[视觉 AI 任务](tasks/index.md)，如检测、分割、姿态估计、跟踪和分类。其高效的架构确保了出色的速度和精度，使其适用于各种应用，包括边缘设备和云 API。

### 如何开始 YOLO 的安装和设置？

开始使用 YOLO 快速而简单。您可以使用 [pip](https://pypi.org/project/ultralytics/) 安装 Ultralytics 包，并在几分钟内启动并运行。以下是基本安装命令：

!!! example "使用 pip 安装"

    === "CLI"

        ```bash
        pip install -U ultralytics
        ```

有关全面的分步指南，请访问我们的[快速开始](quickstart.md)页面。此资源将帮助您完成安装说明、初始设置和运行第一个模型。

### 如何在我的数据集上训练自定义 YOLO 模型？

在您的数据集上训练自定义 YOLO 模型涉及几个详细步骤：

1. 准备您的标注数据集。
2. 在 YAML 文件中配置训练参数。
3. 使用 `yolo TASK train` 命令开始训练。（每个 `TASK` 都有其自己的参数）

以下是目标检测任务的示例代码：

!!! example "目标检测任务的训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载一个预训练的 YOLO 模型（您可以选择 n、s、m、l 或 x 版本）
        model = YOLO("yolo26n.pt")

        # 在您的自定义数据集上开始训练
        model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从命令行训练一个 YOLO 模型
        yolo detect train data=path/to/dataset.yaml epochs=100 imgsz=640
        ```

有关详细演练，请查看我们的[训练模型](modes/train.md)指南，其中包含优化训练过程的示例和技巧。

### Ultralytics YOLO 有哪些许可选项？

Ultralytics 为 YOLO 提供两种许可选项：

- **AGPL-3.0 许可证**：这种开源许可证非常适合教育和非商业用途，促进开放协作。
- **企业许可证**：专为商业应用设计，允许将 Ultralytics 软件无缝集成到商业产品中，不受 AGPL-3.0 许可证的限制。

有关更多详细信息，请访问我们的[许可](https://www.ultralytics.com/license)页面。

### 如何将 Ultralytics YOLO 用于实时目标跟踪？

Ultralytics YOLO 支持高效且可定制的多目标跟踪。要利用跟踪功能，您可以使用 `yolo track` 命令，如下所示：

!!! example "视频上目标跟踪的示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载一个预训练的 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 开始跟踪视频中的目标
        # 您也可以使用实时视频流或网络摄像头输入
        model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # 从命令行对视频执行目标跟踪
        # 您可以指定不同的源，如网络摄像头（0）或 RTSP 流
        yolo track source=path/to/video.mp4
        ```

有关设置和运行目标跟踪的详细指南，请查看我们的[跟踪模式](modes/track.md)文档，其中解释了实时场景中的配置和实际应用。
