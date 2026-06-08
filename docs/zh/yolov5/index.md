---
comments: true
description: 浏览全面的 Ultralytics YOLOv5 文档，包含训练、部署和模型优化的分步教程。即刻赋能您的视觉项目！
keywords: YOLOv5, Ultralytics, object detection, computer vision, deep learning, AI, tutorials, PyTorch, model optimization, machine learning, neural networks, YOLOv5 tutorial
---

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/yolo" target="_blank">
      <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov5-splash.avif" alt="Ultralytics YOLOv5 v7.0 banner">
    </a>
  </p>

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

</div>

# Ultralytics YOLOv5 全面指南

欢迎来到 Ultralytics [YOLOv5](https://github.com/ultralytics/yolov5)🚀 文档！Ultralytics YOLOv5 是革命性的"你只看一次"[目标检测](https://www.ultralytics.com/glossary/object-detection)模型的第五次迭代，旨在提供实时的高速、高精度结果。虽然 YOLOv5 依然是一款强大的工具，但也建议您探索其后继者——[Ultralytics YOLOv8](../models/yolov8.md)、[YOLO11](../models/yolo11.md) 和 [YOLO26](../models/yolo26.md)，以获取最新的技术进展。

YOLOv5 基于 [PyTorch](https://pytorch.org/) 构建，这款强大的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)框架因其多功能性、易用性和高性能而广受欢迎。我们的文档将引导您完成安装过程，解析模型的架构细节，展示各种应用场景，并提供一系列详细的教程。这些资源将帮助您充分发挥 YOLOv5 在[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目中的全部潜力。让我们开始吧！

## 探索与学习

以下是一系列综合教程，将引导您了解 YOLOv5 的各个方面。

- [训练自定义数据](tutorials/train_custom_data.md) 🚀 推荐：学习如何在自定义数据集上训练 YOLOv5 模型。
- [最佳训练结果技巧](tutorials/tips_for_best_training_results.md) ☘️：揭示优化模型训练过程的实用技巧。
- [多 GPU 训练](tutorials/multi_gpu_training.md)：了解如何利用多个 GPU 加速训练。
- [PyTorch Hub](tutorials/pytorch_hub_model_loading.md) 🌟 新增：学习通过 PyTorch Hub 加载预训练模型。
- [TFLite、ONNX、CoreML、TensorRT 导出](tutorials/model_export.md) 🚀：了解如何将模型导出为不同格式。
- [测试时增强 (TTA)](tutorials/test_time_augmentation.md)：探索如何使用 TTA 提高模型预测精度。
- [模型集成](tutorials/model_ensembling.md)：学习组合多个模型以提升性能的策略。
- [模型剪枝/稀疏化](tutorials/model_pruning_and_sparsity.md)：了解剪枝和稀疏化概念，以及如何创建更高效的模型。
- [超参数演化](tutorials/hyperparameter_evolution.md)：探索自动化[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)以获得更好的模型性能。
- [冻结层的迁移学习](tutorials/transfer_learning_with_frozen_layers.md)：学习如何通过冻结 YOLOv5 中的层来实现[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)。
- [架构概述](tutorials/architecture_description.md) 🌟 深入了解 YOLOv5 模型的结构细节。阅读 [YOLOv5 v6.0 博客文章](https://www.ultralytics.com/blog/yolov5-v6-0-is-here)获取更多见解。
- [ClearML 日志集成](tutorials/clearml_logging_integration.md) 🌟 学习如何集成 [ClearML](https://clear.ml/) 在模型训练期间实现高效日志记录。
- [YOLOv5 与 Neural Magic](tutorials/neural_magic_pruning_quantization.md)：了解如何使用 [Neural Magic 的 DeepSparse](https://github.com/neuralmagic/deepsparse/blob/main/README.md) 对 YOLOv5 模型进行剪枝和量化。
- [Comet 日志集成](tutorials/comet_logging_integration.md) 🌟 新增：探索如何利用 [Comet](https://www.comet.com/site/) 改进模型训练日志记录。

## 支持的环境

Ultralytics 提供一系列即用型环境，每个环境都预装了必要的依赖项，如 [CUDA](https://developer.nvidia.com/cuda)、[CuDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/)，助您快速启动项目。您还可以使用 [Ultralytics 平台](https://platform.ultralytics.com)来管理模型和数据集。

- **免费 GPU Notebook**：<a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**：[GCP 快速入门指南](environments/google_cloud_quickstart_tutorial.md)
- **Amazon**：[AWS 快速入门指南](environments/aws_quickstart_tutorial.md)
- **Azure**：[AzureML 快速入门指南](environments/azureml_quickstart_tutorial.md)
- **Docker**：[Docker 快速入门指南](environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## 项目状态

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

此徽章表示所有 [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) 持续集成 (CI) 测试均已成功通过。这些 CI 测试严格检查 YOLOv5 在多个关键方面的功能和性能：[训练](https://github.com/ultralytics/yolov5/blob/master/train.py)、[验证](https://github.com/ultralytics/yolov5/blob/master/val.py)、[推理](https://github.com/ultralytics/yolov5/blob/master/detect.py)、[导出](https://github.com/ultralytics/yolov5/blob/master/export.py)和[基准测试](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py)。它们确保在 macOS、Windows 和 Ubuntu 上的一致且可靠运行，测试每 24 小时及每次新提交时执行。

<br>
<div align="center">
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

## 连接与贡献

您的 YOLOv5 之旅不必独自前行。加入我们在 [GitHub](https://github.com/ultralytics/yolov5) 上的活跃社区，在 [LinkedIn](https://www.linkedin.com/company/ultralytics/) 上与专业人士交流，在 [Twitter](https://twitter.com/ultralytics) 上分享您的成果，在 [YouTube](https://www.youtube.com/ultralytics?sub_confirmation=1) 上查找学习资源。关注我们的 [TikTok](https://www.tiktok.com/@ultralytics) 和 [BiliBili](https://ultralytics.com/bilibili) 获取更多精彩内容。

有兴趣贡献吗？我们欢迎各种形式的贡献，从代码改进和错误报告到文档更新。请查看我们的[贡献指南](../help/contributing.md)了解更多信息。

我们期待看到您以创新的方式使用 YOLOv5。投入其中，大胆尝试，彻底变革您的计算机视觉项目！🚀

## 常见问题

### Ultralytics YOLOv5 有哪些关键特性？

Ultralytics YOLOv5 以其高速、高[精度](https://www.ultralytics.com/glossary/accuracy)的目标检测能力而闻名。它基于 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 构建，功能多样且易于使用，适用于各种计算机视觉项目。关键特性包括实时推理、支持多种训练技巧（如测试时增强 TTA 和模型集成），以及与 TFLite、ONNX、CoreML 和 TensorRT 等导出格式的兼容性。深入了解 Ultralytics YOLOv5 如何提升您的项目，请参阅我们的 [TFLite、ONNX、CoreML、TensorRT 导出指南](tutorials/model_export.md)。

### 如何在自定义数据集上训练 YOLOv5 模型？

在自定义数据集上训练 YOLOv5 模型涉及几个关键步骤。首先，需要按照要求的格式准备数据集并进行标签标注。然后，配置 YOLOv5 训练参数，并使用 `train.py` 脚本开始训练过程。如需详细了解此流程，请参阅我们的[训练自定义数据指南](tutorials/train_custom_data.md)，其中提供了分步说明，以确保针对您的特定用例获得最佳结果。

### 为什么应该选择 Ultralytics YOLOv5 而非 RCNN 等其他目标检测模型？

Ultralytics YOLOv5 之所以优于 [R-CNN](https://www.ultralytics.com/glossary/object-detection-architectures) 等模型，是因为它在实时目标检测方面具有卓越的速度和精度。YOLOv5 一次性处理整个图像，与需要多次遍历的区域式 RCNN 方法相比速度显著更快。此外，YOLOv5 与各种导出格式的无缝集成以及详尽的文档使其成为初学者和专业人士的绝佳选择。了解更多架构优势，请参阅我们的[架构概述](tutorials/architecture_description.md)。

### 如何在训练过程中优化 YOLOv5 模型性能？

优化 YOLOv5 模型性能涉及调整各种超参数并结合[数据增强](https://www.ultralytics.com/glossary/data-augmentation)和迁移学习等技术。Ultralytics 提供了关于[超参数演化](tutorials/hyperparameter_evolution.md)和[剪枝/稀疏化](tutorials/model_pruning_and_sparsity.md)的全面资源，以提高模型效率。您可以在[最佳训练结果技巧指南](tutorials/tips_for_best_training_results.md)中找到实用技巧，其中提供了在训练期间实现最佳性能的可行建议。

### YOLOv5 应用支持哪些运行环境？

Ultralytics YOLOv5 支持多种环境，包括 [Gradient](https://bit.ly/yolov5-paperspace-notebook)、[Google Colab](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) 和 [Kaggle](https://www.kaggle.com/models/ultralytics/yolov5) 上的免费 GPU Notebook，以及 [Google Cloud](environments/google_cloud_quickstart_tutorial.md)、[Amazon AWS](environments/aws_quickstart_tutorial.md) 和 [Azure](environments/azureml_quickstart_tutorial.md) 等主流云平台。同时也提供 [Docker 镜像](https://hub.docker.com/r/ultralytics/yolov5)以便捷部署。有关设置这些环境的详细指南，请查看我们的[支持的环境](#支持的环境)部分，其中包含每个平台的分步说明。
