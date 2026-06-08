---
comments: true
description: 通过 Ultralytics 教程掌握 YOLO，涵盖训练、部署和优化。找到解决方案，提升指标，轻松部署。
keywords: Ultralytics, YOLO, 教程, 指南, 目标检测, 深度学习, PyTorch, 训练, 部署, 优化, 计算机视觉
---

# Ultralytics YOLO 综合教程

欢迎来到 Ultralytics YOLO 指南。我们的综合教程涵盖了 YOLO [目标检测](https://www.ultralytics.com/glossary/object-detection)模型的方方面面，从训练、预测到部署。YOLO 基于 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 构建，以其在实时目标检测任务中卓越的速度和[精度](https://www.ultralytics.com/glossary/accuracy)而脱颖而出。

无论您是[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)的初学者还是专家，我们的教程都能为您在[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目中实施和优化 YOLO 提供宝贵见解。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/96NkhsV-W1U"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics YOLO26 指南概览
</p>

## 指南

以下是一些深度指南的汇编，帮助您掌握 Ultralytics YOLO 的各个方面。

- [模型测试指南](model-testing.md)：一份关于在真实场景中测试计算机视觉模型的详尽指南。学习如何根据项目目标验证准确性、可靠性和性能。
- [AzureML 快速入门](azureml-quickstart.md)：在 Microsoft Azure [机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)平台上快速上手 Ultralytics YOLO 模型。学习如何在云端训练、部署和扩展您的目标检测项目。
- [模型部署最佳实践](model-deployment-practices.md)：浏览计算机视觉项目中高效部署模型的技巧和最佳实践，重点涵盖优化、故障排除和安全性。
- [COCO 到 YOLO 转换](coco-to-yolo.md)：将 COCO JSON 标注转换为 YOLO 训练格式的完整指南。涵盖检测、分割和关键点，并提供 CVAT、Label Studio 和 Roboflow 的工具专属技巧。
- [COCO JSON 训练](coco-json-training.md)：使用自定义数据集类和训练器，直接在 COCO JSON 标注上训练 YOLO，无需转换为 YOLO 格式。
- [Conda 快速入门](conda-quickstart.md)：为 Ultralytics 设置 [Conda](https://anaconda.org/conda-forge/ultralytics) 环境的分步指南。学习如何使用 Conda 高效安装并开始使用 Ultralytics 包。
- [自定义训练器](custom-trainer.md)：学习如何子类化 YOLO 训练器，以记录自定义指标、添加类别加权损失、自定义模型保存、冻结/解冻骨干网络以及设置逐层学习率。
- [数据收集与标注](data-collection-and-annotation.md)：探索为计算机视觉模型收集和标注数据以创建高质量输入的工具、技术和最佳实践。
- [NVIDIA Jetson 上的 DeepStream](deepstream-nvidia-jetson.md)：使用 DeepStream 和 TensorRT 在 NVIDIA Jetson 设备上部署 YOLO 模型的快速入门指南。
- [定义计算机视觉项目目标](defining-project-goals.md)：了解如何有效地为计算机视觉项目定义清晰且可衡量的目标。学习一个明确的问题陈述的重要性，以及它如何为您的项目绘制路线图。
- [Docker 快速入门](docker-quickstart.md)：使用 [Docker](https://hub.docker.com/r/ultralytics/ultralytics) 设置和使用 Ultralytics YOLO 模型的完整指南。学习如何安装 Docker、管理 GPU 支持，并在隔离容器中运行 YOLO 模型，实现一致的开发和部署。
- [Raspberry Pi 上的 Edge TPU](coral-edge-tpu-on-raspberry-pi.md)：[Google Edge TPU](https://developers.google.com/coral) 加速 YOLO 在 [Raspberry Pi](https://www.raspberrypi.com/) 上的推理。
- [端到端检测](end2end-detection.md)：了解 YOLO26 的无 NMS 端到端检测、导出兼容性、输出格式变化，以及如何从旧版 YOLO 模型迁移。
- [导出非 YOLO 模型](export-non-yolo-models.md)：使用 Ultralytics 独立导出工具将任意 `torch.nn.Module`（timm、torchvision、自定义模型）转换为 ONNX、TorchScript、OpenVINO、CoreML、NCNN、MNN、PaddlePaddle、ExecuTorch 和 TensorFlow SavedModel。
- [在自定义数据上微调 YOLO](finetuning-guide.md)：使用预训练权重在自定义数据集上微调 YOLO26 的完整指南，涵盖迁移学习、层冻结、优化器选择、两阶段训练和故障排除。
- [超参数调优](hyperparameter-tuning.md)：了解如何使用 Tuner 类和遗传进化算法微调超参数来优化您的 YOLO 模型。
- [模型评估与微调洞察](model-evaluation-insights.md)：深入了解评估和微调计算机视觉模型的策略和最佳实践。了解精炼模型以达到最佳结果的迭代过程。
- [分离分割对象](isolating-segmentation-objects.md)：关于如何使用 Ultralytics 分割从图像中提取和/或隔离对象的分步方案和说明。
- [K 折交叉验证](kfold-cross-validation.md)：学习如何使用 K 折交叉验证技术提高模型泛化能力。
- [维护您的计算机视觉模型](model-monitoring-and-maintenance.md)：了解监控、维护和记录计算机视觉模型的关键实践，以确保准确性、发现异常并缓解数据漂移。
- [模型部署选项](model-deployment-options.md)：YOLO [模型部署](https://www.ultralytics.com/glossary/model-deployment)格式概览，如 ONNX、OpenVINO 和 TensorRT，附有各格式的优缺点，为您的部署策略提供参考。
- [模型 YAML 配置指南](model-yaml-config.md)：对 Ultralytics 模型架构定义的全面深入解析。探索 YAML 格式，理解模块解析系统，并学习如何无缝集成自定义模块。
- [NVIDIA DALI GPU 预处理](nvidia-dali.md)：使用 NVIDIA DALI 在 GPU 上运行 YOLO 的 letterbox 缩放、填充和归一化，消除 CPU 预处理瓶颈，并集成 Triton Inference Server。
- [NVIDIA DGX Spark](nvidia-dgx-spark.md)：在 NVIDIA DGX Spark 设备上部署 YOLO 模型的快速入门指南。
- [NVIDIA Jetson](nvidia-jetson.md)：在 NVIDIA Jetson 设备上部署 YOLO 模型的快速入门指南。
- [OpenVINO 延迟与吞吐量模式](optimizing-openvino-latency-vs-throughput-modes.md)：学习延迟和吞吐量优化技术，以获得 YOLO 推理的最佳性能。
- [预处理标注数据](preprocessing_annotated_data.md)：学习在计算机视觉项目中使用 YOLO26 对图像数据进行预处理和增强，包括归一化、数据集增强、分割和探索性数据分析（EDA）。
- [Raspberry Pi](raspberry-pi.md)：在最新 Raspberry Pi 硬件上运行 YOLO 模型的快速入门教程。
- [ROS 快速入门](ros-quickstart.md)：学习如何将 YOLO 与机器人操作系统（ROS）集成，用于机器人应用中的实时目标检测，包括点云和深度图像。
- [SAHI 分块推理](sahi-tiled-inference.md)：关于利用 SAHI 切片推理能力与 YOLO26 进行高分辨率图像目标检测的综合指南。
- [计算机视觉项目的步骤](steps-of-a-cv-project.md)：了解计算机视觉项目涉及的关键步骤，包括定义目标、选择模型、准备数据和评估结果。
- [模型训练技巧](model-training-tips.md)：探索优化[批次大小](https://www.ultralytics.com/glossary/batch-size)、使用[混合精度](https://www.ultralytics.com/glossary/mixed-precision)、应用预训练权重等技巧，让训练计算机视觉模型变得轻松。
- [Triton Inference Server 集成](triton-inference-server.md)：深入了解 Ultralytics YOLO26 与 NVIDIA Triton Inference Server 的集成，实现可扩展且高效的深度学习推理部署。
- [使用 Docker 部署 Vertex AI](vertex-ai-deployment-with-docker.md)：将 YOLO 模型通过 Docker 容器化并部署到 Google Cloud Vertex AI 的简化指南——涵盖构建、推送、自动扩缩和监控。
- [在终端中查看推理图像](view-results-in-terminal.md)：使用 VSCode 集成终端在 Remote Tunnel 或 SSH 会话中查看推理结果。
- [YOLO26 训练配方](yolo26-training-recipe.md)：用于在 COCO 上训练官方 YOLO26 基础检查点的超参数、增强流水线和优化器设置的完整文档，并提供实用的微调指导。
- [YOLO 常见问题](yolo-common-issues.md) ⭐ 推荐：针对使用 Ultralytics YOLO 模型时最常遇到的问题的实用解决方案和故障排除技巧。
- [YOLO 数据增强](yolo-data-augmentation.md)：掌握 YOLO 中完整的数据增强技术，从基本变换到提高模型鲁棒性和性能的高级策略。
- [YOLO 性能指标](yolo-performance-metrics.md) ⭐ 必读：了解用于评估 YOLO 模型性能的关键指标，如 mAP、IoU 和 [F1 分数](https://www.ultralytics.com/glossary/f1-score)。包含实用示例和提高检测精度与速度的技巧。
- [YOLO 线程安全推理](yolo-thread-safe-inference.md)：以线程安全方式进行 YOLO 模型推理的指南。了解线程安全的重要性以及防止竞态条件、确保一致预测的最佳实践。

## 贡献指南

我们欢迎社区贡献！如果您已经掌握了 Ultralytics YOLO 的某个方面但尚未在我们的指南中涵盖，我们鼓励您分享您的专业知识。编写指南是回馈社区的绝佳方式，有助于使我们的文档更加全面和用户友好。

要开始贡献，请阅读我们的[贡献指南](../help/contributing.md)，了解如何提交 Pull Request（PR）。我们期待您的贡献。

## 常见问题

### 如何使用 Ultralytics YOLO 训练自定义目标检测模型？

使用 Ultralytics YOLO 训练自定义目标检测模型非常简单。首先按正确格式准备数据集并安装 Ultralytics 包。使用以下代码启动训练：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # 加载预训练 YOLO 模型
        model.train(data="path/to/dataset.yaml", epochs=50)  # 在自定义数据集上训练
        ```

    === "CLI"

        ```bash
        yolo task=detect mode=train model=yolo26n.pt data=path/to/dataset.yaml epochs=50
        ```

有关数据集格式化和更多选项的详细信息，请参阅我们的[模型训练技巧](model-training-tips.md)指南。

### 我应该使用哪些性能指标来评估我的 YOLO 模型？

评估 YOLO 模型性能对于理解其有效性至关重要。关键指标包括[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map)（mAP）、[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou)（IoU）和 F1 分数。这些指标有助于评估目标检测任务的准确性和[精确率](https://www.ultralytics.com/glossary/precision)。您可以在我们的 [YOLO 性能指标](yolo-performance-metrics.md)指南中了解更多关于这些指标以及如何改进模型的信息。

### 为什么应该在计算机视觉项目中使用 Ultralytics Platform？

Ultralytics Platform 是一个无代码平台，简化了 YOLO 模型的管理、训练和部署。它支持无缝集成、实时跟踪和云端训练，非常适合初学者和专业人士。通过我们的 [Ultralytics Platform](https://docs.ultralytics.com/platform) 快速入门指南了解更多功能以及它如何简化您的工作流程。

### YOLO 模型训练过程中常见的问题有哪些，如何解决？

YOLO 模型训练过程中常见的问题包括数据格式错误、模型架构不匹配和[训练数据](https://www.ultralytics.com/glossary/training-data)不足。为解决这些问题，请确保数据集格式正确、检查模型版本兼容性并扩充训练数据。有关完整的解决方案列表，请参阅我们的 [YOLO 常见问题](yolo-common-issues.md)指南。

### 如何在边缘设备上部署 YOLO 模型进行实时目标检测？

在 NVIDIA Jetson 和 Raspberry Pi 等边缘设备上部署 YOLO 模型需要将模型转换为兼容格式，如 TensorRT 或 TFLite。按照我们的 [NVIDIA Jetson](nvidia-jetson.md) 和 [Raspberry Pi](raspberry-pi.md) 部署分步指南，开始在边缘硬件上进行实时目标检测。这些指南将带您完成安装、配置和性能优化的全过程。