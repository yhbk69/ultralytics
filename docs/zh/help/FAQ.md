---
comments: true
description: 探索与 Ultralytics YOLO 相关的常见问题及解决方案，涵盖硬件要求、模型微调和实时检测等主题。
keywords: Ultralytics, YOLO, 常见问题, 目标检测, 硬件要求, 微调, ONNX, TensorFlow, 实时检测, 模型精度
---

# Ultralytics YOLO 常见问题解答 (FAQ)

本 FAQ 章节旨在解答用户在使用 [Ultralytics](https://www.ultralytics.com/) YOLO 仓库时可能遇到的常见问题和疑惑。

## 常见问题

### 什么是 Ultralytics？它提供什么？

Ultralytics 是一家专注于最先进的目标检测和[图像分割](https://www.ultralytics.com/glossary/image-segmentation)模型的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv) AI 公司，其核心是 YOLO（You Only Look Once）系列模型。公司产品包括：

- 开源的 [YOLO26](https://docs.ultralytics.com/models/yolo26)（最新版）和 [YOLO11](https://docs.ultralytics.com/models/yolo11)（上一代）实现
- 针对各种计算机视觉任务的多种[预训练模型](https://docs.ultralytics.com/models)
- 一个全面的 [Python 包](https://docs.ultralytics.com/usage/python)，方便将 YOLO 模型无缝集成到项目中
- 用于训练、测试和部署模型的多种[工具](https://docs.ultralytics.com/modes)
- [详尽的文档](https://docs.ultralytics.com/)和一个活跃的社区

### 如何安装 Ultralytics 包？

使用 pip 可轻松安装 Ultralytics 包：

```bash
pip install ultralytics
```

如需安装最新的开发版本，可直接从 GitHub 仓库安装：

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

详细的安装指南请参阅[快速入门指南](https://docs.ultralytics.com/quickstart)。

### 运行 Ultralytics 模型需要什么系统配置？

最低要求：

- Python 3.8+
- [PyTorch](https://www.ultralytics.com/glossary/pytorch) 1.8+
- 支持 CUDA 的 GPU（用于 GPU 加速）

推荐配置：

- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU，CUDA 11.2+
- 8GB+ 内存
- 50GB+ 可用磁盘空间（用于数据集存储和模型训练）

有关常见问题的排查，请访问 [YOLO 常见问题](https://docs.ultralytics.com/guides/yolo-common-issues)页面。

### 如何在自己的数据集上训练自定义 YOLO 模型？

训练自定义 YOLO 模型的步骤：

1. 将数据集准备为 [YOLO 格式](../datasets/detect/index.md#ultralytics-yolo-format)（图片和对应的标签 txt 文件）。
2. 创建一个描述数据集结构和类别的 YAML 文件（参见[数据集 YAML 示例](../datasets/detect/index.md#ultralytics-yolo-format)）。
3. 使用以下 Python 代码启动训练：

    ```python
    from ultralytics import YOLO

    # 加载模型
    model = YOLO("yolo26n.yaml")  # 从头构建新模型
    model = YOLO("yolo26n.pt")  # 加载预训练模型（推荐用于训练）

    # 训练模型
    results = model.train(data="path/to/your/data.yaml", epochs=100, imgsz=640)
    ```

如需更深入的指南（包括数据准备和高级训练选项），请参阅完整的[训练指南](https://docs.ultralytics.com/modes/train)。

### Ultralytics 提供哪些预训练模型？

Ultralytics 提供多种多样的预训练模型，适用于不同的任务：

- 目标检测：YOLO26n、YOLO26s、YOLO26m、YOLO26l、YOLO26x
- [实例分割](https://www.ultralytics.com/glossary/instance-segmentation)：YOLO26n-seg、YOLO26s-seg、YOLO26m-seg、YOLO26l-seg、YOLO26x-seg
- 分类：YOLO26n-cls、YOLO26s-cls、YOLO26m-cls、YOLO26l-cls、YOLO26x-cls
- 姿态估计：YOLO26n-pose、YOLO26s-pose、YOLO26m-pose、YOLO26l-pose、YOLO26x-pose
- 有向目标检测 (OBB)：YOLO26n-obb、YOLO26s-obb、YOLO26m-obb、YOLO26l-obb、YOLO26x-obb

这些模型在大小和复杂度上各异，在速度和[精度](https://www.ultralytics.com/glossary/accuracy)之间提供了不同的权衡。请浏览全部[预训练模型](https://docs.ultralytics.com/models)以找到最适合您项目的模型。

### 如何使用训练好的 Ultralytics 模型进行推理？

使用训练好的模型进行推理：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("path/to/your/model.pt")

# 执行推理
results = model("path/to/image.jpg")

# 处理结果
for r in results:
    print(r.boxes)  # 打印边界框预测
    print(r.masks)  # 打印掩码预测
    print(r.probs)  # 打印类别概率
```

有关高级推理选项（包括批处理和视频推理），请查阅详细的[预测指南](https://docs.ultralytics.com/modes/predict)。

### Ultralytics 模型能否部署到边缘设备或生产环境中？

当然可以！Ultralytics 模型专为跨多种平台的灵活部署而设计：

- 边缘设备：使用 TensorRT、ONNX 或 OpenVINO 在 NVIDIA Jetson 或 Intel Neural Compute Stick 等设备上优化推理。
- 移动端：通过将模型转换为 TFLite 或 Core ML，在 Android 或 iOS 设备上部署。
- 云端：利用 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Serving 或 PyTorch Serve 等框架实现可扩展的云端部署。
- Web 端：使用 ONNX.js 或 TensorFlow.js 实现浏览器内推理。

Ultralytics 提供导出功能，可将模型转换为多种部署格式。请浏览广泛的[部署选项](https://docs.ultralytics.com/guides/model-deployment-options)以找到适合您用例的最佳方案。

### YOLO11 和 YOLO26 有什么区别？

主要区别包括：

- [端到端无 NMS 推理](../guides/end2end-detection.md)：YOLO26 原生支持端到端推理，无需非极大值抑制 (NMS) 即可直接生成预测结果，减少延迟并简化部署。
- 移除 DFL：YOLO26 移除了 Distribution Focal Loss 模块，简化导出流程，提升与边缘和低功耗设备的兼容性。
- MuSGD 优化器：SGD 与 Muon（受月之暗面 Kimi K2 启发）的混合优化器，训练更稳定，收敛更快。
- CPU 性能：YOLO26 的 CPU 推理速度提升高达 43%，非常适合无 GPU 的设备。
- 任务特定优化：通过语义损失和多尺度原型增强分割，使用 RLE 实现精确姿态估计，以及通过角度损失改进 OBB 解码。
- 任务支持：两种模型都在统一框架中支持[目标检测](https://www.ultralytics.com/glossary/object-detection)、实例分割、分类、姿态估计和有向目标检测 (OBB)。

如需功能和性能指标的详细对比，请访问 [YOLO26 文档页面](https://docs.ultralytics.com/models/yolo26)。

### 如何为 Ultralytics 开源项目做贡献？

为 Ultralytics 贡献代码是改进项目并提升技能的好方法。参与方式如下：

1. 在 GitHub 上 Fork Ultralytics 仓库。
2. 为您的功能或错误修复创建一个新分支。
3. 进行修改并确保所有测试通过。
4. 提交 Pull Request，并清晰描述您的修改内容。
5. 参与代码评审流程。

您也可以通过报告错误、建议新功能或改进文档来做出贡献。有关详细指南和最佳实践，请参阅[贡献指南](https://docs.ultralytics.com/help/contributing)。

### 如何在 Python 中安装 Ultralytics 包？

在 Python 中安装 Ultralytics 包非常简单。在终端或命令提示符中运行以下 pip 命令即可：

```bash
pip install ultralytics
```

如需安装前沿开发版本，可直接从 GitHub 仓库安装：

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

有关特定环境的安装说明和故障排查技巧，请参阅全面的[快速入门指南](https://docs.ultralytics.com/quickstart)。

### Ultralytics YOLO 有哪些主要功能？

Ultralytics YOLO 拥有丰富的高级计算机视觉功能：

- 实时检测：在实时场景中高效检测和分类物体。
- 多任务能力：在统一框架中执行目标检测、实例分割、分类和姿态估计。
- 预训练模型：提供多种平衡速度与精度的[预训练模型](https://docs.ultralytics.com/models)，适用于不同用例。
- 自定义训练：通过灵活的[训练流程](https://docs.ultralytics.com/modes/train)轻松在自定义数据集上微调模型。
- 广泛的[部署选项](https://docs.ultralytics.com/guides/model-deployment-options)：将模型导出为 TensorRT、ONNX、CoreML 等多种格式，跨平台部署。
- 详尽文档：全面的[文档](https://docs.ultralytics.com/)和活跃的社区为您的计算机视觉工作流提供支持。

### 如何提升 YOLO 模型的性能？

可通过以下几种技术来提升 YOLO 模型的性能：

1. [超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)：参照[超参数调优指南](https://docs.ultralytics.com/guides/hyperparameter-tuning)尝试不同的超参数组合，优化模型表现。
2. [数据增强](https://www.ultralytics.com/glossary/data-augmentation)：应用翻转、缩放、旋转和颜色调整等技术，丰富训练数据集，提升模型泛化能力。
3. [迁移学习](https://www.ultralytics.com/glossary/transfer-learning)：利用预训练模型，结合[训练指南](../modes/train.md)在您的特定数据集上微调。
4. 导出为高效格式：使用[导出指南](../modes/export.md)将模型转换为 TensorRT 或 ONNX 等优化格式，实现更快的推理。
5. 基准测试：利用[基准测试模式](https://docs.ultralytics.com/modes/benchmark)系统地衡量和改进推理速度与精度。

### 能否将 Ultralytics YOLO 模型部署到移动端和边缘设备上？

可以，Ultralytics YOLO 模型专为多平台灵活部署而设计，包括移动端和边缘设备：

- 移动端：将模型转换为 TFLite 或 CoreML，无缝集成到 Android 或 iOS 应用中。请参阅 [TFLite 集成指南](https://docs.ultralytics.com/integrations/tflite)和 [CoreML 集成指南](https://docs.ultralytics.com/integrations/coreml)获取平台特定说明。
- 边缘设备：使用 TensorRT 或 ONNX 在 NVIDIA Jetson 或其他边缘硬件上优化推理。[Edge TPU 集成指南](https://docs.ultralytics.com/integrations/edge-tpu)提供了边缘部署的详细步骤。

如需跨平台部署策略的全面概述，请参阅[部署选项指南](https://docs.ultralytics.com/guides/model-deployment-options)。

### 如何使用训练好的 Ultralytics YOLO 模型进行推理？

使用训练好的 Ultralytics YOLO 模型进行推理非常简单：

1. 加载模型：

    ```python
    from ultralytics import YOLO

    model = YOLO("path/to/your/model.pt")
    ```

2. 运行推理：

    ```python
    results = model("path/to/image.jpg")

    for r in results:
        print(r.boxes)  # 打印边界框预测
        print(r.masks)  # 打印掩码预测
        print(r.probs)  # 打印类别概率
    ```

有关高级推理技术（包括批处理、视频推理和自定义预处理），请参阅详细的[预测指南](https://docs.ultralytics.com/modes/predict)。

### 在哪里可以找到使用 Ultralytics 的示例和教程？

Ultralytics 提供了丰富的资源，帮助您入门并掌握其工具：

- 📚 [官方文档](https://docs.ultralytics.com/)：全面的指南、API 参考和最佳实践。
- 💻 [GitHub 仓库](https://github.com/ultralytics/ultralytics)：源代码、示例脚本和社区贡献。
- ✍️ [Ultralytics 博客](https://www.ultralytics.com/blog)：深度文章、用例和技术见解。
- 💬 [社区论坛](https://community.ultralytics.com/)：与其他用户交流、提问并分享经验。
- 🎥 [YouTube 频道](https://www.youtube.com/ultralytics?sub_confirmation=1)：涵盖 Ultralytics 各类主题的视频教程、演示和网络研讨会。

这些资源提供了代码示例、真实用例以及针对 Ultralytics 模型各种任务的分步指南。

如需进一步帮助，请查阅 Ultralytics 文档，或通过 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) 或官方[讨论论坛](https://github.com/orgs/ultralytics/discussions)联系社区。
