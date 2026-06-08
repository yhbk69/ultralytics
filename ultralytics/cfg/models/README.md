<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics 模型配置

欢迎来到 [Ultralytics](https://www.ultralytics.com/) 模型配置目录。本目录包含一组模型配置文件（`*.yaml`），用于定义 Ultralytics YOLO 模型架构。这些配置广泛应用于常见的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务，例如[目标检测](https://docs.ultralytics.com/tasks/detect)、[图像分割](https://docs.ultralytics.com/tasks/segment)、姿态估计、定向边界框（OBB）以及图像分类。

这些配置经过设计，能够在从标准 [CPU](https://en.wikipedia.org/wiki/Central_processing_unit) 到现代 [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) 的各种硬件上高效运行。请选择一个与您的约束条件（延迟、内存和精度）相匹配的基础模型，然后按需进行定制。

要开始使用，请选择一个 `*.yaml` 文件（参见 [YAML 格式](https://www.ultralytics.com/glossary/yaml)），然后将其用于[训练](https://docs.ultralytics.com/modes/train)或导出模型。更多详情，请参阅 Ultralytics [文档](https://docs.ultralytics.com/)，或在 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) 上提交问题。

## 🚀 使用方法

模型配置文件（`*.yaml`）可通过 `yolo` 命令直接从[命令行界面（CLI）](https://docs.ultralytics.com/usage/cli)使用：

```bash
# 使用 coco8 数据集训练一个 YOLO26n 检测模型，训练 100 个 epoch
yolo task=detect mode=train model=yolo26n.yaml data=coco8.yaml epochs=100 imgsz=640
```

同样的 YAML 文件也可以在 [Python](https://www.python.org/) 中使用，其[配置参数](https://docs.ultralytics.com/usage/cfg)与 CLI 中相同：

```python
from ultralytics import YOLO

# 通过 YAML 配置文件初始化 YOLO26n 模型
# 该操作仅创建模型架构，不加载预训练权重
model = YOLO("yolo26n.yaml")

# 或者直接加载预训练的 YOLO26n 模型
# 该操作同时加载架构和在 COCO 上训练好的权重
# model = YOLO("yolo26n.pt")

# 展示模型信息（架构、层、参数量等）
model.info()

# 使用 COCO8 数据集（COCO 的一个小型子集）训练模型 100 个 epoch
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# 使用训练后的模型对图像进行推理
results = model("path/to/image.jpg")
```

## 🏗️ 预训练模型架构

Ultralytics 支持多种模型架构。请访问 [Ultralytics Models](https://docs.ultralytics.com/models) 文档页面了解详细信息和使用示例，包括：

- [YOLO26](https://docs.ultralytics.com/models/yolo26)
- [YOLO12](https://docs.ultralytics.com/models/yolo12)
- [YOLO11](https://docs.ultralytics.com/models/yolo11)
- [YOLOv10](https://docs.ultralytics.com/models/yolov10)
- [YOLOv9](https://docs.ultralytics.com/models/yolov9)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8)
- [YOLOv5](https://docs.ultralytics.com/models/yolov5)
- [更多模型...](https://docs.ultralytics.com/models)

您可以通过加载这些模型的配置文件（`.yaml`）或其[预训练](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)检查点（`.pt`）来轻松使用它们。

## 🤝 贡献新模型

您是否开发了新颖的 YOLO 变体、尝试了独特的架构，或通过特定的调优达到了领先水平（state-of-the-art）的结果？我们鼓励您将创新成果贡献到 Models 板块，与社区分享！新的模型配置、架构改进或性能优化等贡献都非常有价值，有助于丰富 Ultralytics 生态。

在此分享您的工作，可以让其他人从您的洞见中受益，并扩展可用模型的选择范围。这是展示您专业能力的绝佳方式，也能让 Ultralytics YOLO 平台更加通用和强大。

如需贡献，请参阅[贡献指南](https://docs.ultralytics.com/help/contributing)了解提交 [Pull Request（PR）](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)的说明。

感谢您帮助完善 Ultralytics 模型库。