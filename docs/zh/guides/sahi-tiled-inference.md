---
comments: true
description: 了解如何使用 YOLO26 结合 SAHI 实现切片推理。优化内存使用并提升大规模应用的检测精度。
keywords: YOLO26, SAHI, 切片推理, 目标检测, Ultralytics, 高分辨率图像, 计算效率, 集成指南
---

# Ultralytics 文档：将 YOLO26 与 SAHI 结合使用进行切片推理

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-ultralytics-yolo-with-sahi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开 SAHI 切片推理"></a>

欢迎阅读 Ultralytics 关于如何将 YOLO26 与 [SAHI](https://github.com/obss/sahi)（切片辅助超推理）结合使用的文档。本综合指南旨在为您提供实现 SAHI 与 YOLO26 并行运作所需的所有基本知识。我们将深入探讨 SAHI 是什么、为什么切片推理对大规模应用至关重要，以及如何将这些功能与 YOLO26 集成以提升[目标检测](https://www.ultralytics.com/glossary/object-detection)性能。

<p align="center">
  <img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/sahi-sliced-inference-overview.avif" alt="SAHI 针对小目标的切片推理">
</p>

## SAHI 简介

SAHI（切片辅助超推理）是一个创新库，旨在针对大规模和高分辨率图像优化目标检测算法。其核心功能在于将图像分割为可管理的切片，在每个切片上运行目标检测，然后将结果拼接在一起。SAHI 兼容多种目标检测模型，包括 YOLO 系列，从而提供灵活性，同时确保计算资源的优化利用。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ILqMBah5ZvI"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO26 进行 SAHI（切片辅助超推理）推理
</p>

### SAHI 的主要特性

- **无缝集成**：SAHI 与 YOLO 模型轻松集成，无需大量修改代码即可开始切片和检测。
- **资源高效**：通过将大图像分解为较小的部分，SAHI 优化了内存使用，使您能够在资源有限的硬件上运行高质量的检测。
- **高[准确率](https://www.ultralytics.com/glossary/accuracy)**：SAHI 在拼接过程中采用智能算法合并重叠的检测框，从而保持检测精度。

## 什么是切片推理？

切片推理是指将大尺寸或高分辨率图像细分为较小的片段（切片），在这些切片上进行目标检测，然后重新组合切片以重建原始图像上的目标位置。此技术在计算资源有限或处理极高分辨率图像（否则可能导致内存问题）的场景中极具价值。

### 切片推理的优势

- **降低计算负担**：较小的图像切片处理速度更快，消耗的内存更少，从而在低端硬件上也能实现更流畅的操作。

- **保持检测质量**：由于每个切片独立处理，只要切片足够大以捕获感兴趣的目标，目标检测质量就不会下降。

- **增强可扩展性**：该技术使目标检测能够更轻松地跨不同尺寸和分辨率的图像进行扩展，非常适合从卫星图像到医学诊断的广泛应用。

<table border="0">
  <tr>
    <th>不使用 SAHI 的 YOLO26</th>
    <th>使用 SAHI 的 YOLO26</th>
  </tr>
  <tr>
    <td><img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov8-without-sahi.avif" alt="不使用 SAHI 的 YOLO26" width="640"></td>
    <td><img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov8-with-sahi.avif" alt="使用 SAHI 的 YOLO26" width="640"></td>
  </tr>
</table>

## 安装与准备

### 安装

首先安装最新版本的 SAHI 和 Ultralytics：

```bash
pip install -U ultralytics sahi
```

### 导入模块和下载资源

以下是如何下载一些测试图像：

```python
from sahi.utils.file import download_from_url

# 下载测试图像
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)
```

## 使用 YOLO26 进行标准推理

### 实例化模型

您可以这样实例化一个用于目标检测的 YOLO26 模型：

```python
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.3,
    device="cpu",  # 或 'cuda:0'
)
```

### 执行标准预测

使用图像路径执行标准推理。

```python
from sahi.predict import get_prediction

result = get_prediction("demo_data/small-vehicles1.jpeg", detection_model)

result.export_visuals(export_dir="demo_data/", hide_conf=True)
```

### 可视化结果

导出并可视化预测的边界框和掩码：

```python
from PIL import Image

# 打开预测图像
processed_image = Image.open("demo_data/prediction_visual.png")

# 显示预测图像
processed_image.show()
```

## 使用 YOLO26 进行切片推理

通过指定切片尺寸和重叠比例来执行切片推理：

```python
from PIL import Image
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# 导出结果
result.export_visuals(export_dir="demo_data/", hide_conf=True)

# 打开预测图像
processed_image = Image.open("demo_data/prediction_visual.png")

# 显示预测图像
processed_image.show()
```

## 处理预测结果

SAHI 提供了 `PredictionResult` 对象，可转换为多种标注格式：

```python
# 访问目标预测列表
object_prediction_list = result.object_prediction_list

# 转换为 COCO 标注、COCO 预测、imantics 和 fiftyone 格式
result.to_coco_annotations()[:3]
result.to_coco_predictions(image_id=1)[:3]
result.to_imantics_annotations()[:3]
result.to_fiftyone_detections()[:3]
```

## 批量预测

对目录中的图像进行批量预测：

```python
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    model_device="cpu",  # 或 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

现在您已准备好将 YOLO26 与 SAHI 结合使用，进行标准推理和切片推理。

## 引用与致谢

如果您在研究或开发工作中使用了 SAHI，请引用原始 SAHI 论文并致谢作者：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{akyon2022sahi,
          title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
          author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
          journal={2022 IEEE International Conference on Image Processing (ICIP)},
          doi={10.1109/ICIP46576.2022.9897990},
          pages={966-970},
          year={2022}
        }
        ```

我们感谢 SAHI 研究小组为[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)社区创建和维护这一宝贵资源。有关 SAHI 及其作者的更多信息，请访问 [SAHI GitHub 仓库](https://github.com/obss/sahi)。

## 常见问题

### 如何在目标检测中将 YOLO26 与 SAHI 集成以进行切片推理？

将 Ultralytics YOLO26 与 SAHI（切片辅助超推理）集成进行切片推理，可通过将高分辨率图像分割为可管理的切片来优化目标检测任务。这种方法改善了内存使用并确保高检测精度。首先，需要安装 ultralytics 和 sahi 库：

```bash
pip install -U ultralytics sahi
```

然后下载测试图像：

```python
from sahi.utils.file import download_from_url

# 下载测试图像
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)
```

更多详细说明，请参阅我们的[切片推理指南](#使用-yolo26-进行切片推理)。

### 为什么要在大型图像的目标检测中使用 SAHI 与 YOLO26？

将 SAHI 与 Ultralytics YOLO26 结合用于大型图像的目标检测具有以下几个优势：

- **降低计算负担**：较小的切片处理速度更快、内存消耗更少，使得在资源有限的硬件上运行高质量检测变得可行。
- **保持检测精度**：SAHI 使用智能算法合并重叠框，保持检测质量。
- **增强可扩展性**：通过跨不同图像尺寸和分辨率扩展目标检测任务，SAHI 非常适合从卫星图像分析到医学诊断的各种应用。

在文档中了解更多关于[切片推理的优势](#切片推理的优势)。

### 使用 YOLO26 与 SAHI 时可以可视化预测结果吗？

可以，您可以在使用 YOLO26 与 SAHI 时可视化预测结果。以下是如何导出和可视化结果：

```python
from PIL import Image

result.export_visuals(export_dir="demo_data/", hide_conf=True)

processed_image = Image.open("demo_data/prediction_visual.png")

processed_image.show()
```

此命令将可视化的预测结果保存到指定目录，然后您可以加载该图像在笔记本或应用程序中查看。详细指南请参阅[标准推理部分](#可视化结果)。

### SAHI 为改进 YOLO26 目标检测提供了哪些功能？

SAHI（切片辅助超推理）提供了若干功能来补充 Ultralytics YOLO26 的目标检测能力：

- **无缝集成**：SAHI 与 YOLO 模型轻松集成，只需少量代码调整。
- **资源高效**：它将大图像分割为较小的切片，从而优化内存使用和速度。
- **高准确率**：通过在拼接过程中有效合并重叠的检测框，SAHI 保持高检测精度。

深入了解 SAHI 的[主要特性](#sahi-的主要特性)。

### 如何使用 YOLO26 和 SAHI 处理大规模推理项目？

要使用 YOLO26 和 SAHI 处理大规模推理项目，请遵循以下最佳实践：

1. **安装所需库**：确保您拥有最新版本的 ultralytics 和 sahi。
2. **配置切片推理**：为您的特定项目确定最佳的切片尺寸和重叠比例。
3. **运行批量预测**：使用 SAHI 的能力对目录中的图像进行批量预测，以提高效率。

批量预测示例：

```python
from sahi.predict import predict

predict(
    model_type="ultralytics",
    model_path="path/to/yolo26n.pt",
    model_device="cpu",  # 或 'cuda:0'
    model_confidence_threshold=0.4,
    source="path/to/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

更多详细步骤，请访问我们的[批量预测](#批量预测)章节。
