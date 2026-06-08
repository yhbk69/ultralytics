---
comments: true
description: 探索 Ultralytics YOLO26 支持的检测、分割、分类、OBB 和姿态估计任务，兼具高精度与高速度。了解如何应用每项任务。
keywords: Ultralytics YOLO26, 检测, 分割, 分类, 旋转目标检测, 姿态估计, 计算机视觉, AI 框架
---

# Ultralytics YOLO26 支持的计算机视觉任务

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-tasks-banner.avif" alt="Ultralytics YOLO 支持的计算机视觉任务">

Ultralytics YOLO26 是一个多功能 AI 框架，支持多种[计算机视觉](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025) **任务**。该框架可用于执行[检测](detect.md)、[分割](segment.md)、[OBB](obb.md)、[分类](classify.md)和[姿态](pose.md)估计。每项任务都有不同的目标和使用场景，让您能够通过单一框架应对各种计算机视觉挑战。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>探索 Ultralytics YOLO 任务：<a href="https://www.ultralytics.com/blog/a-guide-to-deep-dive-into-object-detection-in-2025">目标检测</a>、分割、OBB、跟踪和姿态估计。
</p>

## [检测](detect.md)

检测是 YOLO26 支持的主要任务。它涉及识别图像或视频帧中的物体，并在其周围绘制边界框。检测到的物体根据其特征被分类到不同的类别中。YOLO26 能够以高[准确率](https://www.ultralytics.com/glossary/accuracy)和速度在单张图像或视频帧中检测多个物体，非常适合[监控系统](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai)和[自动驾驶](https://www.ultralytics.com/solutions/ai-in-automotive)等实时应用场景。

[检测示例](detect.md){ .md-button }

## [图像分割](segment.md)

分割在目标检测的基础上更进一步，为每个物体生成像素级的掩膜。这种精度对于[医学影像](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency)、[农业分析](https://www.ultralytics.com/blog/from-farm-to-table-how-ai-drives-innovation-in-agriculture)和[制造业质量控制](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision)等应用尤为有用。

[分割示例](segment.md){ .md-button }

## [分类](classify.md)

分类涉及根据图像内容对整个图像进行分类。该任务对于电子商务中的[产品分类](https://www.ultralytics.com/blog/understanding-vision-language-models-and-their-applications)、[内容审核](https://www.ultralytics.com/blog/ai-in-document-authentication-with-image-segmentation)和[野生动物监测](https://www.ultralytics.com/blog/monitoring-animal-behavior-using-ultralytics-yolov8)等应用至关重要。

[分类示例](classify.md){ .md-button }

## [姿态估计](pose.md)

姿态估计检测图像或视频帧中的特定关键点，以跟踪运动或估计姿态。这些关键点可以表示人体关节、面部特征或其他重要关注点。YOLO26 在关键点检测方面表现出色，具有高精度和高速度，对于[健身应用](https://www.ultralytics.com/blog/ai-in-our-day-to-day-health-and-fitness)、[体育分析](https://www.ultralytics.com/blog/exploring-the-applications-of-computer-vision-in-sports)和[人机交互](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-for-dog-pose-estimation)非常有价值。

[姿态示例](pose.md){ .md-button }

## [OBB](obb.md)

旋转边界框（Oriented Bounding Box，OBB）检测通过添加方向角度来增强传统目标检测，从而更好地定位旋转物体。此功能对于[航拍图像分析](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery)、[文档处理](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-smart-document-analysis)和[工业应用](https://www.ultralytics.com/blog/yolo11-enhancing-efficiency-conveyor-automation)等物体以不同角度出现的场景尤其有价值。YOLO26 能够在各种场景下高精度、高速度地检测旋转物体。

[旋转检测](obb.md){ .md-button }

## 总结

Ultralytics YOLO26 支持多种计算机视觉任务，包括检测、分割、分类、旋转目标检测和关键点检测。每项任务都针对计算机视觉领域中的特定需求，从基本的物体识别到详细的姿态分析。通过了解每项任务的能力和应用场景，您可以为特定的计算机视觉挑战选择最合适的方法，并利用 YOLO26 的强大功能构建有效的解决方案。

## 常见问题

### Ultralytics YOLO26 可以执行哪些计算机视觉任务？

Ultralytics YOLO26 是一个多功能 AI 框架，能够以高精度和高速度执行多种计算机视觉任务。这些任务包括：

- **[目标检测](detect.md)：** 通过在物体周围绘制边界框来识别和定位图像或视频帧中的物体。
- **[图像分割](segment.md)：** 根据内容将图像分割成不同区域，适用于医学影像等应用。
- **[分类](classify.md)：** 根据内容对整个图像进行分类。
- **[姿态估计](pose.md)：** 检测图像或视频帧中的特定关键点，以跟踪运动或估计姿态。
- **[旋转目标检测 (OBB)](obb.md)：** 通过添加方向角度来检测旋转物体，以获得更高的精度。

### 如何使用 Ultralytics YOLO26 进行目标检测？

要使用 Ultralytics YOLO26 进行目标检测，请按照以下步骤操作：

1. 以适当的格式准备您的数据集。
2. 使用检测任务训练 YOLO26 模型。
3. 通过输入新的图像或视频帧，使用模型进行预测。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLO 模型（根据需要调整模型类型）
        model = YOLO("yolo26n.pt")  # 可选 n、s、m、l、x 版本

        # 对图像执行目标检测
        results = model.predict(source="image.jpg")  # 也可以使用视频、目录、URL 等

        # 显示结果
        results[0].show()  # 显示第一张图像的结果
        ```

    === "CLI"

        ```bash
        # 从命令行运行 YOLO 检测
        yolo detect model=yolo26n.pt source="image.jpg" # 根据需要调整模型和数据源
        ```

有关更详细的说明，请查看我们的[检测示例](detect.md)。

### 使用 YOLO26 进行分割任务有哪些优势？

使用 YOLO26 进行分割任务具有以下几个优势：

1. **高精度：** 分割任务提供精确的像素级掩膜。
2. **速度快：** YOLO26 针对实时应用进行了优化，即使对高分辨率图像也能快速处理。
3. **广泛应用：** 非常适合医学影像、自动驾驶以及其他需要详细图像分割的应用。

在[图像分割章节](segment.md)中了解更多关于 YOLO26 分割的优势和使用场景。

### Ultralytics YOLO26 能否处理姿态估计和关键点检测？

是的，Ultralytics YOLO26 能够以高精度和高速度有效地执行姿态估计和关键点检测。此功能对于体育分析、医疗保健和人机交互应用中的运动跟踪特别有用。YOLO26 可以检测图像或视频帧中的关键点，从而实现精确的姿态估计。

有关更多细节和实现技巧，请访问我们的[姿态估计示例](pose.md)。

### 为什么应该选择 Ultralytics YOLO26 进行旋转目标检测 (OBB)？

使用 YOLO26 的旋转目标检测 (OBB) 通过附加的角度参数来检测物体，从而提供更高的[精确率](https://www.ultralytics.com/glossary/precision)。此功能对于需要准确定位旋转物体的应用非常有益，例如航拍图像分析和仓库自动化。

- **更高的精确率：** 角度分量减少了旋转物体的误检。
- **多样化的应用：** 适用于地理空间分析、机器人等任务。

查看[旋转目标检测章节](obb.md)了解更多细节和示例。
