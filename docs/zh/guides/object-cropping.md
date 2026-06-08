---
comments: true
description: 了解如何使用 Ultralytics YOLO26 进行对象裁剪与提取，实现聚焦分析、减少数据量并提升精度。
keywords: Ultralytics, YOLO26, 对象裁剪, 对象检测, 图像处理, 视频分析, AI, 机器学习
---

# 使用 Ultralytics YOLO26 进行对象裁剪

## 什么是对象裁剪？

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 进行对象裁剪，即从图像或视频中隔离并提取特定检测到的对象。借助 YOLO26 模型的能力，可以精确识别和圈定对象，从而实现精准裁剪，以便进行进一步分析或处理。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/J1BaCqytBmA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO 进行对象裁剪
</p>

## 对象裁剪的优势

- **聚焦分析**：YOLO26 支持有针对性的对象裁剪，可对场景中的单个项目进行深入检查或处理。
- **减少数据量**：仅提取相关对象，有助于最小化数据大小，使存储、传输或后续计算任务更高效。
- **增强精度**：YOLO26 的[对象检测](https://www.ultralytics.com/glossary/object-detection)[准确度](https://www.ultralytics.com/glossary/accuracy)可确保裁剪后的对象保持其空间关系，保留视觉信息的完整性，便于详细分析。

## 效果展示

|                                                                                 机场行李                                                                                  |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO26 在机场传送带裁剪行李箱](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/suitcases-cropping-airport-conveyor-belt.avif) |
|                                                       使用 Ultralytics YOLO26 在机场传送带裁剪行李箱                                                        |

!!! example "使用 Ultralytics YOLO 进行对象裁剪"

    === "CLI"

        ```bash
        # 裁剪对象
        yolo solutions crop show=True

        # 传入源视频
        yolo solutions crop source="path/to/video.mp4"

        # 裁剪特定类别
        yolo solutions crop classes="[0, 2]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # 初始化对象裁剪器
        cropper = solutions.ObjectCropper(
            show=True,  # 显示输出
            model="yolo26n.pt",  # 用于对象裁剪的模型，例如 yolo26x.pt
            classes=[0, 2],  # 裁剪特定类别，如使用 COCO 预训练模型裁剪人和车
            # conf=0.5,  # 调整对象的置信度阈值
            # crop_dir="cropped-detections",  # 设置裁剪结果的目录名称
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或处理已完成。")
                break

            results = cropper(im0)

            # print(results)  # 访问输出

        cap.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

        当提供可选的 `crop_dir` 参数时，每个裁剪后的对象都会写入该文件夹，文件名包含源图像名称和类别。这样可以方便地检查检测结果或构建下游数据集，无需额外编写代码。

### `ObjectCropper` 参数

以下是 `ObjectCropper` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "crop_dir"]) }}

此外，还可以使用以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## 常见问题

### Ultralytics YOLO26 中的对象裁剪是什么，它是如何工作的？

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 进行对象裁剪，即基于 YOLO26 的检测能力从图像或视频中隔离并提取特定对象。这一过程利用 YOLO26 以高精度识别对象并进行相应裁剪，从而实现聚焦分析、减少数据量并提升[精度](https://www.ultralytics.com/glossary/precision)。详细教程请参阅[对象裁剪示例](#object-cropping-using-ultralytics-yolo26)。

### 为什么应该选择 Ultralytics YOLO26 而非其他方案进行对象裁剪？

Ultralytics YOLO26 以其精度、速度和易用性脱颖而出。它能够进行详细而准确的对象检测与裁剪，对于需要高数据完整性的[聚焦分析](#advantages-of-object-cropping)和应用至关重要。此外，YOLO26 可与 [OpenVINO](../integrations/openvino.md) 和 [TensorRT](../integrations/tensorrt.md) 等工具无缝集成，用于需要在多种硬件上实现实时能力和优化的部署场景。更多优势请参阅[模型导出指南](../modes/export.md)。

### 如何使用对象裁剪来减少数据集的数据量？

通过使用 Ultralytics YOLO26 仅裁剪图像或视频中的相关对象，可以显著减小数据大小，使存储和处理更加高效。这一过程涉及训练模型检测特定对象，然后利用结果仅裁剪并保存这些部分。有关如何充分利用 Ultralytics YOLO26 能力的更多信息，请访问我们的[快速入门指南](../quickstart.md)。

### 能否使用 Ultralytics YOLO26 进行实时视频分析和对象裁剪？

可以，Ultralytics YOLO26 可以处理实时视频流，动态检测并裁剪对象。该模型的高速推理能力使其非常适合[安防监控](security-alarm-system.md)、体育分析和自动化检测系统等实时应用。请查看[跟踪模式](../modes/track.md)和[预测模式](../modes/predict.md)以了解如何实现实时处理。

### 高效运行 YOLO26 进行对象裁剪需要什么硬件配置？

Ultralytics YOLO26 同时针对 CPU 和 GPU 环境进行了优化，但为了获得最佳性能，特别是在实时或大批量推理场景下，建议使用专用 GPU（如 NVIDIA Tesla、RTX 系列）。若要在轻量级设备上部署，可考虑使用适用于 iOS 的 [CoreML](../integrations/coreml.md) 或适用于 Android 的 [TFLite](../integrations/tflite.md)。更多关于支持的设备和格式的详细信息，请参阅我们的[模型部署选项](../guides/model-deployment-options.md)。
