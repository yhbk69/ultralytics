---
comments: true
description: 了解如何使用 YOLO 进行多目标跟踪。探索数据集格式、跟踪算法以及使用 Python 或 CLI 进行实时目标跟踪的实现示例。
keywords: YOLO, 多目标跟踪, 跟踪数据集, Python 跟踪示例, CLI 跟踪示例, 目标检测, Ultralytics, AI, 机器学习, BoT-SORT, ByteTrack
---

# 多目标跟踪数据集概览

多目标跟踪是视频分析中的关键组成部分，它能够识别目标并为视频帧中每个检测到的目标维护唯一 ID。Ultralytics YOLO 提供强大的跟踪能力，可应用于监控、体育分析和交通监控等多个领域。

## 数据集格式（即将推出）

Ultralytics 跟踪目前复用检测、分割或姿态模型，无需进行跟踪器专用训练。原生的跟踪器训练支持正在积极开发中。

## 可用跟踪器

Ultralytics YOLO 支持以下跟踪算法：

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 使用 `botsort.yaml` 启用此跟踪器（默认）
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) - 使用 `bytetrack.yaml` 启用此跟踪器

## 使用方法

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.1, iou=0.7, show=True)
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.1 iou=0.7 show=True
        ```

## 跨帧持久化跟踪

要在视频帧间实现连续跟踪，可使用 `persist=True` 参数：

!!! example

    === "Python"

        ```python
        import cv2

        from ultralytics import YOLO

        # 加载 YOLO 模型
        model = YOLO("yolo26n.pt")

        # 打开视频文件
        cap = cv2.VideoCapture("path/to/video.mp4")

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # 跨帧运行持久化跟踪
                results = model.track(frame, persist=True)

                # 可视化结果
                annotated_frame = results[0].plot()
                cv2.imshow("Tracking", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        ```

## 常见问题

### 如何使用 Ultralytics YOLO 进行多目标跟踪？

要使用 Ultralytics YOLO 进行多目标跟踪，可以从以下 Python 或 CLI 示例开始：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # 加载 YOLO26 模型
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.1, iou=0.7, show=True)
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.1 iou=0.7 show=True
        ```

这些命令加载 YOLO26 模型，并使用指定的置信度（`conf`）和[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou)（`iou`）阈值对给定视频源进行目标跟踪。更多详情请参阅[跟踪模式文档](../../modes/track.md)。

### Ultralytics 跟踪器训练的后续功能有哪些？

Ultralytics 持续增强其 AI 模型。即将推出的功能将支持独立跟踪器的训练。在此之前，多目标检测器使用预训练的检测、分割或姿态模型进行跟踪，无需独立训练。请关注我们的[博客](https://www.ultralytics.com/blog)或查看[即将推出的功能](../../reference/trackers/track.md)以获取最新动态。

### 为什么应该使用 Ultralytics YOLO 进行多目标跟踪？

Ultralytics YOLO 是一款最先进的[目标检测](https://www.ultralytics.com/glossary/object-detection)模型，以其实时性能和高[准确率](https://www.ultralytics.com/glossary/accuracy)著称。使用 YOLO 进行多目标跟踪具有以下优势：

- **实时跟踪：** 实现高效且高速的跟踪，非常适合动态环境。
- **预训练模型的灵活性：** 无需从头训练，直接使用预训练的检测、分割或姿态模型。
- **易于使用：** 简单的 API 集成，同时支持 Python 和 CLI，使跟踪流程的设置变得简单明了。
- **丰富的文档和社区支持：** Ultralytics 提供全面的文档和活跃的社区论坛，帮助排查问题并优化跟踪模型。

有关设置和使用 YOLO 进行跟踪的更多详情，请访问我们的[跟踪使用指南](../../modes/track.md)。

### 可以使用自定义数据集进行 Ultralytics YOLO 多目标跟踪吗？

可以，您可以使用自定义数据集进行 Ultralytics YOLO 的多目标跟踪。虽然独立跟踪器训练的支持即将推出，但您已经可以在自定义数据集上使用预训练模型。请按照文档以兼容 YOLO 的格式准备数据集，并进行集成。

### 如何解读 Ultralytics YOLO 跟踪模型的结果？

使用 Ultralytics YOLO 运行跟踪任务后，结果包含各种数据点，如跟踪目标 ID、边界框和置信度分数。以下是解读这些结果的简要概述：

- **跟踪 ID：** 每个目标被分配一个唯一 ID，有助于跨帧跟踪。
- **边界框：** 这些指示了帧内被跟踪目标的位置。
- **置信度分数：** 这些反映了模型对检测到被跟踪目标的置信程度。

有关解读和可视化这些结果的详细指南，请参阅[结果处理指南](../../reference/engine/results.md)。

### 如何自定义跟踪器配置？

您可以通过创建修改后的跟踪器配置文件来自定义跟踪器。从 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 复制现有配置文件，根据需要修改参数，并在运行跟踪器时指定该文件：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.track(source="video.mp4", tracker="custom_tracker.yaml")
```
