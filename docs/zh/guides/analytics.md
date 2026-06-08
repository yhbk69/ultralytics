---
comments: true
description: 学习使用 Python 创建折线图、柱状图和饼图，包含详细指导与代码示例。最大化你的数据可视化技能！
keywords: Ultralytics, YOLO26, 数据可视化, 折线图, 柱状图, 饼图, Python, 分析, 教程, 指南
---

# 使用 Ultralytics YOLO26 进行分析

## 简介

本指南全面介绍了三种基本的[数据可视化](https://www.ultralytics.com/glossary/data-visualization)类型：折线图、柱状图和饼图。每个部分都包含使用 Python 创建这些可视化的分步说明和代码片段。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/tVuLIMt4DMY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 生成分析图表 | 折线图、柱状图、面积图和饼图
</p>

### 可视化示例

|                                                              折线图                                                              |                                                             柱状图                                                             |                                                               饼图                                                               |
| :------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------: |
| ![YOLO 目标追踪分析折线图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-line-graph.avif) | ![YOLO 检测计数分析柱状图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-bar-plot.avif) | ![YOLO 类别分布分析饼图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-pie-chart.avif) |

### 为什么图表很重要

- 折线图非常适合追踪短期和长期的变化，以及比较多个组在同一时期内的变化趋势。
- 柱状图则适用于比较不同类别之间的数量差异，展示类别与其数值之间的关系。
- 饼图则能有效地展示类别之间的比例关系，展示各部分在整体中的占比。

!!! example "使用 Ultralytics YOLO 进行分析"

    === "CLI"

        ```bash
        yolo solutions analytics show=True

        # 指定源
        yolo solutions analytics source="path/to/video.mp4"

        # 生成饼图
        yolo solutions analytics analytics_type="pie" show=True

        # 生成柱状图
        yolo solutions analytics analytics_type="bar" show=True

        # 生成面积图
        yolo solutions analytics analytics_type="area" show=True
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            "analytics_output.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (1280, 720),  # 固定尺寸
        )

        # 初始化分析对象
        analytics = solutions.Analytics(
            show=True,  # 显示输出
            analytics_type="line",  # 指定分析类型，可选 "pie"、"bar" 或 "area"
            model="yolo26n.pt",  # YOLO26 模型文件路径
            # classes=[0, 2],  # 仅对特定检测类别进行分析
        )

        # 处理视频
        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if success:
                frame_count += 1
                results = analytics(im0, frame_count)  # 每帧更新分析图表

                # print(results)  # 查看输出

                out.write(results.plot_im)  # 写入视频文件
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `Analytics` 参数

以下是 Analytics 参数的概览：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "analytics_type"]) }}

你也可以在 `Analytics` 方案中使用不同的 [`track`](../modes/track.md) 参数。

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## 总结

理解何时以及如何使用不同类型的可视化对于有效的数据分析至关重要。折线图、柱状图和饼图是帮助你更清晰有效地传达数据故事的基础工具。Ultralytics YOLO26 Analytics 方案提供了一种简化的方式，从你的[目标检测](https://www.ultralytics.com/glossary/object-detection)和追踪结果中生成这些可视化，让你能更轻松地从视觉数据中提取有意义的洞察。

## 常见问题

### 如何使用 Ultralytics YOLO26 Analytics 创建折线图？

使用 Ultralytics YOLO26 Analytics 创建折线图，请按照以下步骤操作：

1. 加载 YOLO26 模型并打开你的视频文件。
2. 初始化 `Analytics` 类，将类型设置为 "line"。
3. 遍历视频帧，使用相关数据（如每帧的目标计数）更新折线图。
4. 保存显示折线图的输出视频。

示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # 固定尺寸
)

analytics = solutions.Analytics(
    analytics_type="line",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        results = analytics(im0, frame_count)  # 每帧更新分析图表
        out.write(results.plot_im)  # 写入视频文件
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

有关配置 `Analytics` 类的更多详情，请参阅[使用 Ultralytics YOLO26 进行分析](#使用-ultralytics-yolo26-进行分析)部分。

### 使用 Ultralytics YOLO26 创建柱状图有哪些优势？

使用 Ultralytics YOLO26 创建柱状图具有以下优势：

1. **实时数据可视化**：将[目标检测](https://www.ultralytics.com/glossary/object-detection)结果无缝集成到柱状图中，实现动态更新。
2. **易于使用**：简洁的 API 和函数使实现和数据可视化变得简单直接。
3. **可定制化**：可自定义标题、标签、颜色等，以满足你的特定需求。
4. **高效**：高效处理大量数据，并在视频处理过程中实时更新图表。

使用以下示例生成柱状图：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # 固定尺寸
)

analytics = solutions.Analytics(
    analytics_type="bar",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        results = analytics(im0, frame_count)  # 每帧更新分析图表
        out.write(results.plot_im)  # 写入视频文件
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

了解更多信息，请参阅指南中的[柱状图](#可视化示例)部分。

### 为什么应该在数据可视化项目中使用 Ultralytics YOLO26 创建饼图？

Ultralytics YOLO26 是创建饼图的绝佳选择，原因如下：

1. **与目标检测集成**：直接将目标检测结果集成到饼图中，获得即时洞察。
2. **用户友好的 API**：设置和使用简单，代码量极少。
3. **可定制化**：提供多种颜色、标签等自定义选项。
4. **实时更新**：实时处理和可视化数据，非常适合视频分析项目。

快速示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # 固定尺寸
)

analytics = solutions.Analytics(
    analytics_type="pie",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        results = analytics(im0, frame_count)  # 每帧更新分析图表
        out.write(results.plot_im)  # 写入视频文件
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

更多信息请参阅指南中的[饼图](#可视化示例)部分。

### Ultralytics YOLO26 能否用于追踪目标并动态更新可视化？

是的，Ultralytics YOLO26 可用于追踪目标并动态更新可视化。它支持实时追踪多个目标，并可根据追踪到的目标数据动态更新各种可视化图表，如折线图、柱状图和饼图。

追踪并更新折线图的示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # 固定尺寸
)

analytics = solutions.Analytics(
    analytics_type="line",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        results = analytics(im0, frame_count)  # 每帧更新分析图表
        out.write(results.plot_im)  # 写入视频文件
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

了解完整功能，请参阅[追踪](../modes/track.md)部分。

### Ultralytics YOLO26 与 [OpenCV](https://www.ultralytics.com/glossary/opencv) 和 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 等其他目标检测方案有何不同？

Ultralytics YOLO26 在以下多个方面优于 OpenCV 和 TensorFlow 等目标检测方案：

1. **最先进的[准确率](https://www.ultralytics.com/glossary/accuracy)**：YOLO26 在目标检测、分割和分类任务中提供卓越的准确率。
2. **易于使用**：用户友好的 API 允许快速实现和集成，无需大量编码。
3. **实时性能**：针对高速推理进行了优化，适用于实时应用。
4. **多样化应用**：支持多种任务，包括多目标追踪、自定义模型训练以及导出为 ONNX、TensorRT 和 CoreML 等不同格式。
5. **全面的文档**：丰富的[文档](https://docs.ultralytics.com/)和[博客资源](https://www.ultralytics.com/blog)为用户提供每一步的指导。

更多详细比较和用例，请浏览我们的 [Ultralytics 博客](https://www.ultralytics.com/blog/ai-use-cases-transforming-your-future)。