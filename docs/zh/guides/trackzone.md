---
comments: true
description: 了解 TrackZone 如何利用 Ultralytics YOLO26 精确跟踪特定区域内的物体，为人群分析、监控和定向监测提供实时洞察。
keywords: TrackZone, 物体跟踪, YOLO26, Ultralytics, 实时物体检测, AI, 深度学习, 人群分析, 监控, 区域跟踪, 资源优化
---

# 使用 Ultralytics YOLO26 的 TrackZone

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-track-the-objects-in-zone-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open TrackZone In Colab"></a>

## 什么是 TrackZone？

TrackZone 专注于监控画面中指定区域内的物体，而非整个画面。它基于 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/)，将物体检测与跟踪集成到视频和实时摄像头画面的特定区域内。YOLO26 的先进算法和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)技术使其成为实时应用场景的理想选择，在人群监测和监控等应用中提供精确高效的物体跟踪。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/SMSJvjUG1ko"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 在区域内跟踪物体 | TrackZone 🚀
</p>

## 区域物体跟踪（TrackZone）的优势

- **定向分析：** 在特定区域内跟踪物体可以获得更聚焦的洞察，实现对关注区域（如入口点或限制区域）的精确监测和分析。
- **提高效率：** 通过将跟踪范围缩小到指定区域，TrackZone 减少了计算开销，确保更快的处理速度和最佳性能。
- **增强安全性：** 区域跟踪通过监测关键区域来改善监控效果，有助于及早发现异常活动或安全漏洞。
- **可扩展方案：** 聚焦特定区域的能力使 TrackZone 能够适应从零售空间到工业场所等多种场景，确保无缝集成和可扩展性。

## 实际应用场景

|                                                                             农业                                                                              |                                                                             交通运输                                                                             |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO26 跟踪田间植物](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/plants-tracking-in-zone-using-ultralytics-yolo11.avif) | ![使用 Ultralytics YOLO26 跟踪道路车辆](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vehicle-tracking-in-zone-using-ultralytics-yolo11.avif) |
|                                                          使用 Ultralytics YOLO26 跟踪田间植物                                                           |                                                           使用 Ultralytics YOLO26 跟踪道路车辆                                                           |

!!! example "使用 Ultralytics YOLO 的 TrackZone 示例"

    === "CLI"

        ```bash
        # 运行 trackzone 示例
        yolo solutions trackzone show=True

        # 传入源视频
        yolo solutions trackzone source="path/to/video.mp4" show=True

        # 传入区域坐标
        yolo solutions trackzone show=True region="[(150, 150), (1130, 150), (1130, 570), (150, 570)]"
        ```

        TrackZone 依赖 `region` 列表来指定需要监测的画面区域。定义多边形以匹配你关心的物理区域（门、闸口等），并在配置时保持 `show=True` 开启，以便验证叠加层是否与视频画面对齐。

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # 定义区域点
        region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化 trackzone（区域内物体跟踪，而非整个画面）
        trackzone = solutions.TrackZone(
            show=True,  # 显示输出
            region=region_points,  # 传入区域点
            model="yolo26n.pt",  # 可使用 Ultralytics 支持的任何模型，如 YOLOv9、YOLOv10
            # line_width=2,  # 调整边界框和文本显示的线宽
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = trackzone(im0)

            # print(results)  # 访问输出结果

            video_writer.write(results.plot_im)  # 写入视频文件

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 关闭所有打开的窗口
        ```

### `TrackZone` 参数

以下是 `TrackZone` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

TrackZone 方案还支持以下 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还提供以下可视化选项：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题

### 如何使用 Ultralytics YOLO26 在视频画面的特定区域或区域内跟踪物体？

使用 Ultralytics YOLO26 在视频画面的指定区域跟踪物体非常简单。只需使用以下命令即可启动跟踪。这种方法确保高效分析和准确结果，非常适合监控、人群管理或任何需要区域跟踪的场景。

```bash
yolo solutions trackzone source="path/to/video.mp4" show=True
```

### 如何在 Python 中使用 Ultralytics YOLO26 的 TrackZone？

只需几行代码，即可在特定区域内设置物体跟踪，轻松集成到你的项目中。

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 定义区域点
region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

# 视频写入器
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 初始化 trackzone（区域内物体跟踪，而非整个画面）
trackzone = solutions.TrackZone(
    show=True,  # 显示输出
    region=region_points,  # 传入区域点
    model="yolo26n.pt",
)

# 处理视频
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = trackzone(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

### 如何使用 Ultralytics TrackZone 配置视频处理的区域点？

使用 Ultralytics TrackZone 配置视频处理的区域点既简单又可定制。你可以直接通过 Python 脚本定义和调整区域，从而精确控制要监测的区域。

```python
# 定义区域点
region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

# 初始化 trackzone
trackzone = solutions.TrackZone(
    show=True,  # 显示输出
    region=region_points,  # 传入区域点
)
```
