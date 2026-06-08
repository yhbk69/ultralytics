---
comments: true
description: 了解如何使用 Ultralytics YOLO26 在指定区域内进行精确的对象计数，从而提升各种应用场景的效率。
keywords: 对象计数, 区域, YOLO26, 计算机视觉, Ultralytics, 效率, 准确性, 自动化, 实时, 应用, 监控, 监测
---

# 使用 Ultralytics YOLO 在不同区域进行对象计数 🚀

## 什么是区域对象计数？

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 进行区域[对象计数](../guides/object-counting.md)，是通过先进的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)技术精确确定指定区域内对象数量的方法。这种方法对于优化流程、增强安全性和提高各种应用场景的效率非常有价值。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/mzLfC13ISF4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO26 在不同区域进行对象计数 | Ultralytics 解决方案 🚀
</p>

## 区域对象计数的优势

- **[精确度](https://www.ultralytics.com/glossary/precision)与准确性：** 采用先进计算机视觉的区域对象计数可确保精确且准确的计数，最大限度地减少手动计数中常见的错误。
- **效率提升：** 自动化对象计数提高了运营效率，提供实时结果并简化不同应用场景的流程。
- **通用性与应用广泛性：** 区域对象计数的通用性使其适用于从制造业、监控到交通监测等各个领域，从而发挥广泛的实用性和有效性。

## 实际应用场景

|                                                                                       零售                                                                                        |                                                                                  商业街区                                                                                  |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO26 在不同区域进行人员计数](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/people-counting-different-region-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO26 在不同区域进行人群计数](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/crowd-counting-different-region-ultralytics-yolov8.avif) |
|                                                            使用 Ultralytics YOLO26 在不同区域进行人员计数                                                            |                                                           使用 Ultralytics YOLO26 在不同区域进行人群计数                                                            |

## 使用示例

!!! example "使用 Ultralytics YOLO 进行区域计数"

    === "Python"

         ```python
         import cv2

         from ultralytics import solutions

         cap = cv2.VideoCapture("path/to/video.mp4")
         assert cap.isOpened(), "Error reading video file"

         # 以列表形式传入区域
         # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

         # 以字典形式传入区域
         region_points = {
             "region-01": [(50, 50), (250, 50), (250, 250), (50, 250)],
             "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)],
         }

         # 视频写入器
         w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
         video_writer = cv2.VideoWriter("region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

         # 初始化区域计数器对象
         regioncounter = solutions.RegionCounter(
             show=True,  # 显示帧
             region=region_points,  # 传入区域坐标点
             model="yolo26n.pt",  # 区域计数模型，例如 yolo26s.pt
         )

         # 处理视频
         while cap.isOpened():
             success, im0 = cap.read()

             if not success:
                 print("Video frame is empty or processing is complete.")
                 break

             results = regioncounter(im0)

             # print(results)  # 访问输出结果

             video_writer.write(results.plot_im)

         cap.release()
         video_writer.release()
         cv2.destroyAllWindows()  # 销毁所有已打开的窗口
         ```

    === "CLI"

         ```bash
         yolo solutions region source="path/to/video.mp4" show=True region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
         ```

!!! tip "Ultralytics 示例代码"

      Ultralytics 区域计数模块可在我们的[示例专区](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py)中找到。您可以探索此示例进行代码定制，并根据具体的使用场景进行修改。

### `RegionCounter` 参数

以下是 `RegionCounter` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

`RegionCounter` 解决方案支持使用对象追踪参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化设置：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题解答

### 什么是使用 Ultralytics YOLO26 在指定区域进行对象计数？

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 在指定区域进行对象计数，是指使用先进的计算机视觉技术检测并统计定义区域内的对象数量。这种精确的方法在制造业、监控和交通监测等各种应用场景中提高了效率和[准确性](https://www.ultralytics.com/glossary/accuracy)。

### 如何使用 Ultralytics YOLO26 运行基于区域的对象计数脚本？

按照以下步骤运行 Ultralytics YOLO26 的区域对象计数：

1. 克隆 Ultralytics 仓库并进入相应目录：

    ```bash
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics/examples/YOLOv8-Region-Counter
    ```

2. 执行区域计数脚本：
    ```bash
    python yolov8_region_counter.py --source "path/to/video.mp4" --save-img
    ```

更多选项请参阅[使用示例](#usage-examples)部分。

### 为什么应该使用 Ultralytics YOLO26 进行区域对象计数？

使用 Ultralytics YOLO26 进行区域对象计数具有以下优势：

1. **实时处理：** YOLO26 的架构支持快速推理，非常适合需要即时计数结果的应用场景。
2. **灵活的区域定义：** 该解决方案允许您将多个自定义区域定义为多边形、矩形或线条，以满足特定的监控需求。
3. **多类别支持：** 在同一区域内同时统计不同类型的对象，提供全面的分析数据。
4. **集成能力：** 通过 Ultralytics Python API 或命令行接口轻松与现有系统集成。

在[优势](#advantages-of-object-counting-in-regions)部分深入了解更多优点。

### 区域对象计数有哪些实际应用场景？

使用 Ultralytics YOLO26 进行区域对象计数可应用于众多实际场景：

- **零售分析：** 统计店铺不同区域内的顾客数量，以优化布局和人员配置。
- **交通管理：** 监控特定路段或路口的车流量。
- **制造业：** 追踪产品在不同生产区域的流转情况。
- **仓储运营：** 统计指定存储区域的库存物品数量。
- **公共安全：** 在活动期间监控特定区域的人群密度。

在[实际应用场景](#real-world-applications)部分以及 [TrackZone](../guides/trackzone.md) 解决方案中查看更多示例，了解基于区域的监控功能。
