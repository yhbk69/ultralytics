---
comments: true
description: 学习使用 Ultralytics YOLO26 在人群分析和监控等应用中实时准确地识别和计数对象。
keywords: 对象计数, YOLO26, Ultralytics, 实时对象检测, AI, 深度学习, 对象跟踪, 人群分析, 监控, 资源优化
---

# 使用 Ultralytics YOLO26 进行对象计数

## 什么是对象计数？

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-count-the-objects-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开对象计数"></a>

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 进行对象计数涉及在视频和摄像头流中准确识别和计数特定对象。YOLO26 凭借其先进的算法和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)能力，在实时应用中表现出色，能够为人群分析和监控等各种场景提供高效且精确的对象计数。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/pJLXmhyuHzA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 进行实时对象计数 🚀
</p>

## 对象计数的优势

- **资源优化：** 对象计数通过提供准确的计数来促进高效的资源管理，优化[库存管理](https://docs.ultralytics.com/guides/analytics)等应用中的资源分配。
- **增强安全性：** 对象计数通过精确跟踪和计数实体来增强安全性和监控能力，有助于主动[威胁检测](https://docs.ultralytics.com/guides/security-alarm-system)。
- **明智决策：** 对象计数为决策提供有价值的洞察，优化零售、[交通管理](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination)以及许多其他领域的流程。

## 实际应用

|                                                                        物流                                                                         |                                                                          水产养殖                                                                          |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO26 进行传送带包裹计数](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/conveyor-belt-packets-counting.avif) | ![使用 Ultralytics YOLO26 进行海中鱼类计数](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/fish-counting-in-sea-using-ultralytics-yolov8.avif) |
|                                                 使用 Ultralytics YOLO26 进行传送带包裹计数                                                  |                                                         使用 Ultralytics YOLO26 进行海中鱼类计数                                                         |

!!! example "使用 Ultralytics YOLO 进行对象计数"

    === "CLI"

        ```bash
        # 运行计数示例
        yolo solutions count show=True

        # 传入源视频
        yolo solutions count source="path/to/video.mp4"

        # 传入区域坐标
        yolo solutions count region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

        `region` 参数接受两个点（用于直线）或具有三个或更多点的多边形。请按照点应该连接的顺序定义坐标，以便计数器准确地知道入口和出口发生的位置。

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # region_points = [(20, 400), (1080, 400)]                                      # 直线计数
        region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # 矩形区域
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # 多边形区域

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化对象计数器
        counter = solutions.ObjectCounter(
            show=True,  # 显示输出
            region=region_points,  # 传入区域点
            model="yolo26n.pt",  # 使用 OBB 模型进行对象计数时，使用 model="yolo26n-obb.pt"
            # classes=[0, 2],  # 计数特定类别，例如使用 COCO 预训练模型时的人和汽车
            # tracker="botsort.yaml",  # 选择跟踪器，例如 "bytetrack.yaml"
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = counter(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `ObjectCounter` 参数

以下是 `ObjectCounter` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "show_in", "show_out", "region"]) }}

`ObjectCounter` 解决方案允许使用多个 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题

### 如何使用 Ultralytics YOLO26 在视频中计数对象？

要使用 Ultralytics YOLO26 在视频中计数对象，可以按照以下步骤操作：

1. 导入必要的库（`cv2`、`ultralytics`）。
2. 定义计数区域（例如多边形、直线等）。
3. 设置视频捕获并初始化对象计数器。
4. 处理每一帧以跟踪对象并在定义的区域内进行计数。

以下是在区域内计数的简单示例：

```python
import cv2

from ultralytics import solutions


def count_objects_in_region(video_path, output_video_path, model_path):
    """在视频的特定区域内计数对象。"""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("path/to/video.mp4", "output_video.avi", "yolo26n.pt")
```

有关更高级的配置和选项，请查看 [RegionCounter 解决方案](https://docs.ultralytics.com/guides/region-counting)，了解如何在多个区域同时计数对象。

### 使用 Ultralytics YOLO26 进行对象计数有哪些优势？

使用 Ultralytics YOLO26 进行对象计数有以下几个优势：

1. **资源优化：** 通过提供准确的计数来促进高效的资源管理，帮助优化[库存管理](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management)等行业中的资源分配。
2. **增强安全性：** 通过精确跟踪和计数实体来增强安全性和监控能力，有助于主动威胁检测和[安全系统](https://docs.ultralytics.com/guides/security-alarm-system)建设。
3. **明智决策：** 为决策提供有价值的洞察，优化零售、交通管理等领域的流程。
4. **实时处理：** YOLO26 的架构支持[实时推理](https://www.ultralytics.com/glossary/real-time-inference)，使其适用于实时视频流和时间敏感的应用。

有关实现示例和实际应用，请探索 [TrackZone 解决方案](https://docs.ultralytics.com/guides/trackzone)，了解如何在特定区域内跟踪对象。

### 如何使用 Ultralytics YOLO26 计数特定类别的对象？

要使用 Ultralytics YOLO26 计数特定类别的对象，需要在跟踪阶段指定感兴趣的类别。以下是 Python 示例：

```python
import cv2

from ultralytics import solutions


def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """在视频中计数特定类别的对象。"""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points = [(20, 400), (1080, 400)]
    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_specific_classes("path/to/video.mp4", "output_specific_classes.avi", "yolo26n.pt", [0, 2])
```

在此示例中，`classes_to_count=[0, 2]` 表示计数类别 `0` 和 `2` 的对象（例如 COCO 数据集中的人和汽车）。有关类别索引的更多信息，请参见 [COCO 数据集文档](https://docs.ultralytics.com/datasets/detect/coco)。

### 为什么应该在实时应用中使用 YOLO26 而不是其他[对象检测](https://www.ultralytics.com/glossary/object-detection)模型？

Ultralytics YOLO26 相比其他对象检测模型（如 [Faster R-CNN](https://docs.ultralytics.com/compare/yolo26-vs-efficientdet)、SSD 以及之前的 YOLO 版本）具有以下几个优势：

1. **速度和效率：** YOLO26 提供实时处理能力，非常适合监控和[自动驾驶](https://www.ultralytics.com/blog/ai-in-self-driving-cars)等需要高速推理的应用。
2. **[准确性](https://www.ultralytics.com/glossary/accuracy)：** 为对象检测和跟踪任务提供最先进的准确性，减少误报数量并提高整体系统可靠性。
3. **易于集成：** YOLO26 支持与各种平台和设备无缝集成，包括移动设备和[边缘设备](https://docs.ultralytics.com/guides/nvidia-jetson)，这对现代 AI 应用至关重要。
4. **灵活性：** 支持对象检测、[分割](https://docs.ultralytics.com/tasks/segment)和跟踪等多种任务，并提供可配置的模型以满足特定用例的需求。

查阅 Ultralytics [YOLO26 文档](https://docs.ultralytics.com/models/yolo26)以深入了解其功能和性能对比。

### 我可以将 YOLO26 用于人群分析和交通管理等高级应用吗？

可以，Ultralytics YOLO26 凭借其实时检测能力、可扩展性和集成灵活性，非常适合人群分析和交通管理等高级应用。其先进功能支持在动态环境中进行高精度的对象跟踪、计数和分类。示例用例包括：

- **人群分析：** 监控和管理大型集会，通过[基于区域的计数](https://docs.ultralytics.com/guides/region-counting)确保安全并优化人流。
- **交通管理：** 利用[速度估算](https://docs.ultralytics.com/guides/speed-estimation)功能，跟踪和计数车辆、分析交通模式并实时管理拥堵。
- **零售分析：** 分析客户移动模式和产品交互，以优化商店布局并改善客户体验。
- **工业自动化：** 计数传送带上的产品，监控生产线以实现质量控制和效率提升。

对于更专业的应用，请探索 [Ultralytics 解决方案](https://docs.ultralytics.com/solutions)，获取专为实际计算机视觉挑战设计的一整套工具。
