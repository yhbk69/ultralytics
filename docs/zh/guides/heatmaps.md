---
comments: true
description: 使用 Ultralytics YOLO26 将复杂数据转化为直观的热力图。通过生动的可视化发现模式、趋势和异常。
keywords: Ultralytics, YOLO26, 热力图, 数据可视化, 数据分析, 复杂数据, 模式, 趋势, 异常
---

# 高级[数据可视化](https://www.ultralytics.com/glossary/data-visualization)：使用 Ultralytics YOLO26 生成热力图 🚀

## 热力图简介

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-generate-heatmaps-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开热力图"></a>

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 生成的热力图将复杂数据转化为一个生动的、颜色编码的矩阵。这种可视化工具采用颜色光谱来表示不同的数据值，其中暖色调表示较高的强度，冷色调表示较低的值。热力图擅长可视化复杂的数据模式、相关性和异常，为跨不同领域的数据解释提供了一种易于理解和引人入胜的方法。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/4ezde5-nZZw"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO26 生成热力图
</p>

## 为什么选择热力图进行数据分析？

- **直观的数据分布可视化：** 热力图简化了对数据集中度和分布的理解，将复杂的数据集转化为易于理解的可视化格式。
- **高效的模式检测：** 通过以热力图格式可视化数据，更容易发现趋势、聚类和异常值，从而加快分析和洞察。
- **增强的空间分析和决策：** 热力图有助于说明空间关系，在商业智能、环境研究和城市规划等领域的决策过程中提供帮助。

## 实际应用

|                                                                    交通运输                                                                     |                                                                零售                                                                |
| :--------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO26 交通运输热力图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-transportation-heatmap.avif) | ![Ultralytics YOLO26 零售热力图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-retail-heatmap.avif) |
|                                                       Ultralytics YOLO26 交通运输热力图                                                       |                                                   Ultralytics YOLO26 零售热力图                                                   |

!!! example "使用 Ultralytics YOLO 生成热力图"

    === "CLI"

        ```bash
        # 运行热力图示例
        yolo solutions heatmap show=True

        # 传入源视频
        yolo solutions heatmap source="path/to/video.mp4"

        # 传入自定义颜色映射
        yolo solutions heatmap colormap=cv2.COLORMAP_INFERNO

        # 热力图 + 物体计数
        yolo solutions heatmap region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 对于带热力图的物体计数，可以传入区域点。
        # region_points = [(20, 400), (1080, 400)]                                      # 线点
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # 矩形区域
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # 多边形点

        # 初始化热力图对象
        heatmap = solutions.Heatmap(
            show=True,  # 显示输出
            model="yolo26n.pt",  # YOLO26 模型文件路径
            colormap=cv2.COLORMAP_PARULA,  # 热力图颜色映射
            # region=region_points,  # 带热力图的物体计数，可以传入 region_points
            # classes=[0, 2],  # 为特定类别生成热力图，例如人和车。
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或处理完成。")
                break

            results = heatmap(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧。

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `Heatmap()` 参数

以下是 `Heatmap` 参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "colormap", "show_in", "show_out", "region"]) }}

您也可以在 `Heatmap` 解决方案中应用不同的 `track` 参数。

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，支持的可视化参数如下：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

#### 热力图颜色映射

| 颜色映射名称                   | 描述                             |
| ------------------------------ | -------------------------------- |
| `cv::COLORMAP_AUTUMN`           | 秋季颜色映射                     |
| `cv::COLORMAP_BONE`             | 骨骼颜色映射                     |
| `cv::COLORMAP_JET`              | 喷气颜色映射                     |
| `cv::COLORMAP_WINTER`           | 冬季颜色映射                     |
| `cv::COLORMAP_RAINBOW`          | 彩虹颜色映射                     |
| `cv::COLORMAP_OCEAN`            | 海洋颜色映射                     |
| `cv::COLORMAP_SUMMER`           | 夏季颜色映射                     |
| `cv::COLORMAP_SPRING`           | 春季颜色映射                     |
| `cv::COLORMAP_COOL`             | 冷色颜色映射                     |
| `cv::COLORMAP_HSV`              | HSV（色调、饱和度、明度）颜色映射 |
| `cv::COLORMAP_PINK`             | 粉色颜色映射                     |
| `cv::COLORMAP_HOT`              | 热色颜色映射                     |
| `cv::COLORMAP_PARULA`           | Parula 颜色映射                  |
| `cv::COLORMAP_MAGMA`            | 岩浆颜色映射                     |
| `cv::COLORMAP_INFERNO`          | 地狱颜色映射                     |
| `cv::COLORMAP_PLASMA`           | 等离子颜色映射                   |
| `cv::COLORMAP_VIRIDIS`          | Viridis 颜色映射                 |
| `cv::COLORMAP_CIVIDIS`          | Cividis 颜色映射                 |
| `cv::COLORMAP_TWILIGHT`         | 暮光颜色映射                     |
| `cv::COLORMAP_TWILIGHT_SHIFTED` | 偏移暮光颜色映射                 |
| `cv::COLORMAP_TURBO`            | 涡轮颜色映射                     |
| `cv::COLORMAP_DEEPGREEN`        | 深绿颜色映射                     |

这些颜色映射通常用于以不同颜色表示来可视化数据。

## Ultralytics YOLO26 中热力图的工作原理

Ultralytics YOLO26 中的[热力图解决方案](../reference/solutions/heatmap.md)扩展了[物体计数器](../reference/solutions/object_counter.md)类，以生成和可视化视频流中的运动模式。初始化时，该解决方案创建一个空白的热力图层，随着物体在帧中移动而更新。

对于每个检测到的物体，该解决方案：

1. 使用 YOLO26 的跟踪功能跨帧跟踪物体
2. 更新物体位置的热力图强度
3. 应用选定的颜色映射来可视化强度值
4. 将着色的热力图叠加到原始帧上

结果是一个随时间累积的动态可视化，揭示了视频数据中的交通模式、人群运动或其他空间行为。

## 常见问题解答

### Ultralytics YOLO26 如何生成热力图？它们有什么好处？

Ultralytics YOLO26 通过将复杂数据转化为颜色编码的矩阵来生成热力图，其中不同的色调表示数据强度。热力图使数据中的模式、相关性和异常更容易可视化。暖色调表示较高的值，而冷色调表示较低的值。主要好处包括直观的数据分布可视化、高效的模式检测以及增强的用于决策的空间分析。有关更多详细信息和配置选项，请参阅[热力图配置](#heatmap-arguments)部分。

### 我能否使用 Ultralytics YOLO26 同时进行物体跟踪和生成热力图？

是的，Ultralytics YOLO26 支持同时进行物体跟踪和热力图生成。这可以通过其与物体跟踪模型集成的 `Heatmap` 解决方案实现。为此，您需要初始化热力图对象并使用 YOLO26 的跟踪功能。以下是一个简单示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=True, model="yolo26n.pt")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = heatmap(im0)
cap.release()
cv2.destroyAllWindows()
```

更多指导，请查看[跟踪模式](../modes/track.md)页面。

### Ultralytics YOLO26 热力图与 [OpenCV](https://www.ultralytics.com/glossary/opencv) 或 Matplotlib 等其他数据可视化工具有何不同？

Ultralytics YOLO26 热力图专门设计用于与其[物体检测](https://www.ultralytics.com/glossary/object-detection)和跟踪模型集成，为实时数据分析提供端到端解决方案。与 OpenCV 或 Matplotlib 等通用可视化工具不同，YOLO26 热力图针对性能和自动化处理进行了优化，支持持久跟踪、衰减因子调整和实时视频叠加等功能。有关 YOLO26 独特功能的更多信息，请访问[Ultralytics YOLO26 简介](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8)。

### 如何使用 Ultralytics YOLO26 在热力图中仅可视化特定的物体类别？

您可以通过在 YOLO 模型的 `track()` 方法中指定所需的类别来可视化特定的物体类别。例如，如果您只想可视化汽车和人（假设它们的类别索引为 0 和 2），可以相应地设置 `classes` 参数。

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
heatmap = solutions.Heatmap(show=True, model="yolo26n.pt", classes=[0, 2])

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = heatmap(im0)
cap.release()
cv2.destroyAllWindows()
```

### 为什么企业在数据分析中应选择 Ultralytics YOLO26 进行热力图生成？

Ultralytics YOLO26 提供了先进的物体检测和实时热力图生成的无缝集成，使其成为希望更有效可视化数据的企业的理想选择。主要优势包括直观的数据分布可视化、高效的模式检测以及增强的用于更好决策的空间分析。此外，YOLO26 的尖端功能，如持久跟踪、可自定义的颜色映射以及对各种导出格式的支持，使其在全面数据分析方面优于 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 和 OpenCV 等其他工具。了解更多商业应用，请访问[Ultralytics 计划](https://www.ultralytics.com/pricing)。