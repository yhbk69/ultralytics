---
comments: true
description: 探索 Ultralytics 解决方案，使用 YOLO26 实现目标计数、模糊处理、安全系统等功能。利用尖端 AI 提升效率并解决现实世界问题。
keywords: Ultralytics, YOLO26, 目标计数, 目标模糊, 安全系统, AI 解决方案, 实时分析, 计算机视觉应用
---

# Ultralytics 解决方案：利用 YOLO26 解决现实世界问题

Ultralytics 解决方案提供 YOLO 模型的尖端应用，包括目标计数、模糊处理和安全系统等现实世界解决方案，帮助各行各业提升效率和[准确性](https://www.ultralytics.com/glossary/accuracy)。探索 YOLO26 在实用、高效部署中的强大能力。

![Ultralytics 解决方案缩略图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-solutions-thumbnail.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/bjkt5OE_ANA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何通过命令行运行 Ultralytics 解决方案 | Ultralytics YOLO26 🚀
</p>

## 解决方案

以下是我们精心整理的 Ultralytics 解决方案列表，可用于创建出色的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)项目。

- [数据分析](../guides/analytics.md)：利用 YOLO26 进行全面的数据分析，发现模式并做出明智决策，涵盖描述性、预测性和规范性分析。
- [距离计算](../guides/distance-calculation.md)：利用 YOLO26 中[边界框](https://www.ultralytics.com/glossary/bounding-box)中心点计算物体之间的距离，是空间分析的基础。
- [热力图](../guides/heatmaps.md)：利用检测热力图可视化矩阵中的数据强度，为计算机视觉任务提供清晰的洞察。
- [实例分割与目标跟踪](../guides/instance-segmentation-and-tracking.md)：使用 YOLO26 实现[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)与目标跟踪，实现精确的目标边界提取和持续监控。
- [使用 Streamlit 实时推理](../guides/streamlit-live-inference.md)：通过用户友好的 Streamlit 界面，直接在浏览器中利用 YOLO26 进行实时[目标检测](https://www.ultralytics.com/glossary/object-detection)。
- [目标模糊](../guides/object-blurring.md)：使用 YOLO26 对图像和视频中的目标进行模糊处理，保护隐私。
- [目标计数](../guides/object-counting.md)：学习使用 YOLO26 进行实时目标计数，掌握在实时视频流中准确计数目标的能力。
- [区域目标计数](../guides/region-counting.md)：使用 YOLO26 对特定区域内的目标进行计数，实现不同区域的精确检测。
- [目标裁剪](../guides/object-cropping.md)：掌握 YOLO26 的目标裁剪技术，从图像和视频中精确提取目标。
- [停车管理](../guides/parking-management.md)：使用 YOLO26 组织和管理停车场内车辆流动，优化空间利用率和用户体验。
- [队列管理](../guides/queue-management.md)：使用 YOLO26 实施高效的队列管理系统，最大限度地减少等待时间并提高生产力。
- [安全报警系统](../guides/security-alarm-system.md)：使用 YOLO26 创建安全报警系统，在检测到新目标时触发警报。可根据具体需求定制系统。
- [相似度搜索](../guides/similarity-search.md)：结合 [OpenAI CLIP](https://cookbook.openai.com/examples/custom_image_embedding_search) 嵌入和 [Meta FAISS](https://ai.meta.com/tools/faiss/)，实现智能图像检索，支持"手持包的人"或"运动中的车辆"等自然语言查询。
- [速度估算](../guides/speed-estimation.md)：利用 YOLO26 和目标跟踪技术估算目标速度，对于自动驾驶和交通监控等应用至关重要。
- [区域目标跟踪](../guides/trackzone.md)：学习如何使用 YOLO26 跟踪视频帧内特定区域的目标，实现精确高效的监控。
- [VisionEye 视角目标映射](../guides/vision-eye.md)：开发模拟人眼聚焦特定目标的系统，增强计算机识别和优先处理细节的能力。
- [健身监控](../guides/workouts-monitoring.md)：探索如何使用 YOLO26 监控健身训练，学习实时跟踪和分析各种健身动作。

### 解决方案参数

{% from "macros/solutions-args.md" import param_table %}
{{ param_table() }}

!!! note "跟踪参数"

     解决方案也支持 `track` 的部分参数，包括 `conf`、`line_width`、`tracker`、`model`、`show`、`verbose` 和 `classes` 等。

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

!!! note "可视化参数"

    你可以使用 `show_conf`、`show_labels` 和其他相关参数来自定义可视化效果。

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

### SolutionAnnotator 的使用

所有 Ultralytics 解决方案都使用独立的 [`SolutionAnnotator`](https://docs.ultralytics.com/reference/solutions/solutions#ultralytics.solutions.solutions.SolutionAnnotator) 类，该类继承自主 [`Annotator`](https://docs.ultralytics.com/reference/utils/plotting#ultralytics.utils.plotting.Annotator) 类，并具有以下方法：

| 方法                               | 返回类型   | 描述                                                       |
| ---------------------------------- | ---------- | ---------------------------------------------------------- |
| `draw_region()`                    | `None`     | 使用指定的点、颜色和线条粗细绘制区域。                         |
| `queue_counts_display()`           | `None`     | 在指定区域显示队列计数。                                      |
| `display_analytics()`              | `None`     | 显示停车场管理的整体统计信息。                                |
| `estimate_pose_angle()`            | `float`    | 计算目标姿态中三点之间的角度。                                |
| `draw_specific_points()`           | `None`     | 在图像上绘制特定的关键点。                                    |
| `plot_workout_information()`       | `None`     | 在图像上绘制带标签的文本框。                                  |
| `plot_angle_and_count_and_stage()` | `None`     | 可视化健身监控的角度、次数和阶段。                             |
| `plot_distance_and_line()`         | `None`     | 显示中心点之间的距离并用线条连接。                             |
| `display_objects_labels()`         | `None`     | 用目标类别标签标注边界框。                                    |
| `sweep_annotator()`                | `None`     | 可视化一条垂直扫描线和可选标签。                               |
| `visioneye()`                      | `None`     | 将目标中心点映射并连接到视觉"眼"点。                          |
| `adaptive_label()`                 | `None`     | 在边界框中心绘制圆形或矩形背景形状标签。                       |

### 使用 SolutionResults

除[相似度搜索](../guides/similarity-search.md)外，每个解决方案调用都会返回一个 `SolutionResults` 对象列表。

- 对于目标计数，结果包括 `in_count`、`out_count` 和 `classwise_count`。

!!! example "SolutionResults"

    ```python
    import cv2

    from ultralytics import solutions

    im0 = cv2.imread("path/to/img")

    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

    counter = solutions.ObjectCounter(
        show=True,  # 显示输出
        region=region_points,  # 传入区域点
        model="yolo26n.pt",  # 使用 OBB 模型进行目标计数时，使用 model="yolo26n-obb.pt"
        # classes=[0, 2],  # 使用 COCO 预训练模型计算特定类别，如人和车。
        # tracker="botsort.yaml"  # 选择跟踪器，如 "bytetrack.yaml"
    )
    results = counter(im0)
    print(results.in_count)  # 显示进入计数
    print(results.out_count)  # 显示离开计数
    print(results.classwise_count)  # 显示各类别计数
    ```

`SolutionResults` 对象具有以下属性：

| 属性                 | 类型               | 描述                                                           |
| -------------------- | ------------------ | -------------------------------------------------------------- |
| `plot_im`            | `np.ndarray`       | 带有可视化叠加层的图像，如计数、模糊效果或特定于解决方案的增强。    |
| `in_count`           | `int`              | 在视频流中检测到的进入定义区域的目标总数。                         |
| `out_count`          | `int`              | 在视频流中检测到的离开定义区域的目标总数。                         |
| `classwise_count`    | `Dict[str, int]`   | 按类别记录进出目标计数的字典，用于高级分析。                      |
| `queue_count`        | `int`              | 当前处于预定义队列或等候区域内的目标数量（适用于队列管理）。        |
| `workout_count`      | `int`              | 运动追踪过程中完成的健身动作总次数。                              |
| `workout_angle`      | `float`            | 健身过程中计算出的关节或姿态角度，用于动作评估。                   |
| `workout_stage`      | `str`              | 当前健身阶段或动作阶段（例如 'up'、'down'）。                    |
| `pixels_distance`    | `float`            | 两个目标或点（如边界框）之间的像素距离（适用于距离计算）。         |
| `available_slots`    | `int`              | 监控区域内未占用的车位数量（适用于停车管理）。                     |
| `filled_slots`       | `int`              | 监控区域内已占用的车位数量（适用于停车管理）。                     |
| `email_sent`         | `bool`             | 指示通知或警报邮件是否已成功发送（适用于安全报警）。               |
| `total_tracks`       | `int`              | 视频分析过程中观察到的唯一目标跟踪总数。                           |
| `region_counts`      | `Dict[str, int]`   | 用户定义区域或分区内的目标计数。                                  |
| `speed_dict`         | `Dict[str, float]` | 按跟踪目标记录的计算速度字典，用于速度分析。                       |
| `total_crop_objects` | `int`              | ObjectCropper 解决方案生成的目标裁剪图像总数。                    |
| `speed`              | `Dict[str, float]` | 包含跟踪和解决方案处理性能指标的字典。                             |

更多详情请参阅 [`SolutionResults` 类文档](https://docs.ultralytics.com/reference/solutions/solutions#ultralytics.solutions.solutions.SolutionAnnotator)。

### 通过命令行使用解决方案

!!! tip "命令信息"

    大多数解决方案可直接通过命令行界面使用，包括：

    `Count`、`Crop`、`Blur`、`Workout`、`Heatmap`、`Isegment`、`Visioneye`、`Speed`、`Queue`、`Analytics`、`Inference`、`Trackzone`

    **语法**

        yolo SOLUTIONS SOLUTION_NAME ARGS

    - **SOLUTIONS** 是必需的关键字。
    - **SOLUTION_NAME** 是以下之一：`['count', 'crop', 'blur', 'workout', 'heatmap', 'isegment', 'queue', 'speed', 'analytics', 'trackzone', 'inference', 'visioneye']`。
    - **ARGS**（可选）是自定义的 `arg=value` 对，如 `show_in=True`，用于覆盖默认设置。

```bash
yolo solutions count show=True # 目标计数

yolo solutions count source="path/to/video.mp4" # 指定视频文件路径
```

### 为我们的解决方案做贡献

我们欢迎社区的贡献！如果你已经掌握了 Ultralytics YOLO 的某个方面，而我们的解决方案尚未覆盖，欢迎分享你的专业知识。撰写指南是回馈社区的绝佳方式，也有助于让我们的文档更加全面和用户友好。

开始之前，请阅读我们的[贡献指南](../help/contributing.md)，了解如何提交 Pull Request (PR) 🛠️。期待你的贡献！

让我们携手让 Ultralytics YOLO 生态更加健壮和多功能 🙏！

## 常见问题

### 如何使用 Ultralytics YOLO 进行实时目标计数？

Ultralytics YOLO26 可利用其先进的目标检测能力进行实时目标计数。你可以参考我们详细的[目标计数](../guides/object-counting.md)指南，设置 YOLO26 进行实时视频流分析。只需安装 YOLO26，加载模型，并处理视频帧即可动态计数目标。

### 使用 Ultralytics YOLO 构建安全系统有哪些优势？

Ultralytics YOLO26 通过提供实时目标检测和警报机制来增强安全系统。借助 YOLO26，你可以创建一个安全报警系统，在监控区域检测到新目标时触发警报。了解如何用 YOLO26 搭建[安全报警系统](../guides/security-alarm-system.md)，实现稳健的安全监控。

### Ultralytics YOLO 如何改善队列管理系统？

Ultralytics YOLO26 可以通过准确计数和跟踪队列中的人员来显著改善队列管理系统，从而帮助减少等待时间并优化服务效率。请参考我们详细的[队列管理](../guides/queue-management.md)指南，了解如何使用 YOLO26 进行有效的队列监控和分析。

### Ultralytics YOLO 能用于健身监控吗？

是的，Ultralytics YOLO26 可以有效地用于健身监控，实时跟踪和分析各种健身动作，从而精确评估运动姿态和表现。请查看我们的[健身监控](../guides/workouts-monitoring.md)指南，了解如何使用 YOLO26 搭建 AI 驱动的健身监控系统。

### Ultralytics YOLO 如何帮助创建[数据可视化](https://www.ultralytics.com/glossary/data-visualization)热力图？

Ultralytics YOLO26 可以生成热力图来可视化指定区域的数据强度，突出显示高活动区域或感兴趣区域。此功能对于理解各种计算机视觉任务中的模式和趋势尤其有用。了解更多关于使用 YOLO26 创建和使用[热力图](../guides/heatmaps.md)的信息，实现全面的数据分析和可视化。
