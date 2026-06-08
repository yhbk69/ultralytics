---
comments: true
description: 学习如何使用 Ultralytics YOLO26 计算物体之间的距离，实现精确的空间定位与场景理解。
keywords: Ultralytics, YOLO26, 距离计算, 计算机视觉, 目标跟踪, 空间定位
---

# 使用 Ultralytics YOLO26 进行距离计算

## 什么是距离计算？

在指定空间内测量两个物体之间的间距称为距离计算。在 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 中，使用用户标记的[边界框](https://www.ultralytics.com/glossary/bounding-box)的质心来计算距离。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Oe0vmsvnY74"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics YOLO 以像素为单位估算检测到的物体之间的距离 🚀
</p>

## 效果展示

|                                         使用 Ultralytics YOLO26 进行距离计算                                          |
| :----------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO26 距离计算](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/distance-calculation.avif) |

## 距离计算的优势

- **定位[精度](https://www.ultralytics.com/glossary/precision)：** 增强[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务中的精确空间定位能力。
- **尺寸估算：** 能够估算物体尺寸，从而更好地理解上下文。
- **场景理解：** 提升三维场景的理解能力，为[自动驾驶](https://www.ultralytics.com/glossary/autonomous-vehicles)和监控系统等应用提供更好的决策支持。
- **碰撞避免：** 通过监控移动物体之间的距离，使系统能够检测潜在的碰撞风险。
- **空间分析：** 便于分析监控环境中物体的关系与交互。

???+ tip "距离计算"

    - 用鼠标左键点击任意两个边界框即可计算距离。
    - 使用鼠标右键删除所有已绘制的点。
    - 在画面中任意位置左键点击可添加新的点。

???+ warning "距离仅为估算值"

    距离仅为估算值，可能不完全准确，因为它是基于二维数据计算的，
    缺少深度信息。

!!! example "使用 Ultralytics YOLO 进行距离计算"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化距离计算对象
        distancecalculator = solutions.DistanceCalculation(
            model="yolo26n.pt",  # YOLO26 模型文件路径
            show=True,  # 显示输出
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或处理已完成。")
                break

            results = distancecalculator(im0)

            print(results)  # 访问输出结果

            video_writer.write(results.plot_im)  # 写入处理后的视频帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 关闭所有打开的窗口
        ```

### `DistanceCalculation()` 参数

下表列出了 `DistanceCalculation` 的参数：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model"]) }}

你还可以在 `DistanceCalculation` 解决方案中使用各种 `track` 参数。

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还可以使用以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 实现细节

`DistanceCalculation` 类通过跨视频帧跟踪物体，并计算所选边界框质心之间的欧几里得距离来工作。当你点击两个物体时，该解决方案会：

1. 提取所选边界框的质心（中心点）
2. 以像素为单位计算这些质心之间的欧几里得距离
3. 在画面上显示距离，并用连线连接两个物体

该实现使用 `mouse_event_for_distance` 方法来处理鼠标交互，允许用户根据需要选择物体和清除选择。`process` 方法负责逐帧处理、跟踪物体和计算距离。

## 应用场景

使用 YOLO26 进行距离计算具有众多实际应用：

- **零售分析：** 测量顾客与产品的接近程度，分析商店布局的有效性
- **工业安全：** 监控工人与机器之间的安全距离
- **交通管理：** 分析车辆间距并检测追尾风险
- **体育分析：** 计算球员之间、球员与球之间以及与关键场地区域之间的距离
- **医疗健康：** 确保候诊区域的适当间距，监控患者活动
- **机器人技术：** 使机器人能够与障碍物和人员保持适当距离

## FAQ

### 如何使用 Ultralytics YOLO26 计算物体之间的距离？

要使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 计算物体之间的距离，你需要识别检测到的物体的边界框质心。此过程包括从 Ultralytics 的 `solutions` 模块初始化 `DistanceCalculation` 类，并使用模型的跟踪输出来计算距离。

### 使用 Ultralytics YOLO26 进行距离计算有哪些优势？

使用 Ultralytics YOLO26 进行距离计算具有以下优势：

- **定位精度：** 为物体提供精确的空间定位。
- **尺寸估算：** 有助于估算物理尺寸，从而更好地理解上下文。
- **场景理解：** 增强三维场景理解能力，助力自动驾驶和监控等应用做出更好的决策。
- **实时处理：** 即时执行计算，适用于实时视频分析。
- **集成能力：** 可与 YOLO26 的其他解决方案无缝协作，如[目标跟踪](../modes/track.md)和[速度估算](speed-estimation.md)。

### 能否使用 Ultralytics YOLO26 在实时视频流中进行距离计算？

可以，你可以使用 Ultralytics YOLO26 在实时视频流中进行距离计算。该过程包括使用 [OpenCV](https://www.ultralytics.com/glossary/opencv) 捕获视频帧，运行 YOLO26 [目标检测](https://www.ultralytics.com/glossary/object-detection)，并使用 `DistanceCalculation` 类计算连续帧中物体之间的距离。详细实现请参考[视频流示例](#使用-ultralytics-yolo26-进行距离计算)。

### 如何删除使用 Ultralytics YOLO26 进行距离计算时绘制的点？

要删除使用 Ultralytics YOLO26 进行距离计算时绘制的点，你可以使用鼠标右键点击。此操作将清除你绘制的所有点。更多详细信息请参阅[距离计算示例](#使用-ultralytics-yolo26-进行距离计算)下方的注释部分。

### 初始化 Ultralytics YOLO26 中 DistanceCalculation 类的关键参数有哪些？

初始化 Ultralytics YOLO26 中 `DistanceCalculation` 类的关键参数包括：

- `model`：YOLO26 模型文件路径。
- `tracker`：要使用的跟踪算法（默认为 'botsort.yaml'）。
- `conf`：检测置信度阈值。
- `show`：是否显示输出的标志。

完整的参数列表和默认值请参考 [DistanceCalculation 的参数](#distancecalculation-参数)。