---
comments: true
description: 了解如何使用 Ultralytics YOLO26 在交通控制、自主导航和监控等应用中估计物体速度。
keywords: Ultralytics YOLO26, 速度估计, 物体跟踪, 计算机视觉, 交通控制, 自主导航, 监控, 安全
---

# 使用 Ultralytics YOLO26 进行速度估计 🚀

## 什么是速度估计？

[速度估计](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)是在给定上下文中计算物体运动速率的过程，常用于[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)应用。使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/)，你可以结合[物体跟踪](../modes/track.md)以及距离和时间数据来计算物体速度，这对于交通监控和安防等任务至关重要。速度估计的准确性直接影响各种应用的效率和可靠性，使其成为推动智能系统和实时决策过程进步的关键组成部分。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rCggzXRRSRo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO26 进行速度估计
</p>

!!! tip "查看我们的博客"

    要深入了解速度估计，请查看我们的博客文章：[Ultralytics YOLO 在计算机视觉项目中的速度估计](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)

## 速度估计的优势

- **高效的交通控制：** 准确的速度估计有助于管理交通流量、提高安全性并减少道路拥堵。
- **精准的自主导航：** 在[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)等自主系统中，可靠的速度估计确保车辆安全、准确地导航。
- **增强的监控安防：** 监控分析中的速度估计有助于识别异常行为或潜在威胁，提高安防措施的有效性。

## 实际应用

|                                                                            交通                                                                             |                                                                              交通                                                                              |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO26 在道路上的速度估计](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/speed-estimation-on-road-using-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO26 在桥梁上的速度估计](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/speed-estimation-on-bridge-using-ultralytics-yolov8.avif) |
|                                                           使用 Ultralytics YOLO26 在道路上的速度估计                                                           |                                                            使用 Ultralytics YOLO26 在桥梁上的速度估计                                                            |

???+ warning "速度仅为估计值"

    速度只是一个估计值，可能并非完全准确。此外，估计值可能因相机规格和相关因素而有所变化。

!!! example "使用 Ultralytics YOLO 进行速度估计"

    === "CLI"

        ```bash
        # 运行速度示例
        yolo solutions speed show=True

        # 传入源视频
        yolo solutions speed source="path/to/video.mp4"

        # 根据相机配置调整每像素米数值
        yolo solutions speed meter_per_pixel=0.05
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化速度估计对象
        speedestimator = solutions.SpeedEstimator(
            show=True,  # 显示输出
            model="yolo26n.pt",  # YOLO26 模型文件路径
            fps=fps,  # 基于每秒帧数调整速度
            # max_speed=120,  # 将速度上限设为最大值（km/h）以避免异常值
            # max_hist=5,  # 计算速度前物体被跟踪的最小帧数
            # meter_per_pixel=0.05,  # 高度依赖于相机配置
            # classes=[0, 2],  # 估计特定类别的速度
            # line_width=2,  # 调整边界框的线宽
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = speedestimator(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 关闭所有打开的窗口
        ```

### `SpeedEstimator` 参数

以下是 `SpeedEstimator` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "fps", "max_hist", "meter_per_pixel", "max_speed"]) }}

`SpeedEstimator` 解决方案允许使用 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化选项：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题

### 如何使用 Ultralytics YOLO26 估计物体速度？

使用 Ultralytics YOLO26 估计物体速度需要结合[物体检测](https://www.ultralytics.com/glossary/object-detection)和跟踪技术。首先，你需要使用 YOLO26 模型检测每一帧中的物体。然后，跨帧跟踪这些物体以计算它们随时间的移动。最后，利用物体在帧之间移动的距离和帧率来估计其速度。

**示例**：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 初始化 SpeedEstimator
speedestimator = solutions.SpeedEstimator(
    model="yolo26n.pt",
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = speedestimator(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

更多详情请参考我们的[官方博客文章](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)。

### 在交通管理中使用 Ultralytics YOLO26 进行速度估计有哪些优势？

使用 Ultralytics YOLO26 进行速度估计在交通管理中具有显著优势：

- **增强安全性**：准确估计车辆速度以检测超速行为，提高道路安全。
- **实时监控**：利用 YOLO26 的实时物体检测能力有效监控交通流量和拥堵情况。
- **可扩展性**：将模型部署到各种硬件配置上，从[边缘设备](https://docs.ultralytics.com/guides/nvidia-jetson)到服务器，为大规模实施提供灵活且可扩展的解决方案。

更多应用请参见[速度估计的优势](#速度估计的优势)。

### YOLO26 能否与 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 或 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 等其他 AI 框架集成？

是的，YOLO26 可以与 TensorFlow 和 PyTorch 等其他 AI 框架集成。Ultralytics 支持将 YOLO26 模型导出为多种格式，如 [ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md) 和 [CoreML](../integrations/coreml.md)，确保与其他 ML 框架的顺畅互操作性。

将 YOLO26 模型导出为 ONNX 格式：

```bash
yolo export model=yolo26n.pt format=onnx
```

更多关于导出模型的信息，请参阅我们的[导出指南](../modes/export.md)。

### 使用 Ultralytics YOLO26 进行速度估计的准确度如何？

使用 Ultralytics YOLO26 进行速度估计的[准确度](https://www.ultralytics.com/glossary/accuracy)取决于多个因素，包括物体跟踪的质量、视频的分辨率和帧率，以及环境变量。虽然速度估计器提供了可靠的估计值，但由于帧处理速度和物体遮挡的差异，可能无法达到 100% 的准确率。

**注意**：始终考虑误差范围，并在可能的情况下使用真实数据验证估计值。

更多提高准确度的技巧，请查看 [`SpeedEstimator` 参数部分](#speedestimator-参数)。
