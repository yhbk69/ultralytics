---
comments: true
description: 了解 Ultralytics YOLO26 驱动的 VisionEye 对象映射与追踪。轻松模拟人眼精度，追踪对象并计算距离。
keywords: VisionEye, YOLO26, Ultralytics, 对象映射, 对象追踪, 距离计算, 计算机视觉, AI, 机器学习, Python, 教程
---

# 使用 Ultralytics YOLO26 的 VisionEye 视角对象映射 🚀

## 什么是 VisionEye 对象映射？

[Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) VisionEye 赋予计算机识别并定位对象的能力，模拟人眼观察的[精度](https://www.ultralytics.com/glossary/precision)。该功能使计算机能够辨别并聚焦于特定对象，就像人眼从特定视角观察细节一样。

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/visioneye-object-mapping-with-tracking.avif" alt="VisionEye 对象映射与 YOLO 追踪">
</p>

!!! example "使用 Ultralytics YOLO 进行 VisionEye 映射"

    === "CLI"

        ```bash
        # 使用 visioneye 监控对象位置
        yolo solutions visioneye show=True

        # 传入源视频
        yolo solutions visioneye source="path/to/video.mp4"

        # 监控特定类别
        yolo solutions visioneye classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化 vision eye 对象
        visioneye = solutions.VisionEye(
            show=True,  # 显示输出
            model="yolo26n.pt",  # 使用 Ultralytics 支持的任何模型，例如 YOLOv10
            classes=[0, 2],  # 为特定类别生成 visioneye 视图
            vision_point=(50, 50),  # VisionEye 观察对象并绘制轨迹的点
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = visioneye(im0)

            print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入视频文件

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

        `vision_point` 元组表示观察者在像素坐标中的位置。请根据相机视角进行调整，使渲染的光线正确展示对象与所选视角的关系。

### `VisionEye` 参数

以下是 `VisionEye` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "vision_point"]) }}

您也可以在 `VisionEye` 方案中使用各种 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## VisionEye 工作原理

VisionEye 通过在画面中建立一个固定的视点，并从该视点向检测到的对象绘制连线来工作。这模拟了人类视觉从单一视角聚焦多个对象的方式。该方案使用[对象追踪](https://docs.ultralytics.com/modes/track)来跨帧保持对象的一致标识，从而直观展示观察者（视点）与场景中对象之间的空间关系。

VisionEye 类中的 `process` 方法执行以下关键操作：

1. 从输入图像中提取追踪信息（边界框、类别和掩码）
2. 创建标注器用于绘制边界框和标签
3. 对于每个检测到的对象，绘制框标签并从视点创建一条视觉连线
4. 返回带有追踪统计信息的标注图像

这种方法特别适用于需要空间感知和对象关系可视化的应用，例如监控系统、自主导航和交互装置。

## VisionEye 的应用

VisionEye 对象映射在各行各业有广泛的实用场景：

- **安防监控**：从固定摄像头位置监控多个感兴趣的目标对象
- **零售分析**：追踪顾客在店铺陈列区域的活动轨迹
- **体育分析**：从教练视角分析球员的站位和跑动
- **自动驾驶**：可视化车辆如何"看到"并优先处理其环境中的对象
- **人机交互**：创建能响应空间关系的更直观的交互界面

将 VisionEye 与其他 Ultralytics 方案（如[距离计算](https://docs.ultralytics.com/guides/distance-calculation)或[速度估计](https://docs.ultralytics.com/guides/speed-estimation)）相结合，您可以构建不仅追踪对象、还能理解其空间关系与行为的综合系统。

## 注意事项

如有任何疑问，欢迎在 [Ultralytics Issue 板块](https://github.com/ultralytics/ultralytics/issues/new/choose)或下方讨论区中提问。

## 常见问题

### 如何开始使用 Ultralytics YOLO26 的 VisionEye 对象映射？

要开始使用 Ultralytics YOLO26 的 VisionEye 对象映射，首先需要通过 pip 安装 Ultralytics YOLO 包。然后，您可以使用文档中提供的示例代码来设置带有 VisionEye 的[对象检测](https://www.ultralytics.com/glossary/object-detection)。以下是一个简单入门示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

# 视频写入器
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("vision-eye-mapping.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 初始化 vision eye 对象
visioneye = solutions.VisionEye(
    show=True,  # 显示输出
    model="yolo26n.pt",  # 使用 Ultralytics 支持的任何模型，例如 YOLOv10
    classes=[0, 2],  # 为特定类别生成 visioneye 视图
)

# 处理视频
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = visioneye(im0)

    print(results)  # 访问输出

    video_writer.write(results.plot_im)  # 写入视频文件

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # 销毁所有打开的窗口
```

### 为什么应该使用 Ultralytics YOLO26 进行对象映射和追踪？

Ultralytics YOLO26 以其速度、[准确度](https://www.ultralytics.com/glossary/accuracy)和易于集成而闻名，是对象映射和追踪的首选。主要优势包括：

1. **一流的性能**：在实时对象检测中提供高准确率。
2. **灵活性**：支持检测、追踪和距离计算等多种任务。
3. **社区与支持**：丰富的文档和活跃的 GitHub 社区，便于问题排查和功能增强。
4. **易用性**：直观的 API 简化了复杂任务，支持快速部署和迭代。

有关应用和优势的更多信息，请参阅 [Ultralytics YOLO26 文档](https://docs.ultralytics.com/models/yolo26)。

### 如何将 VisionEye 与 Comet 或 ClearML 等[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)工具集成？

Ultralytics YOLO26 可以与 Comet 和 ClearML 等各种机器学习工具无缝集成，增强实验追踪、协作和可复现性。请参阅详细指南：[如何将 YOLOv5 与 Comet 结合使用](https://www.ultralytics.com/blog/how-to-use-yolov5-with-comet)以及[将 YOLO26 与 ClearML 集成](https://docs.ultralytics.com/integrations/clearml)。

如需更多探索和集成示例，请查看 [Ultralytics 集成指南](https://docs.ultralytics.com/integrations)。