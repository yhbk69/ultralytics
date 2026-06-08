---
comments: true
description: 使用 Ultralytics YOLO26 实时监控锻炼，优化您的健身计划。追踪并改善您的运动姿态和表现。
keywords: 锻炼监控, Ultralytics YOLO26, 姿态估计, 健身追踪, 运动评估, 实时反馈, 运动姿态, 表现指标
---

# 使用 Ultralytics YOLO26 进行锻炼监控

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-monitor-workouts-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开锻炼监控"></a>

通过 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 的姿态估计来监控锻炼，能够实时精确追踪关键身体标志点和关节，从而增强运动评估能力。该技术可即时反馈运动姿态、追踪训练计划并衡量表现指标，帮助用户和教练优化训练效果。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Ck7DW96dNok"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO 监控锻炼动作 | 深蹲、腿屈伸、俯卧撑等
</p>

## 锻炼监控的优势

- **优化表现：** 基于监控数据定制训练方案，以获得更好的效果。
- **达成目标：** 追踪并调整健身目标，实现可量化的进步。
- **个性化：** 根据个人数据定制训练计划，提升有效性。
- **健康意识：** 早期发现预示健康问题或过度训练的模式。
- **明智决策：** 以数据为驱动，调整训练计划并设定切实可行的目标。

## 实际应用

|                                                      锻炼监控                                                       |                                                      锻炼监控                                                       |
| :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| ![YOLO 俯卧撑计数与姿态估计](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/pushups-counting.avif) | ![YOLO 引体向上计数与姿态估计](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/pullups-counting.avif) |
|                                                        俯卧撑计数                                                        |                                                        引体向上计数                                                        |

!!! example "使用 Ultralytics YOLO 进行锻炼监控"

    === "命令行"

        ```bash
        # 运行锻炼示例
        yolo solutions workout show=True

        # 传入源视频
        yolo solutions workout source="path/to/video.mp4"

        # 使用关键点进行俯卧撑检测
        yolo solutions workout kpts="[6, 8, 10]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("workouts_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化 AIGym
        gym = solutions.AIGym(
            show=True,  # 显示画面
            kpts=[6, 8, 10],  # 用于监控特定动作的关键点，默认为俯卧撑
            model="yolo26n-pose.pt",  # YOLO26 姿态估计模型文件路径
            # line_width=2,  # 调整边界框和文本显示的线宽
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = gym(im0)

            # print(results)  # 访问输出结果

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### 关键点映射

![YOLO 姿态估计关键点顺序图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/keypoints-order-ultralytics-yolov8-pose.avif)

### `AIGym` 参数

以下是 `AIGym` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "up_angle", "down_angle", "kpts"]) }}

`AIGym` 解决方案还支持一系列目标追踪参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还可以应用以下可视化设置：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 常见问题

### 如何使用 Ultralytics YOLO26 监控我的锻炼？

要使用 Ultralytics YOLO26 监控锻炼，您可以利用[姿态估计功能](https://docs.ultralytics.com/tasks/pose)实时追踪和分析关键身体标志点和关节。这让您能够即时获得运动姿态反馈、计算重复次数并衡量表现指标。您可以从以下俯卧撑、引体向上或腹肌训练的示例代码开始：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = gym(im0)

cv2.destroyAllWindows()
```

如需更多自定义和设置，请参考文档中的 [AIGym](#aigym-arguments) 章节。

### 使用 Ultralytics YOLO26 进行锻炼监控有哪些优势？

使用 Ultralytics YOLO26 进行锻炼监控具有以下几个关键优势：

- **优化表现：** 基于监控数据定制训练，获得更好效果。
- **达成目标：** 轻松追踪并调整健身目标，实现可量化的进步。
- **个性化：** 根据个人数据获取定制训练计划，达到最佳效果。
- **健康意识：** 早期发现预示潜在健康问题或过度训练的模式。
- **明智决策：** 以数据为驱动调整训练计划并设定切实可行的目标。

您可以观看 [YouTube 视频演示](https://www.youtube.com/watch?v=LGGxqLZtvuw) 来了解这些优势的实际效果。

### Ultralytics YOLO26 在检测和追踪运动动作方面有多准确？

Ultralytics YOLO26 凭借其先进的[姿态估计](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-pose-estimation)能力，在检测和追踪运动动作方面具有很高的准确度。它可以精确追踪关键身体标志点和关节，提供运动姿态和表现指标的实时反馈。该模型的预训练权重和稳健架构确保了高[精度](https://www.ultralytics.com/glossary/precision)和可靠性。有关实际示例，请查看文档中的[实际应用](#real-world-applications)章节，其中展示了俯卧撑和引体向上的计数效果。

### 我可以将 Ultralytics YOLO26 用于自定义训练计划吗？

可以，Ultralytics YOLO26 可以适配自定义训练计划。`AIGym` 类支持不同的动作类型，如 `pushup`（俯卧撑）、`pullup`（引体向上）和 `abworkout`（腹肌训练）。您可以指定关键点和角度来检测特定动作。以下是示例配置：

```python
from ultralytics import solutions

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],  # 俯卧撑用——可自定义为其他动作
)
```

有关设置参数的更多细节，请参考 [Arguments `AIGym`](#aigym-arguments) 章节。这种灵活性让您可以监控各种运动，并根据您的[健身目标](https://www.ultralytics.com/blog/ai-in-our-day-to-day-health-and-fitness)定制训练计划。

### 如何使用 Ultralytics YOLO26 保存锻炼监控输出？

要保存锻炼监控输出，您可以修改代码，加入视频写入器来保存处理后的帧。示例如下：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("workouts.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = gym(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

此设置将监控视频写入输出文件，让您稍后回顾锻炼表现，或将其分享给教练以获取更多反馈。
