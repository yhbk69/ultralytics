---
comments: true
description: 了解如何使用 Ultralytics YOLO26 管理和优化排队，在各种实际应用中减少等待时间并提高效率。
keywords: 排队管理, YOLO26, Ultralytics, 减少等待时间, 效率, 客户满意度, 零售, 机场, 医疗, 银行
---

# 使用 Ultralytics YOLO26 进行排队管理 🚀

## 什么是排队管理？

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-monitor-objects-in-queue-using-queue-management-solution.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开排队管理"></a>

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 进行排队管理，涉及组织和控制人群或车辆的排队，以减少等待时间并提高效率。其核心在于优化排队流程，从而在零售、银行、机场和医疗机构等各种场景中提升顾客满意度和系统性能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Gxr9SpYPLh0"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics YOLO 构建排队管理系统 | 零售、银行和人群用例 🚀
</p>

## 排队管理的优势

- **减少等待时间：** 排队管理系统高效组织队列，最大限度地减少顾客等待时间。随着顾客等待时间减少、有更多时间参与产品或服务，满意度水平也得以提升。
- **提高效率：** 实施排队管理使企业能够更有效地分配资源。通过分析排队数据并优化员工部署，企业可以简化运营、降低成本并提高整体生产力。
- **实时洞察：** 基于 YOLO26 的排队管理可提供队列长度和等待时间的即时数据，使管理人员能够快速做出明智的决策。
- **提升顾客体验：** 通过减少长时间等待带来的挫败感，企业可以显著提高顾客满意度和忠诚度。

## 实际应用

|                                                                                             物流                                                                                             |                                                                             零售                                                                             |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO26 在机场售票柜台进行排队管理](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/queue-management-airport-ticket-counter-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO26 在人群中监控排队](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/queue-monitoring-crowd-ultralytics-yolov8.avif) |
|                                                               使用 Ultralytics YOLO26 在机场售票柜台进行排队管理                                                                |                                                          使用 Ultralytics YOLO26 在人群中监控排队                                                          |

!!! example "使用 Ultralytics YOLO 进行排队管理"

    === "CLI"

        ```bash
        # 运行排队示例
        yolo solutions queue show=True

        # 传入源视频
        yolo solutions queue source="path/to/video.mp4"

        # 传入排队区域坐标
        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 视频输出
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 定义排队区域点位
        queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # 区域点位
        # queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]    # 多边形点位

        # 初始化排队管理器对象
        queuemanager = solutions.QueueManager(
            show=True,  # 显示输出
            model="yolo26n.pt",  # YOLO26 模型文件路径
            region=queue_region,  # 传入排队区域点位
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("视频帧为空或处理已完成。")
                break
            results = queuemanager(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `QueueManager` 参数

以下是 `QueueManager` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

`QueueManagement` 解决方案还支持部分 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还可以使用以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 实施策略

在使用 YOLO26 实施排队管理时，请考虑以下最佳实践：

1. **战略性摄像头部署：** 将摄像头放置在能够无遮挡地捕获整个排队区域的位置。
2. **定义合适的排队区域：** 根据实际空间布局仔细设置排队边界。
3. **调整检测置信度：** 根据光照条件和人群密度微调置信度阈值。
4. **与现有系统集成：** 将排队管理解决方案与数字标牌或员工通知系统连接，实现自动响应。

## 常见问题

### 如何使用 Ultralytics YOLO26 进行实时排队管理？

要使用 Ultralytics YOLO26 进行实时排队管理，可以按照以下步骤操作：

1. 使用 `YOLO("yolo26n.pt")` 加载 YOLO26 模型。
2. 使用 `cv2.VideoCapture` 捕获视频流。
3. 为排队管理定义感兴趣区域（ROI）。
4. 处理帧以检测对象并管理排队。

以下是一个最简示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
queue_region = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

queuemanager = solutions.QueueManager(
    model="yolo26n.pt",
    region=queue_region,
    line_width=3,
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if success:
        results = queuemanager(im0)

cap.release()
cv2.destroyAllWindows()
```

利用 [Ultralytics 平台](https://docs.ultralytics.com/platform) 可以简化这一流程，为部署和管理排队管理解决方案提供一个用户友好的平台。

### 使用 Ultralytics YOLO26 进行排队管理有哪些关键优势？

使用 Ultralytics YOLO26 进行排队管理具有以下多个优势：

- **大幅减少等待时间：** 高效组织队列，减少顾客等待时间，提升满意度。
- **提升效率：** 分析排队数据以优化员工部署和运营，从而降低成本。
- **实时告警：** 针对排队过长提供实时通知，便于快速干预。
- **可扩展性：** 易于在不同环境中扩展，如零售、机场和医疗。

详情请查阅我们的 [排队管理](https://docs.ultralytics.com/reference/solutions/queue_management) 解决方案。

### 为什么在排队管理中应选择 Ultralytics YOLO26 而非 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 或 Detectron2 等竞品？

在排队管理方面，Ultralytics YOLO26 相比 TensorFlow 和 Detectron2 具有多项优势：

- **实时性能：** YOLO26 以其实时检测能力著称，提供更快的处理速度。
- **易用性：** Ultralytics 通过 [Ultralytics 平台](https://docs.ultralytics.com/platform) 提供从训练到部署的全流程友好体验。
- **预训练模型：** 可访问一系列预训练模型，最大限度地缩短搭建时间。
- **社区支持：** 详尽的文档和活跃的社区支持使问题解决更加容易。

了解如何开始使用 [Ultralytics YOLO](https://docs.ultralytics.com/quickstart)。

### Ultralytics YOLO26 能否处理多种类型的排队场景，例如机场和零售环境？

可以，Ultralytics YOLO26 能够管理多种类型的排队场景，包括机场和零售环境。通过使用特定的区域和设置配置 QueueManager，YOLO26 可以适应不同的排队布局和密度。

机场示例：

```python
queue_region_airport = [(50, 600), (1200, 600), (1200, 550), (50, 550)]
queue_airport = solutions.QueueManager(
    model="yolo26n.pt",
    region=queue_region_airport,
    line_width=3,
)
```

有关多样化应用的更多信息，请参阅我们的 [实际应用](#实际应用) 章节。

### Ultralytics YOLO26 在排队管理中有哪些实际应用？

Ultralytics YOLO26 在排队管理中被广泛应用于多种实际场景：

- **零售：** 监控收银排队以减少等待时间，提升顾客满意度。
- **机场：** 管理售票柜台和安检点的排队，使旅客体验更加顺畅。
- **医疗：** 优化诊所和医院的患者流动。
- **银行：** 通过高效管理银行排队来提升客户服务。

查看我们的 [关于实际排队管理的博客](https://www.ultralytics.com/blog/a-look-at-real-time-queue-monitoring-enabled-by-computer-vision)，了解更多关于计算机视觉如何变革各行业排队监控的信息。
