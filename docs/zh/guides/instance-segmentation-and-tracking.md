---
comments: true
description: 使用 Ultralytics YOLO26 掌握实例分割与跟踪。学习精确对象识别与跟踪技术。
keywords: 实例分割, 跟踪, YOLO26, Ultralytics, 目标检测, 机器学习, 计算机视觉, python
---

# 使用 Ultralytics YOLO26 进行实例分割与跟踪 🚀

## 什么是实例分割？

[实例分割](https://www.ultralytics.com/glossary/instance-segmentation) 是一种计算机视觉任务，涉及在像素级别识别和勾勒图像中的各个对象。与仅按类别对像素进行分类的[语义分割](https://www.ultralytics.com/glossary/semantic-segmentation)不同，实例分割会为每个对象实例进行唯一标记并精确描绘，这对于需要详细空间理解的应用（如医学成像、自动驾驶和工业自动化）至关重要。

[Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 提供强大的实例分割功能，能够实现精确的对象边界检测，同时保持 YOLO 模型所著称的速度和效率。

Ultralytics 包中提供两种类型的实例分割跟踪：

- **按类别对象进行实例分割：** 每个类别对象被分配一种独特的颜色，以实现清晰的视觉区分。

- **按对象轨迹进行实例分割：** 每个轨迹由一种不同的颜色表示，便于跨视频帧轻松识别和跟踪。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/75G_S1Ngji8"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 使用 Ultralytics YOLO26 进行带对象跟踪的实例分割
</p>

## 示例

|                                                         实例分割                                                         |                                                                  实例分割 + 对象跟踪                                                                   |
| :-----------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics 实例分割](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-instance-segmentation.avif) | ![Ultralytics 带对象跟踪的实例分割](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-instance-segmentation-object-tracking.avif) |
|                                                 Ultralytics 实例分割 😍                                                  |                                                         带对象跟踪的 Ultralytics 实例分割 🔥                                                          |

!!! example "使用 Ultralytics YOLO 进行实例分割"

    === "CLI"

        ```bash
        # 使用 Ultralytics YOLO26 进行实例分割
        yolo solutions isegment show=True

        # 传入视频源
        yolo solutions isegment source="path/to/video.mp4"

        # 监控特定类别
        yolo solutions isegment classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件时出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("isegment_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化实例分割对象
        isegment = solutions.InstanceSegmentation(
            show=True,  # 显示输出
            model="yolo26n-seg.pt",  # model="yolo26n-seg.pt" 用于使用 YOLO26 进行对象分割。
            # classes=[0, 2],  # 分割特定类别，例如使用预训练模型分割人和车。
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("视频帧为空或视频处理已完成。")
                break

            results = isegment(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧。

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `InstanceSegmentation` 参数

以下是 `InstanceSegmentation` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

您还可以在 `InstanceSegmentation` 解决方案中利用 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还提供以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 实例分割的应用

YOLO26 的实例分割在多个行业中有许多实际应用：

### 废物管理与回收

YOLO26 可用于[废物管理设施](https://www.ultralytics.com/blog/simplifying-e-waste-management-with-ai-innovations)中，以识别和分类不同类型的材料。该模型可以高精度地分割塑料废物、纸板、金属和其他可回收物，使自动化分拣系统能够更高效地处理废物。考虑到全球产生的 70 亿吨塑料废物中只有约 10% 被回收，这一点尤其有价值。

### 自动驾驶汽车

在[自动驾驶汽车](https://www.ultralytics.com/solutions/ai-in-automotive)中，实例分割有助于在像素级别识别和跟踪行人、车辆、交通标志和其他道路元素。这种对环境的精确理解对于导航和安全决策至关重要。YOLO26 的实时性能使其成为这些时间敏感型应用的理想选择。

### 医学成像

实例分割可以识别和勾勒医学扫描图像中的肿瘤、器官或细胞结构。YOLO26 精确描绘对象边界的能力使其在[医学诊断](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency)和治疗规划中具有价值。

### 建筑工地监控

在建筑工地，实例分割可以跟踪重型机械、工人和材料。这有助于通过监控设备位置和检测工人何时进入危险区域来确保安全，同时还能优化工作流程和资源分配。

## 注意

如有任何疑问，请随时在 [Ultralytics 问题部分](https://github.com/ultralytics/ultralytics/issues/new/choose)或下面提到的讨论区发布您的问题。

## 常见问题

### 如何使用 Ultralytics YOLO26 进行实例分割？

要使用 Ultralytics YOLO26 进行实例分割，请使用 YOLO26 的分割版本初始化 YOLO 模型，并通过它处理视频帧。以下是一个简化的代码示例：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "读取视频文件时出错"

# 视频写入器
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 初始化 InstanceSegmentation
isegment = solutions.InstanceSegmentation(
    show=True,  # 显示输出
    model="yolo26n-seg.pt",  # model="yolo26n-seg.pt" 用于使用 YOLO26 进行对象分割。
)

# 处理视频
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("视频帧为空或处理完成。")
        break
    results = isegment(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

有关实例分割的更多信息，请参阅 [Ultralytics YOLO26 指南](https://docs.ultralytics.com/tasks/segment)。

### Ultralytics YOLO26 中实例分割与对象跟踪有什么区别？

实例分割识别并勾勒图像中的各个对象，为每个对象提供唯一的标签和掩码。对象跟踪通过为跨视频帧的对象分配一致的 ID 来扩展此功能，便于随时间持续跟踪同一对象。当两者结合使用时，就像在 YOLO26 的实现中一样，您将获得强大的功能，用于分析视频中对象的移动和行为，同时保持精确的边界信息。

### 为什么我应该使用 Ultralytics YOLO26 进行实例分割和跟踪，而不是其他模型如 Mask R-CNN 或 Faster R-CNN？

与 Mask R-CNN 或 Faster R-CNN 等其他模型相比，Ultralytics YOLO26 提供实时性能、卓越的[准确性](https://www.ultralytics.com/glossary/accuracy)和易用性。YOLO26 在单次传递中处理图像（单阶段检测），使其速度显著更快，同时保持高精度。它还提供与 [Ultralytics 平台](https://platform.ultralytics.com) 的无缝集成，允许用户高效地管理模型、数据集和训练管道。对于需要速度和准确性的应用，YOLO26 提供了最佳平衡。

### Ultralytics 是否提供适用于训练 YOLO26 模型进行实例分割和跟踪的数据集？

是的，Ultralytics 提供了几个适用于训练 YOLO26 模型进行实例分割的数据集，包括 [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco)、[COCO8-Seg](https://docs.ultralytics.com/datasets/segment/coco8-seg)（用于快速测试的较小子集）、[Package-Seg](https://docs.ultralytics.com/datasets/segment/package-seg) 和 [Crack-Seg](https://docs.ultralytics.com/datasets/segment/crack-seg)。这些数据集带有实例分割任务所需的像素级注释。对于更专业的应用，您还可以按照 Ultralytics 格式创建自定义数据集。完整的数据集信息和使用说明可在 [Ultralytics 数据集文档](https://docs.ultralytics.com/datasets) 中找到。