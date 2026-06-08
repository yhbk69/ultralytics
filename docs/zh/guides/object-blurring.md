---
comments: true
description: 了解如何使用 Ultralytics YOLO26 进行实时目标模糊处理，以增强图像和视频中的隐私保护和焦点控制。
keywords: YOLO26, 目标模糊, 实时处理, 隐私保护, 图像处理, 视频编辑, Ultralytics
---

# 使用 Ultralytics YOLO26 进行目标模糊处理 🚀

## 什么是目标模糊处理？

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 进行目标模糊处理，是指对图像或视频中检测到的特定目标应用模糊效果。这可以利用 YOLO26 模型的能力来识别和操控给定场景中的目标。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/m-Lc5MXbydg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics 平台训练人脸检测模型及模糊人脸 | Ultralytics YOLO26 🚀
</p>

## 目标模糊处理的优势

- **隐私保护**：目标模糊处理是一种有效的隐私保护工具，能够隐藏图像或视频中敏感或可识别个人身份的信息。
- **选择性聚焦**：YOLO26 支持选择性模糊，用户可以针对特定目标进行处理，在隐私保护和保留相关视觉信息之间取得平衡。
- **实时处理**：YOLO26 的高效性能使得目标模糊处理可以实时进行，适用于需要在动态环境中即时增强隐私的应用场景。
- **法规合规**：通过对视觉内容中的可识别信息进行匿名化处理，帮助组织遵守 GDPR 等数据保护法规。
- **内容审核**：可用于模糊媒体平台中不当或敏感内容，同时保留整体上下文。

!!! example "使用 Ultralytics YOLO 进行目标模糊处理"

    === "CLI"

        ```bash
        # 模糊目标
        yolo solutions blur show=True

        # 传入源视频
        yolo solutions blur source="path/to/video.mp4"

        # 模糊特定类别
        yolo solutions blur classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化目标模糊器
        blurrer = solutions.ObjectBlurrer(
            show=True,  # 显示输出
            model="yolo26n.pt",  # 用于目标模糊的模型，例如 yolo26m.pt
            # line_width=2,  # 边界框宽度
            # classes=[0, 2],  # 模糊特定类别，例如使用 COCO 预训练模型的人和汽车
            # blur_ratio=0.5,  # 调整模糊强度百分比，取值范围 0.1 - 1.0
        )

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = blurrer(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

### `ObjectBlurrer` 参数

以下是 `ObjectBlurrer` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "line_width", "blur_ratio"]) }}

`ObjectBlurrer` 解决方案还支持一系列 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还可以使用以下可视化参数：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 实际应用场景

### 监控中的隐私保护

[安防摄像头](https://www.ultralytics.com/blog/the-cutting-edge-world-of-ai-security-cameras)和监控系统可以使用 YOLO26 自动模糊人脸、车牌或其他识别信息，同时仍然捕获重要活动。这有助于在公共空间中维护安全的同时尊重隐私权。

### 医疗数据匿名化

在[医学影像](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency)中，患者信息经常出现在扫描件或照片中。YOLO26 可以检测并模糊这些信息，以便在出于研究或教育目的共享医疗数据时遵守 HIPAA 等法规。

### 文档脱敏

当共享包含敏感信息的文档时，YOLO26 可以自动检测并模糊特定元素，如签名、账号或个人详细信息，从而简化脱敏流程，同时保持文档完整性。

### 媒体与内容创作

内容创作者可以使用 YOLO26 模糊视频和图像中的品牌标志、受版权保护的材料或不当内容，有助于避免法律问题，同时保持整体内容质量。

## 常见问题

### 什么是使用 Ultralytics YOLO26 进行目标模糊处理？

使用 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 进行目标模糊处理是指自动检测并对图像或视频中的特定目标应用模糊效果。该技术通过隐藏敏感信息来增强隐私保护，同时保留相关视觉数据。YOLO26 的实时处理能力使其适用于需要即时隐私保护和选择性聚焦调整的应用场景。

### 如何使用 YOLO26 实现实时目标模糊处理？

要使用 YOLO26 实现实时目标模糊处理，请参考以下 Python 示例。这涉及使用 YOLO26 进行[目标检测](https://www.ultralytics.com/glossary/object-detection)以及使用 OpenCV 应用模糊效果。以下是简化版本：

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 视频写入器
video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 初始化 ObjectBlurrer
blurrer = solutions.ObjectBlurrer(
    show=True,  # 显示输出
    model="yolo26n.pt",  # model="yolo26n-obb.pt" 用于使用 YOLO26 OBB 模型进行目标模糊
    blur_ratio=0.5,  # 设置模糊百分比，例如 0.7 表示对检测到的目标应用 70% 模糊
    # line_width=2,  # 边界框宽度
    # classes=[0, 2],  # 统计特定类别，例如使用 COCO 预训练模型的人和汽车
)

# 处理视频
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = blurrer(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

### 使用 Ultralytics YOLO26 进行目标模糊处理有哪些优势？

Ultralytics YOLO26 在目标模糊处理方面具有以下优势：

- **隐私保护**：有效模糊敏感或可识别信息。
- **选择性聚焦**：针对特定目标进行模糊处理，保留必要的视觉内容。
- **实时处理**：在动态环境中高效执行目标模糊处理，适用于即时隐私增强。
- **可自定义强度**：调整模糊比例以平衡隐私需求和视觉上下文。
- **按类别模糊**：仅选择性地模糊特定类型的目标，其余目标保持可见。

更多详细信息请查看[目标模糊处理的优势章节](#advantages-of-object-blurring)。

### 我可以使用 Ultralytics YOLO26 模糊视频中的人脸以保护隐私吗？

可以。Ultralytics YOLO26 可以配置为检测并模糊视频中的人脸以保护隐私。通过训练或使用预训练模型专门识别人脸，检测结果可以与 [OpenCV](https://www.ultralytics.com/glossary/opencv) 结合处理以应用模糊效果。请参考我们的 [YOLO26 目标检测指南](https://docs.ultralytics.com/models/yolo26)并根据需要修改代码以针对人脸检测。

### YOLO26 在目标模糊处理方面与 Faster R-CNN 等其他目标检测模型相比如何？

Ultralytics YOLO26 在速度方面通常优于 Faster R-CNN 等模型，使其更适合实时应用。虽然两种模型都能提供准确的检测，但 YOLO26 的架构针对快速推理进行了优化，这对于实时目标模糊处理等任务至关重要。了解更多技术差异和性能指标，请参阅我们的 [YOLO26 文档](https://docs.ultralytics.com/models/yolo26)。
