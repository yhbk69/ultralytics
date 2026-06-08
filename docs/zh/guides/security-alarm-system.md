---
comments: true
description: 使用 Ultralytics YOLO26 的实时目标检测来增强您的安防能力。减少误报并实现与现有系统的无缝集成。
keywords: YOLO26, 安防报警系统, 实时目标检测, Ultralytics, 计算机视觉, 集成, 误报
---

# 基于 Ultralytics YOLO26 的安防报警系统项目

<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/security-alarm-system-ultralytics-yolov8.avif" alt="基于 AI 目标检测的安防报警系统">

基于 Ultralytics YOLO26 的安防报警系统项目集成了先进的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)能力，以增强安防措施。由 Ultralytics 开发的 YOLO26 提供实时[目标检测](https://www.ultralytics.com/glossary/object-detection)功能，使系统能够及时识别并响应潜在的安全威胁。该项目具有以下优势：

- **实时检测：**YOLO26 的高效性能使安防报警系统能够实时检测并响应安全事故，最大限度地缩短响应时间。
- **[准确率](https://www.ultralytics.com/glossary/accuracy)：**YOLO26 以其在目标检测方面的准确性而闻名，能够减少误报并提高安防报警系统的可靠性。
- **集成能力：**该项目可以与现有安防基础设施无缝集成，提供升级版的智能监控层。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/DTjtBnSK2fY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用 Ultralytics YOLO26 + Solutions 的安防报警系统 <a href="https://www.ultralytics.com/glossary/object-detection">目标检测</a>
</p>

???+ note

    需要生成应用专用密码

- 前往[应用专用密码生成器](https://myaccount.google.com/apppasswords)，指定一个应用名称（例如 "security project"），获取一个 16 位密码。复制此密码并粘贴到下方代码中指定的 `password` 字段。

!!! example "使用 Ultralytics YOLO 的安防报警系统"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        from_email = "abc@gmail.com"  # 发件人邮箱地址
        password = "---- ---- ---- ----"  # 通过 https://myaccount.google.com/apppasswords 生成的 16 位密码
        to_email = "xyz@gmail.com"  # 收件人邮箱地址

        # 初始化安防报警对象
        securityalarm = solutions.SecurityAlarm(
            show=True,  # 显示输出
            model="yolo26n.pt",  # 例如 yolo26s.pt, yolo26m.pt
            records=1,  # 发送邮件所需的总检测次数
        )

        securityalarm.authenticate(from_email, password, to_email)  # 认证邮件服务器

        # 处理视频
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = securityalarm(im0)

            # print(results)  # 访问输出

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

    === "CLI"

        ```bash
        yolo solutions security source="path/to/video.mp4" show=True
        ```

        !!! note
            邮件警报需要通过 Python API 调用 `.authenticate()`。CLI 仅提供检测和可视化功能。

运行代码后，如果检测到任何目标，您将收到一封邮件通知。通知是即时发送的，不会重复发送。您可以根据项目需求自定义代码。

#### 收到的邮件示例

<img width="256" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/email-received-sample.avif" alt="安防警报邮件通知示例">

### `SecurityAlarm` 参数

以下是 `SecurityAlarm` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "records"]) }}

`SecurityAlarm` 解决方案支持多种 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还提供以下可视化设置：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## 工作原理

安防报警系统使用[目标跟踪](https://docs.ultralytics.com/modes/track)来监控视频流并检测潜在的安全威胁。当系统检测到的目标超过指定阈值（由 `records` 参数设置）时，它会自动发送一封附有检测目标图像的邮件通知。

系统基于 [SecurityAlarm 类](https://docs.ultralytics.com/reference/solutions/security_alarm)实现，提供以下方法：

1. 处理帧并提取目标检测结果
2. 在检测到的目标周围标注边界框
3. 当检测阈值被超过时发送邮件通知

该实现非常适合家庭安防、零售监控以及其他需要即时通知检测到目标的监控应用场景。

## 常见问题

### Ultralytics YOLO26 如何提高安防报警系统的准确性？

Ultralytics YOLO26 通过提供高精度、实时的目标检测来增强安防报警系统。其先进算法显著减少了误报，确保系统仅对真正的威胁做出响应。这种更高的可靠性可以与现有安防基础设施无缝集成，提升整体监控质量。

### 可以将 Ultralytics YOLO26 与我现有的安防基础设施集成吗？

可以，Ultralytics YOLO26 能够与您现有的安防基础设施无缝集成。该系统支持多种模式，并提供灵活的自定义选项，使您能够通过先进的目标检测能力增强现有系统。有关在项目中集成 YOLO26 的详细说明，请访问[集成部分](https://docs.ultralytics.com/integrations)。

### 运行 Ultralytics YOLO26 需要多少存储空间？

在标准配置上运行 Ultralytics YOLO26 通常需要约 5GB 的可用磁盘空间。这包括存储 YOLO26 模型和任何额外依赖项的空间。对于云端解决方案，[Ultralytics Platform](https://docs.ultralytics.com/platform) 提供高效的项目管理和数据集处理功能，可以优化存储需求。了解有关 [Pro 计划](../platform/account/billing.md)的更多信息，获取包括扩展存储在内的增强功能。

### Ultralytics YOLO26 与其他目标检测模型（如 Faster R-CNN 或 SSD）有何不同？

Ultralytics YOLO26 相比 Faster R-CNN 或 SSD 等模型，具有实时检测能力和更高的准确性优势。其独特的架构使其能够在不牺牲[精度](https://www.ultralytics.com/glossary/precision)的情况下更快地处理图像，非常适合安防报警系统等对时间敏感的应用。有关目标检测模型的全面对比，请参阅我们的[指南](https://docs.ultralytics.com/models)。

### 如何使用 Ultralytics YOLO26 减少安防系统中的误报频率？

要减少误报，请确保您的 Ultralytics YOLO26 模型使用多样化且标注良好的数据集进行充分训练。微调超参数并定期用新数据更新模型可以显著提高检测准确性。详细的[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)技术请参阅我们的[超参数调优指南](../guides/hyperparameter-tuning.md)。
