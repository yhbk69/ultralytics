---
comments: true
description: 使用 Ultralytics YOLO 实现高效、灵活且可自定义的多目标跟踪。轻松学习实时视频流的跟踪。
keywords: 多目标跟踪, Ultralytics YOLO, 视频分析, 实时跟踪, 目标检测, AI, 机器学习
---

# 使用 Ultralytics YOLO 进行多目标跟踪

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/multi-object-tracking-examples.avif" alt="带轨迹路径的 YOLO 多目标跟踪">

视频分析领域中的目标跟踪是一项关键任务，它不仅能够在视频帧内识别目标的位置和类别，还能在视频进行过程中为每个检测到的目标维护唯一的 ID。其应用场景几乎无限——从监控安防到实时体育分析等各个领域均有涉及。

## 为什么选择 Ultralytics YOLO 进行目标跟踪？

Ultralytics 跟踪器的输出与标准的[目标检测](https://www.ultralytics.com/glossary/object-detection)结果一致，但额外提供了目标 ID。这使得在视频流中跟踪目标并进行后续分析变得非常简单。以下是选择 Ultralytics YOLO 进行目标跟踪的理由：

- **高效：** 在实时处理视频流的同时不牺牲[精度](https://www.ultralytics.com/glossary/accuracy)。
- **灵活：** 支持多种跟踪算法和配置。
- **易于使用：** 简洁的 Python API 和命令行选项，便于快速集成和部署。
- **可定制：** 可与自定义训练的 YOLO 模型轻松配合使用，便于集成到特定领域的应用中。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/qQkzKISt5GE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics YOLO26 运行多目标跟踪 | BoT-SORT 与 ByteTrack | VisionAI 🚀
</p>

## 实际应用

|           交通运输           |              零售业              |          水产养殖          |
| :--------------------------: | :------------------------------: | :------------------------: |
| ![车辆跟踪][vehicle track] | ![行人跟踪][people track] | ![鱼类跟踪][fish track] |
|           车辆跟踪           |            行人跟踪            |         鱼类跟踪         |

## 功能概览

Ultralytics YOLO 在其目标检测功能的基础上进行了扩展，提供了强大且多功能的目标跟踪能力：

- **实时跟踪：** 无缝跟踪高帧率视频中的目标。
- **多跟踪器支持：** 可从多种成熟的跟踪算法中选择。
- **可定制的跟踪器配置：** 通过调整各种参数，可根据特定需求定制跟踪算法。

## 可用的跟踪器

Ultralytics YOLO 支持以下跟踪算法。通过传入相应的 YAML 配置文件（如 `tracker=tracker_type.yaml`）即可启用：

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 使用 `botsort.yaml` 启用此跟踪器。
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) - 使用 `bytetrack.yaml` 启用此跟踪器。

默认跟踪器为 BoT-SORT。

## 跟踪

要在视频流上运行跟踪器，请使用训练好的检测、分割或姿态模型，如 YOLO26n、YOLO26n-seg 或 YOLO26n-pose。您可以在本地训练自定义模型，或通过 [Ultralytics 平台](https://platform.ultralytics.com) 在云端 GPU 上进行训练。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载官方或自定义模型
        model = YOLO("yolo26n.pt")  # 加载官方检测模型
        model = YOLO("yolo26n-seg.pt")  # 加载官方分割模型
        model = YOLO("yolo26n-pose.pt")  # 加载官方姿态模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练模型

        # 使用模型进行跟踪
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # 使用默认跟踪器进行跟踪
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # 使用 ByteTrack
        ```

    === "CLI"

        ```bash
        # 通过命令行界面使用不同模型进行跟踪
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4"      # 官方检测模型
        yolo track model=yolo26n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # 官方分割模型
        yolo track model=yolo26n-pose.pt source="https://youtu.be/LNwODJXcvt4" # 官方姿态模型
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" # 自定义训练模型

        # 使用 ByteTrack 跟踪器
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4" tracker="bytetrack.yaml"
        ```

从上方的用法可以看出，跟踪适用于对视频或流媒体源运行的所有检测、分割和姿态模型。

## 配置

### 跟踪参数

跟踪配置与预测模式共享属性，如 `conf`、`iou` 和 `show`。更多配置请参阅[预测](../modes/predict.md#inference-arguments)模型页面。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 配置跟踪参数并运行跟踪器
        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.1, iou=0.7, show=True)
        ```

    === "CLI"

        ```bash
        # 通过命令行界面配置跟踪参数并运行跟踪器
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.1 iou=0.7 show
        ```

### 跟踪器选择

Ultralytics 还允许您使用修改后的跟踪器配置文件。为此，只需从 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 复制一份跟踪器配置文件（例如 `custom_tracker.yaml`），然后根据需要修改任何配置（`tracker_type` 除外）。

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型并使用自定义配置文件运行跟踪器
        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        # 通过命令行界面加载模型并使用自定义配置文件运行跟踪器
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

有关每个参数的详细说明，请参阅[跟踪器参数](#tracker-arguments)部分。

### 跟踪器参数

某些跟踪行为可以通过编辑每个跟踪算法专用的 YAML 配置文件来进行微调。这些文件定义了阈值、缓冲区和匹配逻辑等参数：

- [`botsort.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)
- [`bytetrack.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml)

下表提供了每个参数的说明：

!!! warning "跟踪器阈值信息"

    如果检测到的置信度分数低于 [`track_high_thresh`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml#L5)，跟踪器将不会更新该目标，从而导致没有活跃的跟踪轨迹。

| **参数**             | **有效值或范围**                               | **描述**                                                                                                   |
| -------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `tracker_type`       | `botsort`, `bytetrack`                         | 指定跟踪器类型。可选 `botsort` 或 `bytetrack`。                                                              |
| `track_high_thresh`  | `0.0-1.0`                                      | 跟踪过程中首次关联使用的阈值。影响检测结果与现有轨迹匹配的置信度。                                              |
| `track_low_thresh`   | `0.0-1.0`                                      | 跟踪过程中第二次关联的阈值。在首次关联失败时使用，条件更宽松。                                                   |
| `new_track_thresh`   | `0.0-1.0`                                      | 当检测结果未匹配到任何现有轨迹时，初始化新轨迹的阈值。控制何时认为某个新目标出现。                                 |
| `track_buffer`       | `>=0`                                          | 缓冲区，用于指示丢失的轨迹在被移除之前应保留的帧数。值越大，对遮挡的容忍度越高。                                   |
| `match_thresh`       | `0.0-1.0`                                      | 匹配轨迹的阈值。值越高，匹配越宽松。                                                                            |
| `fuse_score`         | `True`, `False`                                | 决定是否在匹配前将置信度分数与 IoU 距离融合。有助于在关联时平衡空间信息和置信度信息。                              |
| `gmc_method`         | `orb`, `sift`, `ecc`, `sparseOptFlow`, `None`  | 用于全局运动补偿的方法。有助于补偿相机运动以改善跟踪效果。                                                        |
| `proximity_thresh`   | `0.0-1.0`                                      | ReID（重识别）有效匹配所需的最小 IoU。在使用外观线索之前确保空间接近性。                                          |
| `appearance_thresh`  | `0.0-1.0`                                      | ReID 所需的最小外观相似度。设置两个检测结果必须达到的视觉相似程度才能关联。                                        |
| `with_reid`          | `True`, `False`                                | 指示是否使用 ReID。启用基于外观的匹配，以便在遮挡期间获得更好的跟踪效果。仅 BoTSORT 支持。                         |
| `model`              | `auto`, `yolo26[nsmlx]-cls.pt`                  | 指定要使用的模型。默认为 `auto`，如果检测器是 YOLO 则使用原生特征，否则使用 `yolo26n-cls.pt`。                    |

### 启用重识别（ReID）

默认情况下，ReID 处于关闭状态以尽量减少性能开销。启用它非常简单——只需在[跟踪器配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/botsort.yaml)中设置 `with_reid: True` 即可。您还可以自定义用于 ReID 的 `model`，从而根据用例在精度和速度之间进行权衡：

- **原生特征（`model: auto`）**：直接利用 YOLO 检测器的特征进行 ReID，开销最小。当您需要一定程度的 ReID 而又不希望显著影响性能时，这是理想的选择。如果检测器不支持原生特征，则会自动回退到使用 `yolo26n-cls.pt`。
- **YOLO 分类模型**：您可以显式设置一个分类模型（例如 `yolo26n-cls.pt`）用于 ReID 特征提取。这样能提供更具区分性的嵌入向量，但由于增加了额外的推理步骤，会带来额外的延迟。

为了获得更好的性能，特别是在使用单独的分类模型进行 ReID 时，您可以将其导出到更快的后端（如 TensorRT）：

!!! example "将 ReID 模型导出到 TensorRT"

    ```python
    from torch import nn

    from ultralytics import YOLO

    # 加载分类模型
    model = YOLO("yolo26n-cls.pt")

    # 添加平均池化层
    head = model.model.model[-1]
    pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))
    pool.f, pool.i = head.f, head.i
    model.model.model[-1] = pool

    # 导出到 TensorRT
    model.export(format="engine", half=True, dynamic=True, batch=32)
    ```

导出后，您可以在跟踪器配置中指向 TensorRT 模型路径，它将在跟踪过程中用于 ReID。

## Python 示例

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/leOPZhE0ckg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何使用 Ultralytics YOLO 构建交互式目标跟踪 | 点击裁剪与显示 ⚡
</p>

### 持久化跟踪循环

以下是一个使用 [OpenCV](https://www.ultralytics.com/glossary/opencv) (`cv2`) 和 YOLO26 在视频帧上运行目标跟踪的 Python 脚本。此脚本假定已安装必要的包（`opencv-python` 和 `ultralytics`）。`persist=True` 参数告诉跟踪器当前图像或帧是序列中的下一帧，并期望在当前图像中看到来自上一帧的跟踪轨迹。

!!! example "使用跟踪的流式 for 循环"

    ```python
    import cv2

    from ultralytics import YOLO

    # 加载 YOLO26 模型
    model = YOLO("yolo26n.pt")

    # 打开视频文件
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频中读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行 YOLO26 跟踪，帧之间保持跟踪轨迹
            results = model.track(frame, persist=True)

            # 在帧上可视化结果
            annotated_frame = results[0].plot()

            # 显示标注后的帧
            cv2.imshow("YOLO26 Tracking", annotated_frame)

            # 如果按下 'q' 键则跳出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果到达视频末尾则跳出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

请注意，从 `model(frame)` 更改为 `model.track(frame)`，从而启用了目标跟踪而不是简单的检测。此修改后的脚本将在视频的每一帧上运行跟踪器，可视化结果并在窗口中显示。按 'q' 键可退出循环。

### 随时间绘制跟踪轨迹

在连续帧上可视化目标跟踪轨迹可以提供有关视频中被检测目标的运动模式和行为的重要洞察。使用 Ultralytics YOLO26，绘制这些轨迹是一个无缝且高效的过程。

在以下示例中，我们演示了如何利用 YOLO26 的跟踪功能来绘制多个视频帧中检测到的目标的运动轨迹。此脚本涉及打开视频文件、逐帧读取，并利用 YOLO 模型来识别和跟踪各种目标。通过保留检测到的边界框的中心点并将它们连接起来，我们可以绘制出代表被跟踪目标所遵循路径的线条。

!!! example "在多个视频帧上绘制跟踪轨迹"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    # 加载 YOLO26 模型
    model = YOLO("yolo26n.pt")

    # 打开视频文件
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 存储跟踪历史
    track_history = defaultdict(lambda: [])

    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频中读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行 YOLO26 跟踪，帧之间保持跟踪轨迹
            result = model.track(frame, persist=True)[0]

            # 获取边界框和跟踪 ID
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # 在帧上可视化结果
                frame = result.plot()

                # 绘制跟踪轨迹
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y 中心点
                    if len(track) > 30:  # 保留最近 30 帧的 30 个轨迹点
                        track.pop(0)

                    # 绘制跟踪线条
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # 显示标注后的帧
            cv2.imshow("YOLO26 Tracking", frame)

            # 如果按下 'q' 键则跳出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果到达视频末尾则跳出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

### 多线程跟踪

多线程跟踪提供了在多个视频流上同时运行目标跟踪的能力。这在处理多个视频输入（例如来自多个监控摄像头）时特别有用，并发处理可以显著提高效率和性能。

在提供的 Python 脚本中，我们利用 Python 的 `threading` 模块并发运行多个跟踪器实例。每个线程负责在一个视频文件上运行跟踪器，所有线程在后台同时运行。

为了确保每个线程收到正确的参数（视频文件、要使用的模型和文件索引），我们定义了一个函数 `run_tracker_in_thread`，它接受这些参数并包含主要的跟踪循环。此函数逐帧读取视频，运行跟踪器并显示结果。

此示例中使用了两个不同的模型：`yolo26n.pt` 和 `yolo26n-seg.pt`，每个模型在不同的视频文件中跟踪目标。视频文件在 `SOURCES` 中指定。

`threading.Thread` 中的 `daemon=True` 参数意味着这些线程将在主程序结束时立即关闭。然后我们使用 `start()` 启动线程，并使用 `join()` 使主线程等待，直到两个跟踪器线程都完成。

最后，在所有线程完成其任务后，使用 `cv2.destroyAllWindows()` 关闭显示结果的窗口。

!!! example "多线程跟踪实现"

    ```python
    import threading

    import cv2

    from ultralytics import YOLO

    # 定义模型名称和视频源
    MODEL_NAMES = ["yolo26n.pt", "yolo26n-seg.pt"]
    SOURCES = ["path/to/video.mp4", "0"]  # 本地视频，0 表示摄像头


    def run_tracker_in_thread(model_name, filename):
        """在自己的线程中运行 YOLO 跟踪器以实现并发处理。

        Args:
            model_name (str): YOLO26 模型对象。
            filename (str): 视频文件的路径或摄像头/外部摄像源的标识符。
        """
        model = YOLO(model_name)
        results = model.track(filename, save=True, stream=True)
        for r in results:
            pass


    # 使用 for 循环创建并启动跟踪器线程
    tracker_threads = []
    for video_file, model_name in zip(SOURCES, MODEL_NAMES):
        thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    # 等待所有跟踪器线程完成
    for thread in tracker_threads:
        thread.join()

    # 清理并关闭窗口
    cv2.destroyAllWindows()
    ```

此示例可以轻松扩展，通过创建更多线程并应用相同的方法来处理更多视频文件和模型。

## 贡献新的跟踪器

您是否精通多目标跟踪，并已成功使用 Ultralytics YOLO 实现或改造了跟踪算法？我们诚邀您为 [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 中的跟踪器部分做出贡献！您的实际应用和解决方案对于从事跟踪任务的用户来说可能是非常宝贵的资源。

通过为此部分做出贡献，您可以扩展 Ultralytics YOLO 框架内可用的跟踪解决方案的范围，为社区增加另一层功能和实用性。

要开始贡献，请参阅我们的[贡献指南](../help/contributing.md)，获取有关提交 Pull Request (PR) 的全面说明 🛠️。我们期待看到您的贡献！

让我们携手增强 Ultralytics YOLO 生态系统的跟踪能力 🙏！

[fish track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/fish-tracking.avif
[people track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/people-tracking.avif
[vehicle track]: https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vehicle-tracking.avif

## 常见问题

### 什么是多目标跟踪，Ultralytics YOLO 如何支持它？

视频分析中的多目标跟踪既涉及识别目标，也涉及在视频帧之间为每个检测到的目标维护唯一的 ID。Ultralytics YOLO 通过提供实时跟踪以及目标 ID 来支持这一点，从而便于完成安全监控和体育分析等任务。该系统使用 [BoT-SORT](https://github.com/NirAharon/BoT-SORT) 和 [ByteTrack](https://github.com/FoundationVision/ByteTrack) 等跟踪器，可通过 YAML 文件进行配置。

### 如何为 Ultralytics YOLO 配置自定义跟踪器？

您可以通过从 [Ultralytics 跟踪器配置目录](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) 复制现有的跟踪器配置文件（例如 `custom_tracker.yaml`），并根据需要修改参数（`tracker_type` 除外），来配置自定义跟踪器。然后在跟踪模型中像这样使用该文件：

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker="custom_tracker.yaml")
        ```

    === "CLI"

        ```bash
        yolo track model=yolo26n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

### 如何在多个视频流上同时运行目标跟踪？

要在多个视频流上同时运行目标跟踪，您可以使用 Python 的 `threading` 模块。每个线程处理一个单独的视频流。以下是设置示例：

!!! example "多线程跟踪"

    ```python
    import threading

    import cv2

    from ultralytics import YOLO

    # 定义模型名称和视频源
    MODEL_NAMES = ["yolo26n.pt", "yolo26n-seg.pt"]
    SOURCES = ["path/to/video.mp4", "0"]  # 本地视频，0 表示摄像头


    def run_tracker_in_thread(model_name, filename):
        """在自己的线程中运行 YOLO 跟踪器以实现并发处理。

        Args:
            model_name (str): YOLO26 模型对象。
            filename (str): 视频文件的路径或摄像头/外部摄像源的标识符。
        """
        model = YOLO(model_name)
        results = model.track(filename, save=True, stream=True)
        for r in results:
            pass


    # 使用 for 循环创建并启动跟踪器线程
    tracker_threads = []
    for video_file, model_name in zip(SOURCES, MODEL_NAMES):
        thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, video_file), daemon=True)
        tracker_threads.append(thread)
        thread.start()

    # 等待所有跟踪器线程完成
    for thread in tracker_threads:
        thread.join()

    # 清理并关闭窗口
    cv2.destroyAllWindows()
    ```

### 使用 Ultralytics YOLO 进行多目标跟踪有哪些实际应用？

使用 Ultralytics YOLO 进行多目标跟踪有众多应用，包括：

- **交通运输：** 用于交通管理和[自动驾驶](https://www.ultralytics.com/blog/ai-in-self-driving-cars)的车辆跟踪。
- **零售业：** 用于店内分析和安防的人员跟踪。
- **水产养殖：** 用于监测水生环境的鱼类跟踪。
- **体育分析：** 跟踪运动员和设备以进行表现分析。
- **安防系统：** [监控可疑活动](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8)并创建[安防警报](https://docs.ultralytics.com/guides/security-alarm-system)。

这些应用得益于 Ultralytics YOLO 以卓越精度实时处理高帧率视频的能力。

### 如何使用 Ultralytics YOLO 在多个视频帧上可视化目标跟踪轨迹？

要在多个视频帧上可视化目标跟踪轨迹，您可以使用 YOLO 模型的跟踪功能以及 OpenCV 来绘制检测到的目标的路径。以下是演示此功能的示例脚本：

!!! example "在多个视频帧上绘制跟踪轨迹"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)
    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            cv2.imshow("YOLO26 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

此脚本将绘制跟踪线条，显示被跟踪目标随时间变化的运动路径，从而为目标行为与模式提供宝贵的洞察。
