---
comments: true
description: 使用 Ultralytics YOLO26 优化停车位并提升安全性。探索实时车辆检测与智能停车解决方案。
keywords: 停车管理, YOLO26, Ultralytics, 车辆检测, 实时跟踪, 停车场优化, 智能停车
---

# 使用 Ultralytics YOLO26 进行停车管理 🚀

## 什么是停车管理系统？

借助 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) 进行停车管理，可通过组织空间和监控可用性，确保高效、安全的停车体验。YOLO26 通过实时车辆检测和停车占用情况洞察，有效改善停车场管理。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hsimB10D6Y0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 构建停车管理系统 | 实时车位检测 🚗
</p>

## 停车管理系统的优势

- **效率**：停车场管理优化了停车位的使用，减少了拥堵。
- **安全与安保**：基于 YOLO26 的停车管理通过监控与安保措施提升了人员与车辆的安全性。
- **减少排放**：基于 YOLO26 的停车管理可管理交通流量，最大限度地减少停车场内的怠速时间和排放。

## 实际应用

|                                                                      停车管理系统                                                                      |                                                                       停车管理系统                                                                       |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO26 进行停车场鸟瞰分析](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/parking-management-aerial-view-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO26 进行停车管理俯视图](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/parking-management-top-view-ultralytics-yolov8.avif) |
|                                                       使用 Ultralytics YOLO26 的停车管理鸟瞰视图                                                        |                                                         使用 Ultralytics YOLO26 的停车管理俯视图                                                          |

## 停车管理系统代码工作流

??? note "点位选择现已简化"

    选择停车点位是停车管理系统中一项关键且复杂的任务。Ultralytics 通过提供"停车位标注器"工具简化了这一流程，让您可以定义停车区域，并用于后续处理。

**第一步：** 从您要管理停车场的视频或摄像头流中截取一帧画面。

**第二步：** 使用提供的代码启动图形界面，在该界面中您可以选择图像，并通过鼠标点击创建多边形来勾勒停车区域。

!!! example "Ultralytics YOLO 停车位标注器"

    ??? note "安装 `tkinter` 的额外步骤"

        通常情况下，`tkinter` 会随 Python 一起预装。但如果未安装，您可以按照以下步骤进行安装：

        - **Linux**（Debian/Ubuntu）：`sudo apt install python3-tk`
        - **Fedora**：`sudo dnf install python3-tkinter`
        - **Arch**：`sudo pacman -S tk`
        - **Windows**：重新安装 Python，并在安装过程中勾选**可选功能**中的 `tcl/tk and IDLE`
        - **MacOS**：从 [https://www.python.org/downloads/macos/](https://www.python.org/downloads/macos/) 重新安装 Python，或执行 `brew install python-tk`

    === "Python"

        ```python
        from ultralytics import solutions

        solutions.ParkingPtsSelection()
        ```

**第三步：** 使用多边形定义好停车区域后，点击 `save` 将包含数据的 JSON 文件保存到您的工作目录中。

![Ultralytics YOLO26 点位选择演示](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-points-selection-demo.avif)

**第四步：** 现在您可以使用以下代码，借助 Ultralytics YOLO 进行停车管理。

!!! example "使用 Ultralytics YOLO 进行停车管理"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        # 视频捕获
        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "读取视频文件出错"

        # 视频写入器
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("parking management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # 初始化停车管理对象
        parkingmanager = solutions.ParkingManagement(
            model="yolo26n.pt",  # 模型文件路径
            json_file="bounding_boxes.json",  # 停车标注文件路径
        )

        while cap.isOpened():
            ret, im0 = cap.read()
            if not ret:
                break

            results = parkingmanager(im0)

            # print(results)  # 访问输出结果

            video_writer.write(results.plot_im)  # 写入处理后的帧

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # 销毁所有打开的窗口
        ```

    === "CLI"

        ```bash
        yolo solutions parking source="path/to/video.mp4" json_file="bounding_boxes.json" show=True
        ```

        !!! note
            请先在 Python 中使用 `ParkingPtsSelection()`（上述第二步）创建停车区域标注，然后将 JSON 文件传递给 CLI 命令。

### `ParkingManagement` 参数

以下是 `ParkingManagement` 的参数表：

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "json_file"]) }}

`ParkingManagement` 方案支持使用多个 `track` 参数：

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

此外，还支持以下可视化选项：

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## 常见问题

### Ultralytics YOLO26 如何增强停车管理系统？

Ultralytics YOLO26 通过提供**实时车辆检测**和监控，极大地增强了停车管理系统。这有助于优化停车位使用、减少拥堵，并通过持续监控提升安全性。[停车管理系统](https://github.com/ultralytics/ultralytics)可实现高效的交通流，最大限度地减少停车场内的怠速时间和排放，从而为环境可持续发展做出贡献。更多详情请参阅[停车管理系统代码工作流](#parking-management-system-code-workflow)。

### 使用 Ultralytics YOLO26 进行智能停车有哪些好处？

使用 Ultralytics YOLO26 进行智能停车可带来诸多好处：

- **效率**：优化停车位使用，减少拥堵。
- **安全与安保**：增强监控，确保车辆与行人的安全。
- **环境影响**：通过减少车辆怠速时间帮助降低排放。更多优势详见[停车管理系统的优势](#advantages-of-parking-management-system)一节。

### 如何使用 Ultralytics YOLO26 定义停车位？

使用 Ultralytics YOLO26 定义停车位非常简单：

1. 从视频或摄像头流中截取一帧画面。
2. 使用提供的代码启动图形界面，选择图像并绘制多边形来定义停车位。
3. 将标注数据以 JSON 格式保存，供后续处理使用。完整说明请参阅上方的点位选择部分。

### 能否针对特定停车管理需求自定义 YOLO26 模型？

可以。Ultralytics YOLO26 支持针对特定停车管理需求进行自定义。您可以调整**已占用区域颜色和可用区域颜色**、文字显示的边距等参数。通过 `ParkingManagement` 类的[参数](#parkingmanagement-arguments)，您可以根据具体需求对模型进行定制，确保最佳效率和效果。

### Ultralytics YOLO26 在停车场管理中有哪些实际应用？

Ultralytics YOLO26 在停车场管理中的实际应用包括：

- **车位检测**：精准识别可用和已占用的车位。
- **监控**：通过实时监控增强安全性。
- **交通流管理**：通过高效的交通调度减少怠速时间和拥堵。展示这些应用的图像可在[实际应用](#real-world-applications)一节中查看。
