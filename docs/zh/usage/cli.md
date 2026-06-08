---
comments: true
description: 探索 YOLO 命令行界面 (CLI)，无需 Python 环境即可轻松执行检测任务。
keywords: YOLO CLI, 命令行界面, YOLO 命令, 检测任务, Ultralytics, 模型训练, 模型预测
---

# 命令行界面

Ultralytics 命令行界面 (CLI) 提供了一种无需 Python 环境即可使用 Ultralytics YOLO 模型的简便方法。CLI 支持使用 `yolo` 命令直接从终端运行各种任务，无需任何自定义或 Python 代码。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=19"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>掌握 Ultralytics YOLO：CLI
</p>

!!! example

    === "语法"

        Ultralytics `yolo` 命令使用以下语法：
        ```bash
        yolo TASK MODE ARGS
        ```

        其中：
        - `TASK`（可选）是 [detect, segment, classify, pose, obb] 之一
        - `MODE`（必需）是 [train, val, predict, export, track, benchmark] 之一
        - `ARGS`（可选）是任意数量的自定义 `arg=value` 对，如 `imgsz=320`，用于覆盖默认值。

        在完整的[配置指南](cfg.md)中或使用 `yolo cfg` 查看所有 ARGS。

    === "训练"

        使用初始[学习率](https://www.ultralytics.com/glossary/learning-rate) 0.01 训练一个检测模型 10 个 [epoch](https://www.ultralytics.com/glossary/epoch)：

        ```bash
        yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
        ```

    === "预测"

        使用预训练的分割模型以图像大小 320 在 YouTube 视频上进行预测：

        ```bash
        yolo predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "验证"

        使用[批次大小](https://www.ultralytics.com/glossary/batch-size) 1 和图像大小 640 验证预训练的检测模型：

        ```bash
        yolo val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640
        ```

    === "导出"

        将 YOLO 分类模型导出为 ONNX 格式，图像大小为 224x128（无需 TASK）：

        ```bash
        yolo export model=yolo26n-cls.pt format=onnx imgsz=224,128
        ```

    === "特殊命令"

        运行特殊命令以查看版本、设置、运行检查等：

        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

其中：

- `TASK`（可选）是 `[detect, segment, classify, pose, obb]` 之一。如果未显式传递，YOLO 将尝试从模型类型推断 `TASK`。
- `MODE`（必需）是 `[train, val, predict, export, track, benchmark]` 之一
- `ARGS`（可选）是任意数量的自定义 `arg=value` 对，如 `imgsz=320`，用于覆盖默认值。有关可用 `ARGS` 的完整列表，请参见[配置](cfg.md)页面和 `default.yaml`。

!!! warning

    参数必须以 `arg=val` 对的形式传递，用等号 `=` 分隔，对之间用空格分隔。不要在参数前使用 `--` 前缀，也不要在参数之间使用逗号 `,`。

    - `yolo predict model=yolo26n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolo26n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolo26n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

## 训练

在 COCO8 数据集上以图像大小 640 训练 YOLO 100 个 epoch。有关可用参数的完整列表，请参见[配置](cfg.md)页面。

!!! example

    === "训练"

        在 COCO8 上以图像大小 640 开始训练 YOLO26n 100 个 epoch：

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

    === "恢复"

        恢复中断的训练会话：

        ```bash
        yolo detect train resume model=last.pt
        ```

## 验证

在 COCO8 数据集上验证训练模型的[准确度](https://www.ultralytics.com/glossary/accuracy)。无需参数，因为 `model` 将其训练 `data` 和参数作为模型属性保留。

!!! example

    === "官方模型"

        验证官方 YOLO26n 模型：

        ```bash
        yolo detect val model=yolo26n.pt
        ```

    === "自定义模型"

        验证自定义训练的模型：

        ```bash
        yolo detect val model=path/to/best.pt
        ```

## 预测

使用训练好的模型对图像运行预测。

!!! example

    === "官方模型"

        使用官方 YOLO26n 模型进行预测：

        ```bash
        yolo detect predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
        ```

    === "自定义模型"

        使用自定义模型进行预测：

        ```bash
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'
        ```

## 导出

将模型导出为不同的格式，如 ONNX 或 CoreML。

!!! example

    === "官方模型"

        将官方 YOLO26n 模型导出为 ONNX 格式：

        ```bash
        yolo export model=yolo26n.pt format=onnx
        ```

    === "自定义模型"

        将自定义训练的模型导出为 ONNX 格式：

        ```bash
        yolo export model=path/to/best.pt format=onnx
        ```

可用的 Ultralytics 导出格式见下表。你可以使用 `format` 参数导出为任何格式，即 `format='onnx'` 或 `format='engine'`。

{% include "macros/export-table.md" %}

在[导出](../modes/export.md)页面上查看完整的 `export` 详情。

## 覆盖默认参数

通过在 CLI 中以 `arg=value` 对的形式传递参数来覆盖默认参数。

!!! tip

    === "训练"

        使用学习率 0.01 训练检测模型 10 个 epoch：

        ```bash
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
        ```

    === "预测"

        使用预训练的分割模型以图像大小 320 在 YouTube 视频上进行预测：

        ```bash
        yolo segment predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "验证"

        使用批次大小 1 和图像大小 640 验证预训练的检测模型：

        ```bash
        yolo detect val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640
        ```

## 覆盖默认配置文件

通过使用 `cfg` 参数传入新文件（如 `cfg=custom.yaml`）来完全覆盖 `default.yaml` 配置文件。

为此，首先使用 `yolo copy-cfg` 命令在当前工作目录中创建 `default.yaml` 的副本，该命令会创建一个 `default_copy.yaml` 文件。

然后可以将此文件作为 `cfg=default_copy.yaml` 传递，同时传递任何额外参数，如本例中的 `imgsz=320`：

!!! example

    === "CLI"

        ```bash
        yolo copy-cfg
        yolo cfg=default_copy.yaml imgsz=320
        ```

## 解决方案命令

Ultralytics 通过 CLI 为常见的计算机视觉应用提供了开箱即用的解决方案。`yolo solutions` 命令公开了对象计数、裁剪、模糊、运动监控、热图、实例分割、VisionEye、速度估计、队列管理、分析、Streamlit 推理和基于区域的跟踪——完整目录请见[解决方案](../solutions/index.md)页面。运行 `yolo solutions help` 列出每个支持的解决方案及其参数。

!!! example

    === "计数"

        在视频或实时流中计数对象：

        ```bash
        yolo solutions count show=True
        yolo solutions count source="path/to/video.mp4" # 指定视频文件路径
        ```

    === "裁剪"

        裁剪检测到的对象并保存到磁盘：

        ```bash
        yolo solutions crop show=True
        yolo solutions crop source="path/to/video.mp4" # 指定视频文件路径
        yolo solutions crop classes="[0, 2]"           # 仅裁剪选定的类别
        ```

    === "模糊"

        模糊视频中检测到的对象以保护隐私或突出其他区域：

        ```bash
        yolo solutions blur show=True
        yolo solutions blur source="path/to/video.mp4" # 指定视频文件路径
        yolo solutions blur classes="[0, 5]"           # 仅模糊选定的类别
        ```

    === "运动"

        使用姿态模型监控锻炼动作：

        ```bash
        yolo solutions workout show=True
        yolo solutions workout source="path/to/video.mp4" # 指定视频文件路径

        # 使用关键点进行腹部锻炼
        yolo solutions workout kpts="[5, 11, 13]" # 左侧
        yolo solutions workout kpts="[6, 12, 14]" # 右侧
        ```

    === "热图"

        生成显示对象密度和移动模式的热图：

        ```bash
        yolo solutions heatmap show=True
        yolo solutions heatmap source="path/to/video.mp4"                                # 指定视频文件路径
        yolo solutions heatmap colormap=cv2.COLORMAP_INFERNO                             # 自定义颜色映射
        yolo solutions heatmap region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" # 将热图限制在某个区域
        ```

    === "实例分割"

        在视频上运行带跟踪的实例分割：

        ```bash
        yolo solutions isegment show=True
        yolo solutions isegment source="path/to/video.mp4" # 指定视频文件路径
        yolo solutions isegment classes="[0, 5]"           # 仅分割选定的类别
        ```

    === "VisionEye"

        使用 VisionEye 绘制对象到观察者的视线：

        ```bash
        yolo solutions visioneye show=True
        yolo solutions visioneye source="path/to/video.mp4" # 指定视频文件路径
        yolo solutions visioneye classes="[0, 5]"           # 仅监控选定的类别
        ```

    === "速度"

        估计视频中移动对象的速度：

        ```bash
        yolo solutions speed show=True
        yolo solutions speed source="path/to/video.mp4" # 指定视频文件路径
        yolo solutions speed meter_per_pixel=0.05       # 设置实际世界单位的比例尺
        ```

    === "队列"

        在指定队列或区域内计数对象：

        ```bash
        yolo solutions queue show=True
        yolo solutions queue source="path/to/video.mp4"                                # 指定视频文件路径
        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" # 配置队列坐标
        ```

    === "分析"

        从跟踪的检测结果生成分析图表（折线图、条形图、面积图或饼图）：

        ```bash
        yolo solutions analytics show=True
        yolo solutions analytics source="path/to/video.mp4" # 指定视频文件路径
        yolo solutions analytics analytics_type="pie" show=True
        yolo solutions analytics analytics_type="bar" show=True
        yolo solutions analytics analytics_type="area" show=True
        ```

    === "推理"

        使用 Streamlit 在网页浏览器中执行目标检测、实例分割或姿态估计：

        ```bash
        yolo solutions inference
        yolo solutions inference model="path/to/model.pt" # 使用自定义模型
        ```

    === "TrackZone"

        仅在指定的多边形区域内跟踪对象：

        ```bash
        yolo solutions trackzone show=True
        yolo solutions trackzone source="path/to/video.mp4"                                  # 指定视频文件路径
        yolo solutions trackzone region="[(150, 150), (1130, 150), (1130, 570), (150, 570)]" # 配置区域坐标
        ```

    === "区域"

        在特定多边形区域内计数对象：

        ```bash
        yolo solutions region show=True
        yolo solutions region source="path/to/video.mp4"                                # 指定视频文件路径
        yolo solutions region region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" # 配置区域坐标
        ```

    === "安全"

        使用目标检测运行安全警报监控：

        ```bash
        yolo solutions security show=True
        yolo solutions security source="path/to/video.mp4" # 指定视频文件路径
        ```

    === "停车"

        使用预定义区域监控停车场占用情况：

        ```bash
        yolo solutions parking source="path/to/video.mp4" json_file="bounding_boxes.json" # 需要预构建的 JSON
        yolo solutions parking source="path/to/video.mp4" json_file="bounding_boxes.json" model="yolo26n.pt"
        ```

    === "帮助"

        查看可用的解决方案及其选项：

        ```bash
        yolo solutions help
        ```

有关 Ultralytics 解决方案的更多信息，请访问[解决方案](../solutions/index.md)页面。

## FAQ

### 如何使用 Ultralytics YOLO 命令行界面 (CLI) 进行模型训练？

要使用 CLI 训练模型，在终端中执行单行命令。例如，使用[学习率](https://www.ultralytics.com/glossary/learning-rate) 0.01 训练检测模型 10 个 epoch，运行：

```bash
yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
```

此命令使用 `train` 模式并带有特定参数。有关可用参数的完整列表，请参见[配置指南](cfg.md)。

### 使用 Ultralytics YOLO CLI 可以执行哪些任务？

Ultralytics YOLO CLI 支持各种任务，包括[检测](../tasks/detect.md)、[分割](../tasks/segment.md)、[分类](../tasks/classify.md)、[姿态估计](../tasks/pose.md)和[定向边界框检测](../tasks/obb.md)。你还可以执行以下操作：

- **训练模型**：运行 `yolo train data=<data.yaml> model=<model.pt> epochs=<num>`。
- **运行预测**：使用 `yolo predict model=<model.pt> source=<data_source> imgsz=<image_size>`。
- **导出模型**：执行 `yolo export model=<model.pt> format=<export_format>`。
- **使用解决方案**：运行 `yolo solutions <solution_name>` 使用现成的应用程序。

使用各种参数自定义每个任务。详细语法和示例请参见相应章节，如[训练](#训练)、[预测](#预测)和[导出](#导出)。

### 如何使用 CLI 验证训练好的 YOLO 模型的准确度？

要验证模型的[准确度](https://www.ultralytics.com/glossary/accuracy)，使用 `val` 模式。例如，使用[批次大小](https://www.ultralytics.com/glossary/batch-size) 1 和图像大小 640 验证预训练的检测模型，运行：

```bash
yolo val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640
```

此命令在指定数据集上评估模型，并提供 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)、[精确度](https://www.ultralytics.com/glossary/precision)和[召回率](https://www.ultralytics.com/glossary/recall)等性能指标。更多详情请参见[验证](#验证)章节。

### 使用 CLI 可以将 YOLO 模型导出为哪些格式？

你可以将 YOLO 模型导出为各种格式，包括 ONNX、TensorRT、CoreML、TensorFlow 等。例如，将模型导出为 ONNX 格式，运行：

```bash
yolo export model=yolo26n.pt format=onnx
```

导出命令支持多种选项来优化模型以适应特定的部署环境。有关所有可用导出格式及其具体参数的完整详情，请访问[导出](../modes/export.md)页面。

### 如何在 Ultralytics CLI 中使用预构建的解决方案？

Ultralytics 通过 `solutions` 命令提供了开箱即用的解决方案。例如，在视频中计数对象：

```bash
yolo solutions count source="path/to/video.mp4"
```

这些解决方案需要最少的配置，并为常见的计算机视觉任务提供即时功能。要查看所有可用解决方案，运行 `yolo solutions help`。每个解决方案都有可以根据你的需求自定义的特定参数。