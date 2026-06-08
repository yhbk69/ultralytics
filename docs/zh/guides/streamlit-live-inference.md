---
comments: true
description: 了解如何使用 Streamlit 和 Ultralytics YOLO26 搭建实时目标检测应用。按照本逐步指南实现基于摄像头的目标检测。
keywords: Streamlit, YOLO26, 实时目标检测, Streamlit 应用, YOLO26 Streamlit 教程, 摄像头目标检测
---

# 使用 Ultralytics YOLO26 的 Streamlit 实时推理应用

## 简介

Streamlit 使得构建和部署交互式 Web 应用变得简单。将其与 Ultralytics YOLO26 结合，可以在浏览器中直接进行实时[目标检测](https://www.ultralytics.com/glossary/object-detection)与分析。YOLO26 的高精度和高速度确保了实时视频流的流畅性能，使其成为安防、零售等领域的理想选择。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Fm72tfuQG70"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO26 和 Streamlit 构建实时推理应用 | 检测与分割 🚀
</p>

|                                                                  水产养殖                                                                  |                                                               畜牧业                                                               |
| :---------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: |
| ![使用 Ultralytics YOLO26 进行鱼类检测](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/fish-detection-ultralytics-yolov8.avif) | ![使用 Ultralytics YOLO26 进行动物检测](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/animals-detection-yolov8.avif) |
|                                                  使用 Ultralytics YOLO26 进行鱼类检测                                                  |                                               使用 Ultralytics YOLO26 进行动物检测                                               |

## 实时推理的优势

- **无缝实时目标检测**：Streamlit 与 YOLO26 结合，可直接从摄像头画面进行实时目标检测。这使得即时分析和洞察成为可能，非常适合[需要即时反馈的应用](https://docs.ultralytics.com/modes/predict)。
- **用户友好的部署方式**：Streamlit 的交互式界面使得部署和使用应用变得容易，无需深厚的技术知识。用户只需简单点击即可开始实时推理，提高了可访问性和易用性。
- **高效的资源利用**：YOLO26 的优化算法确保以最小的计算资源实现高速处理。这种效率使得即使在标准硬件上也能实现流畅可靠的摄像头推理，让更广泛的用户群体能够接触到先进的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)技术。

## Streamlit 应用代码

!!! tip "Ultralytics 安装"

    在开始构建应用之前，请确保已安装 Ultralytics Python 包。

    ```bash
    pip install ultralytics
    ```

!!! example "使用 Streamlit 与 Ultralytics YOLO 进行推理"

    === "CLI"

        ```bash
        yolo solutions inference

        yolo solutions inference model="path/to/model.pt"
        ```

        这些命令将启动 Ultralytics 自带的默认 Streamlit 界面。如果需要自定义体验而不想编辑 Python 代码，可以使用 `yolo solutions inference --help` 查看其他参数，如 `source`、`conf` 或 `persist`。

    === "Python"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo26n.pt",  # 可以使用 Ultralytics 支持的任何模型，例如 YOLO26 或自定义训练的模型
        )

        inf.inference()

        # 请确保使用 `streamlit run path/to/file.py` 命令运行此文件
        ```

这将在默认 Web 浏览器中启动 Streamlit 应用。你将看到主标题、副标题以及包含配置选项的侧边栏。选择所需的 YOLO26 模型，设置置信度和 [NMS 阈值](https://www.ultralytics.com/glossary/non-maximum-suppression-nms)，然后点击"Start"按钮即可开始实时目标检测。

## 工作原理

在底层，Streamlit 应用使用 [Ultralytics solutions 模块](https://docs.ultralytics.com/reference/solutions/streamlit_inference)创建交互式界面。当你启动推理时，应用会：

1. 从摄像头或上传的视频文件中捕获视频
2. 通过 YOLO26 模型处理每一帧
3. 使用指定的置信度和 IoU 阈值进行目标检测
4. 实时显示原始帧和标注帧
5. 如果选中，可选择启用目标追踪

该应用提供了简洁、用户友好的界面，可以随时调整模型参数并启动/停止推理。

## 总结

通过遵循本指南，你已成功创建了一个使用 Streamlit 和 Ultralytics YOLO26 的实时目标检测应用。该应用让你能够通过摄像头体验 YOLO26 强大的目标检测能力，拥有用户友好的界面，并可随时停止视频流。

如需进一步改进，你可以探索添加更多功能，例如录制视频流、保存标注帧，或与其他[计算机视觉库](https://www.ultralytics.com/blog/exploring-vision-ai-frameworks-tensorflow-pytorch-and-opencv)集成。

## 与社区分享你的想法

参与社区互动，学习更多知识、解决问题并分享你的项目：

### 在哪里获取帮助和支持

- **GitHub Issues：**访问 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics/issues)提出问题、报告 Bug 和建议新功能。
- **Ultralytics Discord 服务器：**加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)与其他用户和开发者交流，获取支持，分享知识并碰撞想法。

### 官方文档

- **Ultralytics YOLO26 文档：**参考[官方 YOLO26 文档](https://docs.ultralytics.com/)获取关于各种计算机视觉任务和项目的全面指南与见解。

## 常见问题

### 如何使用 Streamlit 和 Ultralytics YOLO26 搭建实时目标检测应用？

使用 Streamlit 和 Ultralytics YOLO26 搭建实时目标检测应用非常简单。首先，确保已安装 Ultralytics Python 包：

```bash
pip install ultralytics
```

然后，你可以创建一个基本的 Streamlit 应用来运行实时推理：

!!! example "Streamlit 应用"

    === "Python"

        ```python
        from ultralytics import solutions

        inf = solutions.Inference(
            model="yolo26n.pt",  # 可以使用 Ultralytics 支持的任何模型，例如 YOLO26、YOLOv10
        )

        inf.inference()

        # 请确保使用 `streamlit run path/to/file.py` 命令运行此文件
        ```

    === "CLI"

        ```bash
        yolo solutions inference
        ```

有关实际设置的更多详细信息，请参考文档中的 [Streamlit 应用代码部分](#streamlit-应用代码)。

### 使用 Ultralytics YOLO26 和 Streamlit 进行实时目标检测的主要优势是什么？

使用 Ultralytics YOLO26 和 Streamlit 进行实时目标检测具有以下优势：

- **无缝实时检测**：直接从摄像头画面实现高[精度](https://www.ultralytics.com/glossary/accuracy)的实时目标检测。
- **用户友好的界面**：Streamlit 直观的界面让使用和部署变得容易，无需深厚的技术知识。
- **资源高效**：YOLO26 的优化算法确保以最小的计算资源实现高速处理。

在[实时推理的优势部分](#实时推理的优势)了解更多关于这些优势的信息。

### 如何在 Web 浏览器中部署 Streamlit 目标检测应用？

编写好集成 Ultralytics YOLO26 的 Streamlit 应用后，可以通过以下命令部署：

```bash
streamlit run path/to/file.py
```

此命令将在默认 Web 浏览器中启动应用，你可以选择 YOLO26 模型，设置置信度和 NMS 阈值，然后一键开始实时目标检测。详细指南请参考 [Streamlit 应用代码](#streamlit-应用代码)部分。

### 使用 Streamlit 和 Ultralytics YOLO26 进行实时目标检测有哪些应用场景？

使用 Streamlit 和 Ultralytics YOLO26 的实时目标检测可应用于多个领域：

- **安防**：实时监控非法入侵和[安防警报系统](https://docs.ultralytics.com/guides/security-alarm-system)。
- **零售**：顾客计数、货架管理和[库存追踪](https://www.ultralytics.com/blog/from-shelves-to-sales-exploring-yolov8s-impact-on-inventory-management)。
- **野生动物与农业**：监测动物和作物状况，支持[保护工作](https://www.ultralytics.com/blog/ai-in-wildlife-conservation)。

更多深入的应用场景和示例，请探索 [Ultralytics Solutions](https://docs.ultralytics.com/solutions)。

### Ultralytics YOLO26 与 YOLOv5 和 RCNN 等其他目标检测模型相比如何？

Ultralytics YOLO26 相比 YOLOv5 和 RCNN 等早期模型提供了多项改进：

- **更高的速度和精度**：针对实时应用提升了性能。
- **更易使用**：简化了接口和部署流程。
- **资源高效**：优化后在最小计算需求下实现更快的速度。

如需全面对比，请查看 [Ultralytics YOLO26 文档](https://docs.ultralytics.com/models/yolo26)以及讨论模型性能的相关博客文章。
