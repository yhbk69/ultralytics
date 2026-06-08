---
comments: true
description: 探索评估和优化 YOLO26 模型性能的最有效方法。了解评估指标、微调过程，以及如何为特定需求定制模型。
keywords: 模型评估, 机器学习模型评估, 机器学习微调, 模型微调, 评估模型, 模型微调, 如何微调模型
---

# 模型评估与微调见解

## 简介

一旦你[训练](./model-training-tips.md)了计算机视觉模型，评估并优化它以获得最佳性能至关重要。仅仅训练模型是不够的。你需要确保模型准确、高效，并满足计算机视觉项目的[目标](./defining-project-goals.md)。通过评估和微调模型，你可以识别弱点，提高其准确性，并提升整体性能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/-aYO-6VaDrw"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 模型评估与微调见解 | 提高平均精度均值的技巧
</p>

在本指南中，我们分享关于模型评估和微调的见解，使[计算机视觉项目中的这一步骤](./steps-of-a-cv-project.md)更易于理解。我们讨论如何理解评估指标并实施微调技术，为你提供提升模型能力的知识。

## 使用指标评估模型性能

评估模型的表现如何，有助于我们了解其工作效果。使用各种指标来衡量性能。这些[性能指标](./yolo-performance-metrics.md)提供了清晰的数值洞察，可以指导改进，确保模型达到预期目标。让我们仔细看看几个关键指标。

### 置信度分数

置信度分数表示模型对检测到的对象属于特定类别的确定程度。其范围从 0 到 1，分数越高表示置信度越高。置信度分数有助于过滤预测；只有置信度分数高于指定阈值的检测才被视为有效。

_快速提示：_ 运行推理时，如果看不到任何预测，并且你已经检查了其他所有方面，请尝试降低置信度分数。有时阈值设置得太高，导致模型忽略有效的预测。降低分数可以让模型考虑更多可能性。这可能不符合你的项目目标，但这是了解模型能力并决定如何对其进行微调的好方法。

### 交并比

[交并比](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) 是[目标检测](https://www.ultralytics.com/glossary/object-detection)中的一个指标，用于衡量预测的[边界框](https://www.ultralytics.com/glossary/bounding-box)与真实边界框的重叠程度。IoU 值的范围从 0 到 1，其中 1 表示完美匹配。IoU 至关重要，因为它衡量预测边界与实际对象边界的匹配程度。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/intersection-over-union-overview.avif" alt="交并比概述">
</p>

### 平均精度均值

[平均精度均值](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) 是衡量目标检测模型性能的一种方法。它查看检测每个对象类别的精度，对这些分数进行平均，并给出一个总体数字，显示模型识别和分类对象的准确程度。

让我们关注两个特定的 mAP 指标：

- *mAP@.5:* 在单个 IoU（交并比）阈值为 0.5 时测量平均精度。此指标检查模型是否能在较宽松的[准确度](https://www.ultralytics.com/glossary/accuracy)要求下正确找到对象。它关注对象是否大致在正确位置，不需要完美放置。它有助于了解模型是否通常擅长发现对象。
- *mAP@.5:.95:* 计算多个 IoU 阈值（从 0.5 到 0.95，步长为 0.05）下的 mAP 值的平均值。此指标更详细、更严格。它更全面地展示了模型在不同严格程度下找到对象的准确程度，对于需要精确目标检测的应用特别有用。

其他 mAP 指标包括 mAP@0.75（使用更严格的 IoU 阈值 0.75）和 mAP@small、medium 和 large（评估不同大小对象的精度）。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mean-average-precision-overview.avif" alt="平均精度均值 mAP 指标">
</p>

## 评估 YOLO26 模型性能

对于 YOLO26，你可以使用[验证模式](../modes/val.md)来评估模型。另外，请务必查看我们深入探讨[YOLO26 性能指标](./yolo-performance-metrics.md)及其解释的指南。

### 常见社区问题

评估 YOLO26 模型时，你可能会遇到一些小问题。根据常见的社区问题，以下是一些帮助你充分利用 YOLO26 模型的提示：

#### 处理不同尺寸的图像

使用不同尺寸的图像评估 YOLO26 模型，可以帮助你了解其在多样化数据集上的性能。使用 `rect=true` 验证参数，YOLO26 会根据图像尺寸调整每个批次的网络步长，使模型能够处理矩形图像，而无需强制将其调整为单一尺寸。

`imgsz` 验证参数设置图像调整大小的最大尺寸，默认为 640。你可以根据数据集的最大尺寸和可用的 GPU 内存进行调整。即使设置了 `imgsz`，`rect=true` 也能让模型通过动态调整步长来有效处理不同尺寸的图像。

#### 访问 YOLO26 指标

如果你想更深入地了解 YOLO26 模型的性能，可以通过几行 Python 代码轻松访问特定的评估指标。下面的代码片段将让你加载模型、运行评估，并打印出显示模型表现的各种指标。

!!! example "用法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 运行评估
        results = model.val(data="coco8.yaml")

        # 打印特定指标
        print("具有平均精度的类别索引:", results.ap_class_index)
        print("所有类别的平均精度:", results.box.all_ap)
        print("平均精度:", results.box.ap)
        print("IoU=0.50 时的平均精度:", results.box.ap50)
        print("平均精度的类别索引:", results.box.ap_class_index)
        print("特定类别结果:", results.box.class_result)
        print("F1 分数:", results.box.f1)
        print("F1 分数曲线:", results.box.f1_curve)
        print("整体适应度分数:", results.box.fitness)
        print("平均精度均值:", results.box.map)
        print("IoU=0.50 时的平均精度均值:", results.box.map50)
        print("IoU=0.75 时的平均精度均值:", results.box.map75)
        print("不同 IoU 阈值下的平均精度均值:", results.box.maps)
        print("不同指标的平均结果:", results.box.mean_results)
        print("平均精度:", results.box.mp)
        print("平均召回率:", results.box.mr)
        print("每张图像的指标:", results.box.image_metrics)
        print("精度:", results.box.p)
        print("精度曲线:", results.box.p_curve)
        print("精度值:", results.box.prec_values)
        print("特定精度指标:", results.box.px)
        print("召回率:", results.box.r)
        print("召回率曲线:", results.box.r_curve)
        ```

结果对象还包括 `image_metrics`，这是一个按图像文件名索引的每张图像字典，包含 `precision`、`recall`、`f1`、`tp`、`fp` 和 `fn`，以及速度指标，如预处理时间、推理时间、损失和后处理时间。通过分析这些指标，你可以微调和优化 YOLO26 模型以获得更好的性能，使其更有效地适用于你的特定用例。

## 微调如何工作？

微调涉及采用预训练模型并调整其参数，以提高在特定任务或数据集上的性能。这个过程也称为模型再训练，使模型能够更好地理解和预测在实际应用中遇到的特定数据的结果。你可以根据模型评估结果重新训练模型，以获得最佳结果。

## 微调模型的技巧

微调模型意味着密切关注几个关键参数和技术，以获得最佳性能。以下是一些指导你完成此过程的基本技巧。

### 从较高的学习率开始

通常，在初始训练[轮次](https://www.ultralytics.com/glossary/epoch)期间，学习率从较低开始，逐渐增加以稳定训练过程。然而，由于你的模型已经从之前的数据集中学习了一些特征，立即从较高的[学习率](https://www.ultralytics.com/glossary/learning-rate)开始可能更有益。

评估 YOLO26 模型时，可以将 `warmup_epochs` 验证参数设置为 `warmup_epochs=0`，以防止学习率起始过低。通过遵循此过程，训练将从提供的权重继续，适应新数据的细微差别。

### 针对小目标的图像分块

图像分块可以提高小目标的检测精度。通过将较大的图像分割成较小的片段，例如将 1280x1280 图像分割成多个 640x640 片段，你可以保持原始分辨率，并且模型可以从高分辨率片段中学习。使用 YOLO26 时，请确保正确调整这些新片段的标签。

## 参与社区

与其他[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)爱好者分享你的想法和问题，可以激发创造性解决方案，克服项目中的障碍。以下是一些学习、故障排除和联系的好方法。

### 寻求帮助和支持

- **GitHub Issues：** 探索 YOLO26 GitHub 仓库，并使用 [Issues 标签](https://github.com/ultralytics/ultralytics/issues) 提问、报告错误和提出功能建议。社区和维护者随时准备帮助你解决遇到的任何问题。
- **Ultralytics Discord 服务器：** 加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)，与其他用户和开发者联系，获得支持，分享知识，并集思广益。

### 官方文档

- **Ultralytics YOLO26 文档：** 查看 [官方 YOLO26 文档](./index.md)，获取关于各种计算机视觉任务和项目的全面指南和宝贵见解。

## 最后思考

评估和微调计算机视觉模型是成功[模型部署](https://www.ultralytics.com/glossary/model-deployment)的重要步骤。这些步骤有助于确保模型准确、高效，并适合你的整体应用。训练最佳模型的关键在于持续的实验和学习。不要犹豫，调整参数，尝试新技术，探索不同的数据集。不断实验，突破可能的界限！

## 常见问题解答

### 评估 YOLO26 模型性能的关键指标有哪些？

评估 YOLO26 模型性能的重要指标包括置信度分数、交并比 (IoU) 和平均精度均值 (mAP)。置信度分数衡量模型对每个检测到的对象类别的确定程度。IoU 评估预测边界框与真实边界框的重叠程度。平均精度均值 (mAP) 聚合了跨类别的精度分数，其中 mAP@.5 和 mAP@.5:.95 是两种常见的针对不同 IoU 阈值的类型。在我们的 [YOLO26 性能指标指南](./yolo-performance-metrics.md) 中了解更多关于这些指标的信息。

### 如何针对我的特定数据集微调预训练的 YOLO26 模型？

微调预训练的 YOLO26 模型涉及调整其参数，以提高在特定任务或数据集上的性能。首先使用指标评估你的模型，然后通过将 `warmup_epochs` 参数调整为 0 来设置较高的初始学习率，以获得即时稳定性。使用 `rect=true` 等参数来有效处理不同尺寸的图像。有关更详细的指导，请参阅我们关于[微调 YOLO26 模型](#how-does-fine-tuning-work)的部分。

### 评估 YOLO26 模型时如何处理不同尺寸的图像？

为了在评估期间处理不同尺寸的图像，请在 YOLO26 中使用 `rect=true` 参数，该参数会根据图像尺寸调整每个批次的网络步长。`imgsz` 参数设置图像调整大小的最大尺寸，默认为 640。调整 `imgsz` 以适应你的数据集和 GPU 内存。有关更多详细信息，请访问我们关于[处理不同尺寸图像](#handling-variable-image-sizes)的部分。

### 我可以采取哪些实际步骤来提高 YOLO26 模型的平均精度均值？

提高 YOLO26 模型的平均精度均值 (mAP) 涉及几个步骤：

1. **调整超参数：** 尝试不同的学习率、[批次大小](https://www.ultralytics.com/glossary/batch-size) 和图像增强。
2. **[数据增强](https://www.ultralytics.com/glossary/data-augmentation)：** 使用 Mosaic 和 MixUp 等技术创建多样化的训练样本。
3. **图像分块：** 将较大的图像分割成较小的块，以提高小目标的检测精度。
   有关具体策略，请参阅我们关于[模型微调](#tips-for-fine-tuning-your-model)的详细指南。

### 如何在 Python 中访问 YOLO26 模型评估指标？

你可以使用 Python 通过以下步骤访问 YOLO26 模型评估指标：

!!! example "用法"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolo26n.pt")

        # 运行评估
        results = model.val(data="coco8.yaml")

        # 打印特定指标
        print("具有平均精度的类别索引:", results.ap_class_index)
        print("所有类别的平均精度:", results.box.all_ap)
        print("IoU=0.50 时的平均精度均值:", results.box.map50)
        print("平均召回率:", results.box.mr)
        ```

分析这些指标有助于微调和优化你的 YOLO26 模型。要深入了解，请查看我们关于 [YOLO26 指标](../modes/val.md) 的指南。