---
comments: true
description: 探索 Roboflow 100 数据集，包含 100 个多样化数据集，旨在测试目标检测模型在从医疗到视频游戏等各个领域的表现。
keywords: Roboflow 100, Ultralytics, 目标检测, 数据集, 基准测试, 机器学习, 计算机视觉, 多样化数据集, 模型评估
---

# Roboflow 100 数据集

Roboflow 100 由 [Intel](https://www.intel.com/) 赞助，是一个开创性的[目标检测](../../tasks/detect.md)基准数据集。它包含 100 个多样化数据集。该基准专门设计用于测试[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型（如 [Ultralytics YOLO 模型](../../models/yolo26.md)）对各种领域的适应性，包括医疗、航拍图像和视频游戏。

!!! question "许可"

    Ultralytics 提供两种许可选项以适应不同的使用场景：

    - **AGPL-3.0 许可证**：此 [OSI 批准](https://opensource.org/license)的开源许可证非常适合学生和爱好者，促进开放协作和知识共享。详情请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件，并访问我们的 [AGPL-3.0 许可证页面](https://www.ultralytics.com/legal/agpl-3-0-software-license)。
    - **企业许可证**：专为商业用途设计，此许可证允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品和服务中。如果你的场景涉及商业应用，请通过 [Ultralytics Licensing](https://www.ultralytics.com/license) 联系我们。

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/roboflow-100-overview.avif" alt="Roboflow 100 多样化目标检测基准">
</p>

## 主要特性

- **多样化领域**：包含七个不同领域的 100 个数据集：航拍、视频游戏、显微、水下、文档、电磁和真实世界。
- **规模**：该基准包含 224,714 张图像，涵盖 805 个类别，代表了超过 11,170 小时的[数据标注](https://www.ultralytics.com/glossary/data-labeling)工作量。
- **标准化**：所有图像均经过[预处理](https://www.ultralytics.com/glossary/data-preprocessing)并调整为 640x640 像素，以确保评估的一致性。
- **干净评估**：专注于消除类别歧义，并过滤掉代表性不足的类别，以确保更干净的[模型评估](../../guides/model-evaluation-insights.md)。
- **标注**：包含物体的[边界框](https://www.ultralytics.com/glossary/bounding-box)，适合使用 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) 等指标[训练](../../modes/train.md)和评估目标检测模型。

## 数据集结构

Roboflow 100 数据集分为七个类别，每个类别包含独特的数据集、图像和类别集合：

- **航拍**：7 个数据集，9,683 张图像，24 个类别。
- **视频游戏**：7 个数据集，11,579 张图像，88 个类别。
- **显微**：11 个数据集，13,378 张图像，28 个类别。
- **水下**：5 个数据集，18,003 张图像，39 个类别。
- **文档**：8 个数据集，24,813 张图像，90 个类别。
- **电磁**：12 个数据集，36,381 张图像，41 个类别。
- **真实世界**：50 个数据集，110,615 张图像，495 个类别。

这种结构为[目标检测](https://www.ultralytics.com/glossary/object-detection)模型提供了多样化且广泛的测试场地，反映了各种 [Ultralytics 解决方案](https://www.ultralytics.com/solutions)中广泛的实际应用场景。

## 基准测试

数据集[基准测试](../../modes/benchmark.md)涉及使用标准化指标评估[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型在特定数据集上的性能。常见指标包括[准确度](https://www.ultralytics.com/glossary/accuracy)、平均精度均值（mAP）和 [F1 分数](https://www.ultralytics.com/glossary/f1-score)。你可以在我们的 [YOLO 性能指标指南](../../guides/yolo-performance-metrics.md)中了解更多。

!!! tip "基准测试结果"

    使用提供的脚本进行基准测试的结果将存储在 `ultralytics-benchmarks/` 目录中，具体在 `evaluation.txt` 文件中。

!!! example "基准测试示例"

    以下脚本演示了如何使用 `RF100Benchmark` 类以编程方式对 Roboflow 100 基准中所有 100 个数据集上的 Ultralytics YOLO 模型（如 YOLO26n）进行基准测试。

    === "Python"

        ```python
        import os
        import shutil
        from pathlib import Path

        from ultralytics.utils.benchmarks import RF100Benchmark

        # 初始化 RF100Benchmark 并设置 API 密钥
        benchmark = RF100Benchmark()
        benchmark.set_key(api_key="YOUR_ROBOFLOW_API_KEY")

        # 解析数据集并定义文件路径
        names, cfg_yamls = benchmark.parse_dataset()
        val_log_file = Path("ultralytics-benchmarks") / "validation.txt"
        eval_log_file = Path("ultralytics-benchmarks") / "evaluation.txt"

        # 对 RF100 中的每个数据集运行基准测试
        for ind, path in enumerate(cfg_yamls):
            path = Path(path)
            if path.exists():
                # 修复 YAML 文件并运行训练
                benchmark.fix_yaml(str(path))
                os.system(f"yolo detect train data={path} model=yolo26s.pt epochs=1 batch=16")

                # 运行验证和评估
                os.system(f"yolo detect val data={path} model=runs/detect/train/weights/best.pt > {val_log_file} 2>&1")
                benchmark.evaluate(str(path), str(val_log_file), str(eval_log_file), ind)

                # 删除 'runs' 目录
                runs_dir = Path.cwd() / "runs"
                shutil.rmtree(runs_dir)
            else:
                print("YAML 文件路径不存在")
                continue

        print("RF100 基准测试完成！")
        ```

## 应用场景

Roboflow 100 对于与[计算机视觉](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025)和[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)相关的各种应用都非常有价值。研究人员和工程师可以利用此基准来：

- 评估目标检测模型在多领域上下文中的性能。
- 测试模型对超越常见[基准数据集](https://www.ultralytics.com/glossary/benchmark-dataset)（如 [COCO](https://cocodataset.org/#home) 或 [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)）的真实世界场景的适应性和[鲁棒性](<https://en.wikipedia.org/wiki/Robustness_(computer_science)>)。
- 对目标检测模型在多样化数据集上的能力进行基准测试，包括医疗、航拍图像和视频游戏等专业领域。
- 比较不同[神经网络](https://www.ultralytics.com/glossary/neural-network-nn)架构和[优化](https://www.ultralytics.com/glossary/optimization-algorithm)技术的模型性能。
- 识别可能需要专门的[模型训练技巧](../../guides/model-training-tips.md)或[微调](https://www.ultralytics.com/glossary/fine-tuning)方法（如[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)）的特定领域挑战。

有关实际应用的更多想法和灵感，请探索[我们的实用项目指南](../../guides/index.md)或查看 [Ultralytics 平台](https://platform.ultralytics.com)，以简化[模型训练](../../modes/train.md)和[部署](../../guides/model-deployment-options.md)。

## 使用方法

Roboflow 100 数据集（包括元数据和下载链接）可在官方 [Roboflow 100 GitHub 仓库](https://github.com/roboflow/roboflow-100-benchmark)上获取。你可以直接从那里访问和使用该数据集以满足你的基准测试需求。Ultralytics `RF100Benchmark` 工具简化了下载和准备这些数据集以与 Ultralytics 模型一起使用的过程。

## 示例数据与标注

Roboflow 100 由从各种角度和领域捕获的多样化图像数据集组成。以下是 RF100 基准中包含的标注图像示例，展示了物体和场景的多样性。在训练过程中，[数据增强](https://www.ultralytics.com/glossary/data-augmentation)等技术可以进一步增强多样性。

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/sample-data-annotations.avif" alt="Roboflow 100 标注示例图像">
</p>

Roboflow 100 基准中展现的多样性代表了传统基准的重大进步，传统基准通常侧重于在有限领域内优化单一指标。这种全面的方法有助于开发更鲁棒、更通用的[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型，能够在多种不同场景中表现良好。

## 引用与致谢

如果你在研究或开发工作中使用 Roboflow 100 数据集，请引用原始论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{rf100benchmark,
            Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
            Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
            Year = {2022},
            Eprint = {arXiv:2211.13523},
            url = {https://arxiv.org/abs/2211.13523}
        }
        ```

我们感谢 Roboflow 团队和所有贡献者为创建和维护 Roboflow 100 数据集所做的重大努力，该数据集是计算机视觉社区的宝贵资源。

如果你有兴趣探索更多数据集以增强你的目标检测和机器学习项目，欢迎访问[我们的综合数据集集合](../index.md)，其中包含各种其他[检测数据集](../detect/index.md)。

## 常见问题

### Roboflow 100 数据集是什么，为什么它对目标检测很重要？

**Roboflow 100** 数据集是[目标检测](../../tasks/detect.md)模型的基准。它包含 100 个多样化数据集，涵盖医疗、航拍图像和视频游戏等领域。其重要性在于提供了一种标准化的方式来测试模型在各种真实世界场景中的适应性和鲁棒性，超越了传统的、通常领域有限的基准。

### Roboflow 100 数据集涵盖哪些领域？

**Roboflow 100** 数据集涵盖七个多样化领域，为[目标检测](https://www.ultralytics.com/glossary/object-detection)模型提供独特的挑战：

1.  **航拍**：7 个数据集（如卫星图像、无人机视图）。
2.  **视频游戏**：7 个数据集（如各种游戏环境中的物体）。
3.  **显微**：11 个数据集（如细胞、颗粒）。
4.  **水下**：5 个数据集（如海洋生物、水下物体）。
5.  **文档**：8 个数据集（如文本区域、表单元素）。
6.  **电磁**：12 个数据集（如雷达信号、光谱数据可视化）。
7.  **真实世界**：50 个数据集（广泛类别，包括日常物品、场景、零售等）。

这种多样性使 RF100 成为评估计算机视觉模型[泛化能力](<https://en.wikipedia.org/wiki/Generalization_(learning)>)的绝佳资源。

### 在研究中使用 Roboflow 100 数据集时应如何引用？

使用 Roboflow 100 数据集时，请引用原始论文以向创建者致谢。以下是推荐的 BibTeX 引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{rf100benchmark,
            Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
            Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
            Year = {2022},
            Eprint = {arXiv:2211.13523},
            url = {https://arxiv.org/abs/2211.13523}
        }
        ```

如需进一步探索，可考虑访问我们的[综合数据集集合](../index.md)或浏览与 Ultralytics 模型兼容的其他[检测数据集](../detect/index.md)。
