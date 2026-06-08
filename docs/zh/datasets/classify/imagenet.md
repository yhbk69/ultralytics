---
comments: true
description: 探索庞大的 ImageNet 数据集，了解它在推动计算机视觉深度学习方面的重要作用。获取预训练模型和训练示例。
keywords: ImageNet, 深度学习, 视觉识别, 计算机视觉, 预训练模型, YOLO, 数据集, 目标检测, 图像分类
---

# ImageNet 数据集

[ImageNet](https://www.image-net.org/) 是一个大规模的标注图像数据库，专为视觉目标识别研究而设计。它包含超过 1400 万张图像，每张图像使用 WordNet 同义词集进行标注，是[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务中训练[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)模型最广泛的资源之一。

## ImageNet 预训练模型

{% include "macros/yolo-cls-perf.md" %}

## 关键特性

- ImageNet 包含超过 1400 万张高分辨率图像，涵盖数千个对象类别。
- 数据集按照 WordNet 层级结构组织，每个同义词集代表一个类别。
- ImageNet 广泛用于计算机视觉领域的训练和基准测试，特别是[图像分类](https://www.ultralytics.com/glossary/image-classification)和[目标检测](https://www.ultralytics.com/glossary/object-detection)任务。
- 一年一度的 ImageNet 大规模视觉识别挑战赛 (ILSVRC) 在推动计算机视觉研究方面发挥了重要作用。

## 数据集结构

ImageNet 数据集使用 WordNet 层级结构组织。层级结构中的每个节点代表一个类别，每个类别由一个同义词集（一组同义词的集合）描述。ImageNet 中的图像使用一个或多个同义词集进行标注，为训练模型识别各种对象及其关系提供了丰富的资源。

## ImageNet 大规模视觉识别挑战赛 (ILSVRC)

一年一度的 [ImageNet 大规模视觉识别挑战赛 (ILSVRC)](https://image-net.org/challenges/LSVRC/) 是计算机视觉领域的重要事件。它为研究人员和开发者提供了一个平台，使其能够在大规模数据集上使用标准化评估指标来评估其算法和模型。ILSVRC 在图像分类、目标检测和其他计算机视觉任务的深度学习模型开发方面取得了重大进展。

## 应用

ImageNet 数据集广泛用于训练和评估各种计算机视觉任务中的深度学习模型，如图像分类、目标检测和目标定位。一些流行的深度学习架构，如 [AlexNet](https://en.wikipedia.org/wiki/AlexNet)、[VGG](https://arxiv.org/abs/1409.1556) 和 [ResNet](https://arxiv.org/abs/1512.03385)，都是使用 ImageNet 数据集进行开发和基准测试的。

## 使用方法

要在 ImageNet 数据集上训练深度学习模型 100 个 [epoch](https://www.ultralytics.com/glossary/epoch)，图像大小为 224x224，可以使用以下代码片段。有关可用参数的完整列表，请参阅模型[训练](../../modes/train.md)页面。

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

## 示例图像与标注

ImageNet 数据集包含涵盖数千个对象类别的高分辨率图像，为训练和评估计算机视觉模型提供了多样化和广泛的数据集。以下是数据集中图像的一些示例：

![ImageNet 分类数据集样本图像](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/imagenet-sample-images.avif)

该示例展示了 ImageNet 数据集中图像的多样性和复杂性，强调了多样化数据集对于训练鲁棒计算机视觉模型的重要性。

## 引用与致谢

如果您在研究或开发工作中使用了 ImageNet 数据集，请引用以下论文：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{ILSVRC15,
                 author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
                 title={ImageNet Large Scale Visual Recognition Challenge},
                 year={2015},
                 journal={International Journal of Computer Vision (IJCV)},
                 volume={115},
                 number={3},
                 pages={211-252}
        }
        ```

我们要感谢由 Olga Russakovsky、Jia Deng 和 Li Fei-Fei 领导的 ImageNet 团队创建并维护 ImageNet 数据集，为[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和计算机视觉研究社区提供了宝贵的资源。有关 ImageNet 数据集及其创建者的更多信息，请访问 [ImageNet 网站](https://www.image-net.org/)。

## 常见问题

### ImageNet 数据集是什么？它在计算机视觉中如何使用？

[ImageNet 数据集](https://www.image-net.org/)是一个大规模数据库，包含超过 1400 万张使用 WordNet 同义词集分类的高分辨率图像。它广泛用于视觉目标识别研究，包括图像分类和目标检测。数据集的标注和庞大数量为训练深度学习模型提供了丰富的资源。值得注意的是，像 AlexNet、VGG 和 ResNet 这样的模型都是使用 ImageNet 进行训练和基准测试的，展示了它在推动计算机视觉发展中的作用。

### 如何使用预训练的 YOLO 模型在 ImageNet 数据集上进行图像分类？

要使用预训练的 Ultralytics YOLO 模型在 ImageNet 数据集上进行图像分类，请按照以下步骤操作：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="imagenet", epochs=100, imgsz=224)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=imagenet model=yolo26n-cls.pt epochs=100 imgsz=224
        ```

有关更深入的训练指导，请参阅我们的[训练页面](../../modes/train.md)。

### 为什么应该为我的 ImageNet 数据集项目使用 Ultralytics YOLO26 预训练模型？

Ultralytics YOLO26 预训练模型在各种计算机视觉任务中提供了最先进的速度和[准确率](https://www.ultralytics.com/glossary/accuracy)性能。例如，YOLO26n-cls 模型的 top-1 准确率为 70.0%，top-5 准确率为 89.4%，针对实时应用进行了优化。预训练模型减少了从头开始训练所需的计算资源，并加速了开发周期。在 [ImageNet 预训练模型部分](#imagenet-预训练模型)了解更多关于 YOLO26 模型性能指标的信息。

### ImageNet 数据集是如何结构的？为什么它很重要？

ImageNet 数据集使用 WordNet 层级结构组织，层级结构中的每个节点代表一个由同义词集（一组同义词的集合）描述的类别。这种结构允许详细的标注，使其非常适合训练模型识别各种对象。ImageNet 的多样性和标注丰富性使其成为开发鲁棒且可泛化的深度学习模型的宝贵数据集。关于这种组织的更多信息，请参见[数据集结构](#数据集结构)部分。

### ImageNet 大规模视觉识别挑战赛 (ILSVRC) 在计算机视觉中扮演什么角色？

一年一度的 [ImageNet 大规模视觉识别挑战赛 (ILSVRC)](https://image-net.org/challenges/LSVRC/) 通过提供一个竞争性平台，在大规模标准化数据集上评估算法，在推动计算机视觉进步方面发挥了关键作用。它提供了标准化的评估指标，促进了图像分类、目标检测和[图像分割](https://www.ultralytics.com/glossary/image-segmentation)等领域的创新和发展。该挑战赛不断推动深度学习和计算机视觉技术的可能性边界。