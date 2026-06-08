---
comments: true
description: 了解成功启动计算机视觉项目的关键步骤，从目标定义到模型部署与维护。
keywords: 计算机视觉, 人工智能, 目标检测, 图像分类, 实例分割, 数据标注, 模型训练, 模型评估, 模型部署
---

# 理解计算机视觉项目的关键步骤

## 引言

计算机视觉是[人工智能](https://www.ultralytics.com/glossary/artificial-intelligence-ai)（AI）的一个子领域，它帮助计算机像人类一样观察和理解世界。它处理和分析图像或视频，以提取信息、识别模式并基于这些数据做出决策。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/CfbHwPG01cE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何开展<a href="https://www.ultralytics.com/glossary/computer-vision-cv">计算机视觉</a>项目 | 分步指南
</p>

计算机视觉技术（如[目标检测](../tasks/detect.md)、[图像分类](../tasks/classify.md)和[实例分割](../tasks/segment.md)）可应用于各行各业，从[自动驾驶](https://www.ultralytics.com/solutions/ai-in-automotive)到[医学影像](https://www.ultralytics.com/solutions/ai-in-healthcare)，以获取有价值的洞察。

亲手实践自己的计算机视觉项目是理解和学习计算机视觉的绝佳方式。然而，一个计算机视觉项目可能包含许多步骤，起初可能会让人感到困惑。在本指南结束时，你将熟悉计算机视觉项目涉及的各个步骤。我们将从头到尾逐一讲解项目的每个部分，并解释为什么每个部分都至关重要。

## 计算机视觉项目概览

在深入讨论计算机视觉项目每个步骤的细节之前，让我们先看看整体流程。如果你今天开始一个计算机视觉项目，你将经历以下步骤：

- 你的首要任务是理解项目需求。
- 然后，你需要收集并准确标注用于训练模型的图像。
- 接下来，你需要清洗数据并应用数据增强技术，为模型训练做好准备。
- 模型训练完成后，你需要全面测试和评估模型，确保它在不同条件下都能稳定运行。
- 最后，你需要将模型部署到实际环境中，并根据新的洞察和反馈进行更新。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/five-stages-of-ml-development-lifecycle.avif" alt="计算机视觉项目步骤概览">
</p>

现在我们已经了解了整体流程，让我们直接深入各个步骤，推动你的项目向前发展。

## 第一步：定义项目目标

任何计算机视觉项目的第一步都是明确定义你要解决的问题。了解最终目标有助于你开始构建解决方案。对于计算机视觉尤其如此，因为你的项目目标将直接影响你需要关注的计算机视觉任务。

以下是一些项目目标以及可用于实现这些目标的计算机视觉任务示例：

- **目标：** 开发一个系统，能够监控和管理高速公路上不同类型车辆的通行，从而改善交通管理和安全性。
    - **计算机视觉任务：** 目标检测非常适合交通监控，因为它能高效地定位和识别多辆车辆。它的计算量比图像分割小（图像分割对此任务提供了不必要的细节），确保更快的实时分析。

- **目标：** 开发一个工具，帮助放射科医生在医学影像扫描中提供精确的、像素级的肿瘤轮廓。
    - **计算机视觉任务：** 图像分割适用于医学影像，因为它提供准确而详细的肿瘤边界，这对于评估大小、形状和制定治疗方案至关重要。

- **目标：** 创建一个数字系统，对各类文档（如发票、收据、法律文件）进行分类，以提高组织效率和文档检索能力。
    - **计算机视觉任务：** [图像分类](https://www.ultralytics.com/glossary/image-classification)在此场景下非常理想，因为它一次处理一份文档，无需考虑文档在图像中的位置。这种方法简化并加速了分类过程。

### 第一步（续）：选择正确的模型和训练方法

在理解项目目标和合适的计算机视觉任务之后，定义项目目标的一个关键部分是[选择合适的模型](../models/index.md)和训练方法。

根据目标的不同，你可能选择先选择模型，也可能在第二步看到能收集到什么数据后再选择。例如，如果你的项目高度依赖特定类型数据的可用性，那么先收集和分析数据再选择模型可能更加务实。另一方面，如果你对模型需求有清晰的了解，可以先选择模型，然后再收集符合这些规格的数据。

选择从头开始训练还是使用[迁移学习](https://www.ultralytics.com/glossary/transfer-learning)会影响你准备数据的方式。从头开始训练需要一个多样化的数据集，以便从零开始构建模型的理解。而迁移学习则允许你使用预训练模型，并用一个更小、更专用的数据集对其进行适配。此外，选择特定的模型进行训练还将决定你需要如何准备数据，例如根据模型的具体要求调整图像大小或添加标注。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/training-from-scratch-vs-transfer-learning.avif" alt="从头训练 vs 使用迁移学习">
</p>

注意：在选择模型时，请考虑其[部署](./model-deployment-options.md)方式，以确保兼容性和性能。例如，轻量级模型由于在资源受限设备上的效率，非常适合[边缘计算](https://www.ultralytics.com/glossary/edge-computing)。要了解更多关于定义项目的关键要点，请阅读我们关于定义项目目标和选择正确模型的[指南](./defining-project-goals.md)。

在进入计算机视觉项目的实际操作之前，对这些细节有清晰的理解非常重要。在进入第二步之前，请再次确认你已经考虑了以下内容：

- 明确定义你要解决的问题。
- 确定项目的最终目标。
- 识别所需的特定计算机视觉任务（例如，目标检测、图像分类、图像分割）。
- 决定是从头开始训练模型还是使用迁移学习。
- 选择适合你的任务和部署需求的模型。

## 第二步：数据收集与数据标注

计算机视觉模型的质量取决于数据集的质量。你可以从互联网上收集图像、自己拍摄照片或使用现有的数据集。以下是一些下载高质量数据集的优质资源：[Google Dataset Search Engine](https://datasetsearch.research.google.com/)、[UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) 和 [Kaggle Datasets](https://www.kaggle.com/datasets)。

像 Ultralytics 这样的库为[各种数据集提供了内置支持](../datasets/index.md)，使得使用高质量数据入门更加容易。这些库通常包含无缝使用流行数据集的工具，可以节省你在项目初期阶段的大量时间和精力。

但是，如果你选择收集图像或自己拍摄照片，你还需要对数据进行标注。[数据标注](https://www.ultralytics.com/annotate)是对数据进行标记以向模型传递知识的过程。你需要进行的数据标注类型取决于你的具体计算机视觉技术。以下是一些示例：

- **图像分类：** 你需要将整个图像标记为单个类别。
- **[目标检测](https://www.ultralytics.com/glossary/object-detection)：** 你需要在图像中每个对象周围绘制边界框，并为每个框添加标签。
- **[图像分割](https://www.ultralytics.com/glossary/image-segmentation)：** 你需要根据每个像素所属的对象对其进行标记，从而创建详细的对象边界。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/different-types-of-image-annotation.avif" alt="边界框、多边形和关键点标注">
</p>

[数据收集与标注](./data-collection-and-annotation.md)可能是一项耗时的手动工作。标注工具可以帮助简化这一过程。以下是一些有用的开源标注工具：[Label Studio](https://github.com/HumanSignal/label-studio)、[CVAT](https://github.com/cvat-ai/cvat) 和 [Labelme](https://github.com/wkentaro/labelme)。

## 第三步：数据增强与数据集划分

在收集和标注图像数据之后，在进行[数据增强](https://www.ultralytics.com/glossary/data-augmentation)之前，首先需要将数据集划分为训练集、验证集和测试集。在增强前划分数据集对于在原始、未修改的数据上测试和验证模型至关重要。这有助于准确评估模型对新数据、未见数据的泛化能力。

以下是数据划分的方式：

- **训练集：** 这是数据中最大的一部分，通常占总量的 70-80%，用于训练模型。
- **验证集：** 通常约占数据的 10-15%；此集合用于在训练过程中调整超参数和验证模型，有助于防止[过拟合](https://www.ultralytics.com/glossary/overfitting)。
- **测试集：** 剩余 10-15% 的数据作为测试集留出，用于在训练完成后评估模型在未见数据上的性能。

在划分数据之后，你可以通过应用旋转、缩放和翻转等变换来进行数据增强，以人为地扩大数据集规模。数据增强使模型对变化更具鲁棒性，并提升其在未见图像上的性能。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/examples-of-data-augmentations.avif" alt="数据增强示例">
</p>

像 [OpenCV](https://www.ultralytics.com/glossary/opencv)、[Albumentations](../integrations/albumentations.md) 和 [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) 这样的库提供了灵活的增强函数供你使用。此外，像 Ultralytics 这样的库在其模型训练函数中直接提供了[内置的增强设置](../modes/train.md)，简化了这一过程。

为了更好地理解你的数据，你可以使用 [Matplotlib](https://matplotlib.org/) 或 [Seaborn](https://seaborn.pydata.org/) 等工具来可视化图像并分析其分布和特征。数据可视化有助于识别模式、异常情况以及增强技术的有效性。[Ultralytics Platform](https://platform.ultralytics.com/) 的 `Charts` 标签页无需编写代码即可呈现许多此类洞察，它会为每个上传的数据集自动生成划分分布、类别计数、图像维度直方图和标注位置热力图。

通过正确[理解、划分和增强数据](./preprocessing_annotated_data.md)，你可以开发出一个经过良好训练、验证和测试的模型，并在实际应用中表现优异。

## 第四步：模型训练

一旦数据集准备好进行训练，你就可以专注于设置必要的环境、管理数据集以及训练模型。

首先，你需要确保环境配置正确。通常包括以下内容：

- 安装必要的库和框架，如 TensorFlow、[PyTorch](https://www.ultralytics.com/glossary/pytorch) 或 [Ultralytics](../quickstart.md)。
- 如果你使用 GPU，安装 CUDA 和 cuDNN 等库将有助于启用 GPU 加速并加快训练过程。

然后，你可以将训练集和验证集加载到你的环境中。通过调整大小、格式转换或增强来标准化和预处理数据。选定模型后，配置网络层并指定超参数。通过设置[损失函数](https://www.ultralytics.com/glossary/loss-function)、优化器和性能指标来编译模型。

像 Ultralytics 这样的库简化了训练过程。你可以用最少的代码将数据输入模型来[开始训练](../modes/train.md)。这些库自动处理权重调整、[反向传播](https://www.ultralytics.com/glossary/backpropagation)和验证。它们还提供了轻松监控进度和调整超参数的工具。训练完成后，只需几条命令即可保存模型及其权重。

需要牢记的是，适当的数据集管理对于高效训练至关重要。使用版本控制来管理数据集，以跟踪更改并确保可复现性。像 [DVC（数据版本控制）](../integrations/dvc.md)这样的工具可以帮助管理大型数据集。

## 第五步：模型评估与模型微调

使用各种指标评估模型的性能，并对其进行优化以提高[准确率](https://www.ultralytics.com/glossary/accuracy)非常重要。[评估](../modes/val.md)有助于识别模型表现优异的领域以及可能需要改进的领域。[微调](https://www.ultralytics.com/glossary/fine-tuning)确保模型优化到最佳性能。

- **[性能指标](./yolo-performance-metrics.md)：** 使用准确率、[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 F1 分数等指标来评估模型性能。这些指标可以洞察模型预测的质量。
- **[超参数调优](./hyperparameter-tuning.md)：** 调整超参数以优化模型性能。网格搜索或随机搜索等技术可以帮助找到最佳的超参数值。
- **微调：** 对模型架构或训练过程进行小幅调整以提升性能。这可能涉及调整[学习率](https://www.ultralytics.com/glossary/learning-rate)、[批次大小](https://www.ultralytics.com/glossary/batch-size)或其他模型参数。

要更深入地了解模型评估和微调技术，请查看我们的[模型评估洞察指南](./model-evaluation-insights.md)。

## 第六步：模型测试

在这一步中，你可以确保模型在完全未见过的数据上表现良好，确认其已准备好部署。模型测试与模型评估的区别在于，它侧重于验证最终模型的性能，而不是迭代改进。

全面测试和调试可能出现的常见问题非常重要。在与训练或验证时使用的数据完全独立的测试数据集上测试你的模型。该数据集应代表真实场景，以确保模型的性能一致且可靠。

同时，解决过拟合、[欠拟合](https://www.ultralytics.com/glossary/underfitting)和数据泄露等常见问题。使用[交叉验证](https://www.ultralytics.com/glossary/cross-validation)和[异常检测](https://www.ultralytics.com/glossary/anomaly-detection)等技术来识别和修复这些问题。有关全面的测试策略，请参考我们的[模型测试指南](./model-testing.md)。

## 第七步：模型部署

当模型经过全面测试后，就可以进行部署了。[模型部署](https://www.ultralytics.com/glossary/model-deployment)是指将模型投入生产环境使用。以下是部署计算机视觉模型的步骤：

- **设置环境：** 为你选择的部署选项配置必要的基础设施，无论是基于云（AWS、Google Cloud、Azure）还是基于边缘（本地设备、物联网）。
- **[导出模型](../modes/export.md)：** 将模型导出为适当的格式（例如，YOLO26 的 ONNX、TensorRT、CoreML），以确保与部署平台兼容。
- **部署模型：** 通过设置 API 或端点并将模型与应用程序集成来部署模型。
- **确保可扩展性：** 实施负载均衡器、自动扩展组和监控工具，以管理资源并处理不断增长的数据和用户请求。

有关部署策略和最佳实践的更详细指导，请查看我们的[模型部署实践指南](./model-deployment-practices.md)。[Ultralytics Platform](https://platform.ultralytics.com) 还提供托管的[部署端点](../platform/deploy/endpoints.md)，在 43 个全球区域实现自动扩展，自动处理基础设施设置。

## 第八步：监控、维护与文档

模型部署后，持续监控其性能、维护以处理任何问题，并记录整个过程以便将来参考和改进，这些都非常重要。

监控工具可以帮助你跟踪关键性能指标（KPI）并检测异常或准确率下降。通过监控模型，你可以了解模型漂移——即模型性能由于输入数据的变化而随时间下降。定期使用更新的数据重新训练模型，以保持准确性和相关性。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/model-monitoring-maintenance-loop.avif" alt="模型监控与维护生命周期">
</p>

除了监控和维护，文档同样关键。全面记录整个过程，包括模型架构、训练过程、超参数、数据预处理步骤，以及部署和维护期间所做的任何更改。良好的文档可确保可复现性，并使未来的更新或问题排查更加容易。通过有效[监控、维护和记录你的模型](./model-monitoring-and-maintenance.md)，你可以确保它在整个生命周期内保持准确、可靠且易于管理。

## 参与社区

与计算机视觉爱好者社区建立联系，可以帮助你自信地应对在计算机视觉项目中遇到的任何问题。以下是一些有效学习、解决问题和建立人脉的方式。

### 社区资源

- **GitHub Issues：** 查看 [YOLO26 GitHub 仓库](https://github.com/ultralytics/ultralytics/issues)，使用 Issues 选项卡提问、报告错误和建议新功能。活跃的社区和维护者随时准备帮助你解决具体问题。
- **Ultralytics Discord 服务器：** 加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)，与其他用户和开发者互动、获得支持并分享见解。

### 官方文档

- **Ultralytics YOLO26 文档：** 浏览[官方 YOLO26 文档](./index.md)，获取关于不同计算机视觉任务和项目的详细指南与实用技巧。

使用这些资源将帮助你克服挑战，并随时了解计算机视觉社区的最新趋势和最佳实践。

## 后续步骤

着手一个计算机视觉项目既令人兴奋又富有回报。通过遵循本指南中的步骤，你可以为成功奠定坚实的基础。每个步骤对于开发一个满足目标并在真实场景中良好运行的解决方案都至关重要。随着经验的积累，你将发现更高级的技术和工具来改进你的项目。

## 常见问题

### 如何为我的项目选择合适的计算机视觉任务？

选择合适的计算机视觉任务取决于你项目的最终目标。例如，如果你想监控交通，**目标检测**是合适的，因为它可以实时定位和识别多种车辆类型。对于医学影像，**图像分割**非常适合提供肿瘤的详细边界，有助于诊断和制定治疗方案。了解更多关于具体任务的信息，如[目标检测](../tasks/detect.md)、[图像分类](../tasks/classify.md)和[实例分割](../tasks/segment.md)。

### 为什么数据标注在计算机视觉项目中至关重要？

数据标注对于教会模型识别模式至关重要。标注类型因任务而异：

- **图像分类：** 整个图像标记为单个类别。
- **目标检测：** 在对象周围绘制边界框。
- **图像分割：** 每个像素根据其所属对象进行标记。

像 [Label Studio](https://github.com/HumanSignal/label-studio)、[CVAT](https://github.com/cvat-ai/cvat) 和 [Labelme](https://github.com/wkentaro/labelme) 这样的工具可以帮助完成这一过程。更多详情，请参考我们的[数据收集与标注指南](./data-collection-and-annotation.md)。

### 我应该遵循哪些步骤来有效地增强和划分数据集？

在增强前划分数据集有助于在原始、未修改的数据上验证模型性能。遵循以下步骤：

- **训练集：** 占数据的 70-80%。
- **验证集：** 10-15% 用于[超参数调优](https://www.ultralytics.com/glossary/hyperparameter-tuning)。
- **测试集：** 剩余 10-15% 用于最终评估。

划分后，应用旋转、缩放和翻转等数据增强技术来增加数据集的多样性。像 [Albumentations](../integrations/albumentations.md) 和 OpenCV 这样的库可以提供帮助。Ultralytics 也提供了[内置的增强设置](../modes/train.md)以方便使用。

### 如何导出训练好的计算机视觉模型用于部署？

导出模型可确保与不同部署平台的兼容性。Ultralytics 提供多种格式，包括 [ONNX](../integrations/onnx.md)、[TensorRT](../integrations/tensorrt.md) 和 [CoreML](../integrations/coreml.md)。要导出你的 YOLO26 模型，请遵循以下指南：

- 使用带有所需格式参数的 `export` 函数。
- 确保导出的模型符合你部署环境（例如边缘设备、云端）的规格。

更多信息，请查看[模型导出指南](../modes/export.md)。

### 监控和维护已部署的计算机视觉模型的最佳实践是什么？

持续监控和维护对于模型的长期成功至关重要。实施工具来跟踪关键性能指标（KPI）和检测异常。定期使用更新的数据重新训练模型以应对模型漂移。记录整个过程，包括模型架构、超参数和更改，以确保可复现性和便于未来更新。更多内容请参阅我们的[监控与维护指南](./model-monitoring-and-maintenance.md)。
