---
comments: true
description: 了解监控、维护和记录计算机视觉模型的关键实践，以确保准确性、发现异常并缓解数据漂移。
keywords: 计算机视觉模型, AI 模型监控, 数据漂移检测, AI 异常检测, 模型维护
---

# 部署后维护你的计算机视觉模型

## 简介

如果你读到了这里，我们可以假设你已经完成了[计算机视觉项目中的许多步骤](./steps-of-a-cv-project.md)：从[收集需求](./defining-project-goals.md)、[标注数据](./data-collection-and-annotation.md)、[训练模型](./model-training-tips.md)，到最终[部署](./model-deployment-practices.md)模型。你的应用已经在上线运行，但项目并没有就此结束。计算机视觉项目中最重要的部分，是确保你的模型能够持续满足[项目目标](./defining-project-goals.md)，而这正是监控、维护和记录计算机视觉模型的用武之地。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/zCupPHqSLTI"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>部署后如何维护计算机视觉模型 | 数据漂移检测
</p>

在本指南中，我们将深入探讨如何在部署后维护你的计算机视觉模型。我们将探索模型监控如何帮助你及早发现问题，如何保持模型准确且与时俱进，以及为什么文档记录对故障排除至关重要。

## 模型监控是关键

密切关注已部署的计算机视觉模型至关重要。缺乏适当的监控，模型可能会失去准确性。一个常见的问题是数据分布偏移或[数据漂移](https://www.ultralytics.com/glossary/data-drift)，即模型遇到的数据与其训练时的数据相比发生了变化。当模型必须对其无法识别的数据进行预测时，就会导致误解和性能下降。异常值（即异常的数据点）也可能影响模型的准确性。

定期模型监控帮助开发者跟踪[模型性能](./model-evaluation-insights.md)、发现异常，并迅速解决数据漂移等问题。它还有助于通过指示何时需要更新来管理资源，避免代价高昂的大修，并保持模型的相关性。

### 模型监控的最佳实践

以下是监控生产环境中的计算机视觉模型时应牢记的一些最佳实践：

- **定期跟踪性能**：持续监控模型性能，以检测随时间发生的变化。
- **双重检查数据质量**：检查数据中的缺失值或异常。
- **使用多样化的数据源**：从多个数据源监控数据，以全面了解模型性能。
- **组合多种监控技术**：结合使用漂移检测算法和基于规则的方法，以识别各种问题。
- **监控输入和输出**：同时关注模型处理的数据和产生的结果，确保一切正常运行。
- **设置告警**：对异常行为（如性能下降）实施告警，以便能够快速采取纠正措施。

### AI 模型监控工具

你可以使用自动化监控工具，使部署后监控模型变得更加容易。许多工具提供实时洞察和告警能力。以下是一些可以协同工作的开源模型监控工具示例：

- **[Prometheus](https://prometheus.io/)**：Prometheus 是一个开源监控工具，用于收集和存储指标数据，以进行详细的性能跟踪。它能够轻松与 Kubernetes 和 Docker 集成，按设定间隔采集数据并将其存储在时序数据库中。Prometheus 还可以抓取 HTTP 端点来收集实时指标。收集到的数据可以使用 PromQL 语言进行查询。
- **[Grafana](https://grafana.com/)**：Grafana 是一个开源的[数据可视化](https://www.ultralytics.com/glossary/data-visualization)和监控工具，允许你对指标进行查询、可视化、告警和理解，无论指标存储在哪里。它与 Prometheus 配合良好，并提供高级的数据可视化功能。你可以创建自定义仪表盘来展示计算机视觉模型的重要指标，如推理延迟、错误率和资源使用情况。Grafana 将收集到的数据转化为易于阅读的仪表盘，包括折线图、热力图和直方图。它还支持告警，可通过 Slack 等渠道发送告警，及时通知团队任何问题。
- **[Evidently AI](https://www.evidentlyai.com/)**：Evidently AI 是一个开源工具，专为监控和调试生产环境中的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型而设计。它从 pandas DataFrame 生成交互式报告，帮助分析机器学习模型。Evidently AI 可以检测数据漂移、模型性能退化以及部署模型中可能出现的其他问题。

上述三个工具——Evidently AI、Prometheus 和 Grafana——可以无缝协作，构成一个完全开源、可用于生产环境的 ML 监控解决方案。Evidently AI 用于收集和计算指标，Prometheus 存储这些指标，Grafana 展示指标并设置告警。虽然还有许多其他工具可用，但这一组合方案是一个令人兴奋的开源选择，为[模型监控](https://www.ultralytics.com/glossary/model-monitoring)和模型维护提供了强大的能力。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/evidently-prometheus-grafana-monitoring-tools.avif" alt="开源模型监控工具概览">
</p>

### 异常检测与告警系统

异常是指任何与预期结果有较大偏差的数据点或模式。对于[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)模型而言，异常可能是与模型训练数据差异很大的图像。这些意外图像可能是数据分布变化、异常值或可能降低模型性能的行为等问题信号。建立检测这些异常的告警系统是模型监控的重要组成部分。

通过为关键指标设置标准性能水平和阈值，你可以及早发现问题。当性能超出这些阈值时，告警会被触发，促使快速修复。随着数据的变化，用新数据定期更新和重新训练模型可以保持模型的相关性和准确性。

#### 配置阈值和告警时的注意事项

在设置告警系统时，请牢记以下最佳实践：

- **标准化告警**：对所有告警使用一致的工具和格式，如邮件或 Slack 等即时通讯工具。标准化使你能更快地理解和响应告警。
- **包含预期行为**：告警信息应清晰说明出了什么问题、预期是什么以及评估的时间范围。这有助于你判断告警的紧迫性和上下文。
- **可配置的告警**：使告警易于配置以适应不断变化的情况。允许自己编辑阈值、暂停、禁用或确认告警。

### 数据漂移检测

数据漂移检测是一个概念，帮助识别输入数据的统计属性随时间变化的情况，这种变化会降低模型性能。在你决定重新训练或调整模型之前，这项技术有助于发现问题所在。数据漂移关注的是整体数据景观随时间的变化，而[异常检测](https://www.ultralytics.com/glossary/anomaly-detection)则专注于识别可能需要立即关注的罕见或意外数据点。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/data-drift-detection-overview.avif" alt="数据漂移检测监控流水线">
</p>

以下是检测数据漂移的几种方法：

**持续监控**：定期监控模型的输入数据和输出，寻找漂移的迹象。跟踪关键指标并将其与历史数据进行比较，以识别显著变化。

**统计技术**：使用 Kolmogorov-Smirnov 检验或群体稳定性指数（PSI）等方法来检测数据分布的变化。这些检验将新数据的分布与[训练数据](https://www.ultralytics.com/glossary/training-data)进行比较，以识别显著差异。

**特征漂移**：监控单个特征的漂移。有时，整体数据分布可能保持稳定，但个别特征可能发生漂移。识别哪些特征在漂移有助于微调重新训练过程。

## 模型维护

模型维护对于保持计算机视觉模型随时间推移仍然准确和相关至关重要。模型维护包括定期更新和重新训练模型、解决数据漂移问题，以及确保模型在数据和环境变化时保持相关性。你可能想知道模型维护与模型监控有何不同。监控是关于实时观察模型性能以及早发现问题。而维护则是关于修复这些问题。

### 定期更新与重新训练

模型部署后，在监控过程中你可能会注意到数据模式或性能的变化，这表明模型发生了漂移。定期更新和重新训练成为模型维护中必不可少的部分，以确保模型能够处理新的模式和场景。根据数据变化的方式，你可以使用几种技术。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/computer-vision-model-drift-overview.avif" alt="计算机视觉模型漂移原因">
</p>

例如，如果数据随时间逐渐变化，增量学习是一个不错的方法。增量学习涉及用新数据更新模型，而不是完全从头重新训练，从而节省计算资源和时间。然而，如果数据发生了剧烈变化，周期性全面重新训练可能是更好的选择，以确保模型不会在新数据上[过拟合](https://www.ultralytics.com/glossary/overfitting)，同时丧失对旧模式的跟踪。

无论采用哪种方法，更新后进行验证和测试都是必须的。重要的是在独立的[测试数据集](./model-testing.md)上验证模型，检查性能是提升还是下降了。

### 决定何时重新训练模型

重新训练计算机视觉模型的频率取决于数据变化和模型性能。每当你观察到显著的性能下降或检测到数据漂移时，就重新训练模型。定期评估可以通过用新数据测试模型来帮助确定合适的重新训练计划。监控性能指标和数据模式可以帮助你判断模型是否需要更频繁的更新以保持[准确性](https://www.ultralytics.com/glossary/accuracy)。

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/when-to-retrain-overview.avif" alt="何时重新训练 ML 模型流程图">
</p>

## 文档记录

记录计算机视觉项目使其更容易理解、复现和协作。良好的文档涵盖模型架构、[超参数](https://www.ultralytics.com/glossary/hyperparameter-tuning)、数据集、评估指标等内容。它提供透明度，帮助团队成员和利益相关者了解做了什么以及为什么这样做。文档还通过提供过去决策和方法的清晰参考，有助于故障排除、维护和未来的增强。

### 需要记录的关键要素

以下是项目文档中应包含的一些关键要素：

- **[项目概述](./steps-of-a-cv-project.md)**：提供项目的高层摘要，包括问题陈述、解决方案方法、预期结果和项目范围。说明计算机视觉在解决问题中的作用，并概述各阶段和交付物。
- **模型架构**：详细说明模型的结构和设计，包括其组件、层和连接。解释所选的超参数以及做出这些选择的理由。
- **[数据准备](./data-collection-and-annotation.md)**：描述数据来源、类型、格式、大小和预处理步骤。讨论数据质量、可靠性以及在训练模型之前应用的任何转换。
- **[训练过程](./model-training-tips.md)**：记录训练过程，包括使用的数据集、训练参数和[损失函数](https://www.ultralytics.com/glossary/loss-function)。说明模型是如何训练的以及训练过程中遇到的任何挑战。
- **[评估指标](./model-evaluation-insights.md)**：指定用于评估模型性能的指标，如准确率、[精确率](https://www.ultralytics.com/glossary/precision)、[召回率](https://www.ultralytics.com/glossary/recall)和 [F1 分数](https://www.ultralytics.com/glossary/f1-score)。包括性能结果和对这些指标的分析。
- **[部署步骤](./model-deployment-options.md)**：概述部署模型所采取的步骤，包括使用的工具和平台、部署配置以及任何特定的挑战或考虑因素。
- **监控与维护流程**：提供部署后监控模型性能的详细计划。包括检测和解决数据漂移及模型漂移的方法，并描述定期更新和重新训练的过程。

### 文档工具

在记录 AI 项目方面有很多选择，开源工具尤为流行。其中两种是 [Jupyter Notebooks](https://docs.ultralytics.com/integrations/jupyterlab) 和 MkDocs。Jupyter Notebooks 允许你创建包含嵌入式代码、可视化和文本的交互式文档，非常适合分享实验和分析。MkDocs 是一个静态站点生成器，易于设置和部署，非常适合在线创建和托管项目文档。

## 与社区建立联系

加入计算机视觉爱好者社区可以帮助你更快地解决问题和学习。以下是一些建立联系、获取支持和分享想法的方式。

### 社区资源

- **GitHub Issues：** 查看 [YOLO26 GitHub 仓库](https://github.com/ultralytics/ultralytics/issues)，使用 Issues 标签页来提问、报告 Bug 和建议新功能。社区和维护者非常活跃且乐于助人。
- **Ultralytics Discord 服务器：** 加入 [Ultralytics Discord 服务器](https://discord.com/invite/ultralytics)，与其他用户和开发者交流，获取支持并分享经验。

### 官方文档

- **Ultralytics YOLO26 文档：** 访问[官方 YOLO26 文档](./index.md)，获取关于各种计算机视觉项目的详细指南和实用技巧。

利用这些资源将帮助你克服挑战，并紧跟计算机视觉社区的最新趋势和实践。

## 关键要点

我们介绍了监控、维护和记录计算机视觉模型的关键技巧。定期更新和重新训练帮助模型适应新的数据模式。检测和修复数据漂移有助于保持模型准确。持续监控能及早发现问题，而良好的文档使协作和未来更新更加容易。遵循这些步骤将帮助你的计算机视觉项目长期保持成功和有效。

## 常见问题

### 如何监控已部署的计算机视觉模型的性能？

监控已部署的计算机视觉模型的性能对于确保其随时间推移的准确性和可靠性至关重要。你可以使用 [Prometheus](https://prometheus.io/)、[Grafana](https://grafana.com/) 和 [Evidently AI](https://www.evidentlyai.com/) 等工具来跟踪关键指标、检测异常和识别数据漂移。定期监控输入和输出，为异常行为设置告警，并使用多样化的数据源来全面了解模型性能。更多详情，请参阅我们的[模型监控是关键](#模型监控是关键)章节。

### 部署后维护计算机视觉模型的最佳实践有哪些？

维护计算机视觉模型包括定期更新、重新训练和监控，以确保持续的准确性和相关性。最佳实践包括：

- **持续监控**：定期跟踪性能指标和数据质量。
- **数据漂移检测**：使用统计技术识别数据分布的变化。
- **定期更新和重新训练**：根据数据变化实施增量学习或周期性全面重新训练。
- **文档记录**：维护模型架构、训练过程和评估指标的详细文档。更多见解，请访问我们的[模型维护](#模型维护)章节。

### 为什么数据漂移检测对 AI 模型很重要？

数据漂移检测至关重要，因为它有助于识别输入数据的统计属性随时间变化的情况，这种变化会降低模型性能。持续监控、统计检验（如 Kolmogorov-Smirnov 检验）和特征漂移分析等技术可以帮助及早发现问题。解决数据漂移可以确保你的模型在不断变化的环境中保持准确和相关。在[数据漂移检测](#数据漂移检测)章节了解更多。

### 我可以使用哪些工具进行计算机视觉模型的异常检测？

对于计算机视觉模型的异常检测，[Prometheus](https://prometheus.io/)、[Grafana](https://grafana.com/) 和 [Evidently AI](https://www.evidentlyai.com/) 等工具非常有效。这些工具可以帮助你建立告警系统，检测与预期行为有偏差的异常数据点或模式。可配置的告警和标准化的消息可以帮助你快速响应潜在问题。在[异常检测与告警系统](#异常检测与告警系统)章节了解更多。

### 如何有效地记录我的计算机视觉项目？

有效记录计算机视觉项目应包括：

- **项目概述**：高层摘要、问题陈述和解决方案方法。
- **模型架构**：模型结构、组件和超参数的详细信息。
- **数据准备**：数据来源、预处理步骤和转换的信息。
- **训练过程**：训练过程的描述、使用的数据集和遇到的挑战。
- **评估指标**：用于性能评估和分析的指标。
- **部署步骤**：[模型部署](https://www.ultralytics.com/glossary/model-deployment)的步骤以及任何特定的挑战。
- **监控与维护流程**：持续监控和维护的计划。更全面的指南，请参阅我们的[文档记录](#文档记录)章节。