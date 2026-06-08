---
comments: true
description: 了解如何在 AzureML 上运行 YOLO26。终端和 Notebook 的快速入门指南，利用 Azure 云计算进行高效模型训练。
keywords: YOLO26, AzureML, 机器学习, 云计算, 快速入门, 终端, Notebook, 模型训练, Python SDK, AI, Ultralytics
---

# YOLO26 🚀 在 AzureML 上

## 什么是 Azure？

[Azure](https://azure.microsoft.com/) 是微软的[云计算](https://www.ultralytics.com/glossary/cloud-computing)平台，旨在帮助组织将工作负载从本地数据中心迁移到云端。Azure 提供全方位的云服务，包括计算、数据库、分析、[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)和网络等，用户可以从这些服务中灵活选择，在公有云上开发、扩展新应用或运行现有应用。

## 什么是 Azure 机器学习 (AzureML)？

Azure 机器学习（通常称为 AzureML）是一项完全托管的云服务，使数据科学家和开发人员能够高效地将预测分析嵌入到应用程序中，帮助组织利用海量数据集，将云计算的优势全面引入机器学习领域。AzureML 提供多种服务和功能，旨在让机器学习变得易于访问、使用简单且具备可扩展性。它提供了自动化机器学习、拖放式模型训练等功能，以及强大的 Python SDK，帮助开发者充分发挥机器学习模型的潜力。

## AzureML 如何惠及 YOLO 用户？

对于 YOLO (You Only Look Once) 的用户来说，AzureML 提供了一个强大、可扩展且高效的平台，既可以训练也可以部署机器学习模型。无论是快速运行原型还是扩展到处理更大规模的数据，AzureML 灵活且用户友好的环境都提供了各种工具和服务来满足需求。你可以利用 AzureML 实现以下目标：

- 轻松管理用于训练的大规模数据集和计算资源。
- 利用内置工具进行数据预处理、特征选择和模型训练。
- 借助 MLOps（机器学习运维）能力更高效地协作，包括但不限于模型和数据的监控、审计和版本管理。

在后续章节中，你将找到一份快速入门指南，详细介绍如何使用 AzureML 运行 YOLO26 目标检测模型，无论是在计算终端还是 Notebook 中。

## 前提条件

在开始之前，请确保你已拥有 AzureML 工作区的访问权限。如果没有，可以按照 Azure 官方文档创建一个新的 [AzureML 工作区](https://learn.microsoft.com/azure/machine-learning/concept-workspace?view=azureml-api-2)。该工作区是管理所有 AzureML 资源的集中场所。

## 创建计算实例

在 AzureML 工作区中，选择 Compute > Compute instances > New，选择符合资源需求的实例。

<p align="center">
  <img width="1280" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/create-compute-arrow.avif" alt="创建 Azure 计算实例">
</p>

## 从终端快速入门

启动计算实例并打开终端：

<p align="center">
  <img width="480" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/open-terminal.avif" alt="打开终端">
</p>

### 创建虚拟环境

使用你偏好的 Python 版本创建 conda 虚拟环境并安装 pip。Python 3.13.1 目前在 AzureML 中存在依赖问题，请使用 Python 3.12。

```bash
conda create --name yolo26env -y python=3.12
conda activate yolo26env
conda install pip -y
```

安装所需依赖：

```bash
cd ultralytics
pip install -r requirements.txt
pip install ultralytics
pip install onnx
```

### 执行 YOLO26 任务

预测：

```bash
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
```

训练一个检测模型，训练 10 个 [epoch](https://www.ultralytics.com/glossary/epoch)，初始学习率 0.01：

```bash
yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
```

更多 [Ultralytics CLI 使用说明请点击此处](../quickstart.md#use-ultralytics-with-cli)。

## 从 Notebook 快速入门

### 创建新的 IPython 内核

打开计算终端。

<p align="center">
  <img width="480" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/open-terminal.avif" alt="打开终端">
</p>

在计算终端中，使用 Python 3.12 创建一个新的 ipykernel，供 Notebook 管理依赖：

```bash
conda create --name yolo26env -y python=3.12
conda activate yolo26env
conda install pip -y
conda install ipykernel -y
python -m ipykernel install --user --name yolo26env --display-name "yolo26env"
```

关闭终端并创建一个新的 Notebook。在 Notebook 中选择刚创建的内核。

然后打开一个 Notebook 单元格，安装所需依赖：

```bash
%%bash
source activate yolo26env
cd ultralytics
pip install -r requirements.txt
pip install ultralytics
pip install onnx
```

请注意，你需要在每个 `%%bash` 单元格中运行 `source activate yolo26env`，以确保该单元格使用预期的环境。

使用 [Ultralytics CLI](../quickstart.md#use-ultralytics-with-cli) 运行一些预测：

```bash
%%bash
source activate yolo26env
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
```

或使用 [Ultralytics Python 接口](../quickstart.md#use-ultralytics-with-python)，例如训练模型：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")  # 加载官方 YOLO26n 模型

# 使用模型
model.train(data="coco8.yaml", epochs=3)  # 训练模型
metrics = model.val()  # 评估模型在验证集上的性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
path = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

你可以使用 Ultralytics CLI 或 Python 接口来运行 YOLO26 任务，如上述终端部分所述。

按照这些步骤操作，你应该能够在 AzureML 上快速运行 YOLO26 进行快速试用。如需更高级的用法，可参考本指南开头链接的完整 AzureML 文档。

## 进一步探索 AzureML

本指南是一个入门介绍，帮助你快速在 AzureML 上运行 YOLO26。然而，这只是 AzureML 能力的冰山一角。要深入挖掘并充分发挥 AzureML 在机器学习项目中的潜力，可以参考以下资源：

- [创建数据资产](https://learn.microsoft.com/azure/machine-learning/how-to-create-data-assets)：了解如何在 AzureML 环境中有效设置和管理数据资产。
- [启动 AzureML 作业](https://learn.microsoft.com/azure/machine-learning/how-to-train-model)：全面了解如何在 AzureML 上启动机器学习训练作业。
- [注册模型](https://learn.microsoft.com/azure/machine-learning/how-to-manage-models)：熟悉模型管理实践，包括注册、版本管理和部署。
- [使用 AzureML Python SDK 训练 YOLO26](https://medium.com/@ouphi/how-to-train-the-yolov8-model-with-azure-machine-learning-python-sdk-8268696be8ba)：探索使用 AzureML Python SDK 训练 YOLO26 模型的分步指南。
- [使用 AzureML CLI 训练 YOLO26](https://medium.com/@ouphi/how-to-train-the-yolov8-model-with-azureml-and-the-az-cli-73d3c870ba8e)：了解如何利用命令行界面在 AzureML 上简化训练和管理 YOLO26 模型。

## 常见问题

### 如何在 AzureML 上运行 YOLO26 进行模型训练？

在 AzureML 上运行 YOLO26 进行模型训练包含以下步骤：

1. **创建计算实例**：在 AzureML 工作区中，导航到 Compute > Compute instances > New，选择所需的实例。

2. **设置环境**：启动计算实例，打开终端，创建 Conda 环境。设置 Python 版本（Python 3.13.1 尚不支持）：

    ```bash
    conda create --name yolo26env -y python=3.12
    conda activate yolo26env
    conda install pip -y
    pip install ultralytics onnx
    ```

3. **运行 YOLO26 任务**：使用 Ultralytics CLI 训练模型：
    ```bash
    yolo train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01
    ```

更多详情请参考 [Ultralytics CLI 使用说明](../quickstart.md#use-ultralytics-with-cli)。

### 使用 AzureML 训练 YOLO26 有哪些优势？

AzureML 为训练 YOLO26 模型提供了强大且高效的生态系统：

- **可扩展性**：随着数据和模型复杂度的增长，轻松扩展计算资源。
- **MLOps 集成**：利用版本管理、监控和审计等功能简化 ML 运维。
- **协作**：在团队内共享和管理资源，增强协作工作流。

这些优势使 AzureML 成为从快速原型到大规模部署等各类项目的理想平台。更多提示请参阅 [AzureML 作业](https://learn.microsoft.com/azure/machine-learning/how-to-train-model)。

### 在 AzureML 上运行 YOLO26 时如何排查常见问题？

排查 AzureML 上 YOLO26 的常见问题可以按以下步骤进行：

- **依赖问题**：确保所有必需的包已安装。请参阅 `requirements.txt` 文件了解依赖项。
- **环境设置**：确认在运行命令前已正确激活 conda 环境。
- **资源分配**：确保计算实例有足够的资源来处理训练工作负载。

更多指导请参阅 [YOLO 常见问题](https://docs.ultralytics.com/guides/yolo-common-issues) 文档。

### 在 AzureML 上可以同时使用 Ultralytics CLI 和 Python 接口吗？

可以，AzureML 允许你无缝使用 Ultralytics CLI 和 Python 接口：

- **CLI**：适合快速任务和直接从终端运行标准脚本。

    ```bash
    yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
    ```

- **Python 接口**：适合需要自定义编码和在 Notebook 中集成的更复杂任务。

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")
    model.train(data="coco8.yaml", epochs=3)
    ```

分步说明请参阅 [CLI 快速入门指南](../quickstart.md#use-ultralytics-with-cli) 和 [Python 快速入门指南](../quickstart.md#use-ultralytics-with-python)。

### 与其他[目标检测](https://www.ultralytics.com/glossary/object-detection)模型相比，使用 Ultralytics YOLO26 有什么优势？

Ultralytics YOLO26 相较于竞品目标检测模型具有以下独特优势：

- **速度**：与 Faster R-CNN 和 SSD 等模型相比，推理和训练速度更快。
- **[准确率](https://www.ultralytics.com/glossary/accuracy)**：通过无锚点设计和增强的数据增强策略，实现高检测精度。
- **易用性**：直观的 API 和 CLI，快速上手，适合初学者和专家。

要了解更多 YOLO26 的功能，请访问 [Ultralytics YOLO](https://www.ultralytics.com/yolo) 页面获取详细信息。