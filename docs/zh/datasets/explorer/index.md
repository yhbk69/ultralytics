---
comments: true
description: 探索 Ultralytics Explorer，一款支持语义搜索、SQL 查询、向量相似度和自然语言数据集探索的工具。
keywords: Ultralytics Explorer, CV 数据集, 语义搜索, SQL 查询, 向量相似度, 数据集可视化, Python API, 机器学习, 计算机视觉
---

# Ultralytics Explorer

!!! warning "社区公告 ⚠️"

    自 **`ultralytics>=8.3.12`** 起，Ultralytics Explorer 已被移除。如需使用 Explorer，请安装 `pip install ultralytics==8.3.11`。[Ultralytics Platform](https://platform.ultralytics.com/) 提供了类似（且功能更丰富）的数据集探索功能。

<p>
    <img width="1709" alt="Ultralytics Explorer 数据集可视化 GUI" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-1.avif">
</p>

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>

Ultralytics Explorer 是一款通过语义搜索、SQL 查询、向量相似度搜索和自然语言提示来探索 CV 数据集的工具。它还提供了 Python API 以访问相同功能。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics Explorer API | 语义搜索、SQL 查询与 Ask AI 功能
</p>

## 可选依赖安装

Explorer 的部分功能依赖外部库。这些依赖会在您使用 Explorer 时自动安装。如需手动安装，请使用以下命令：

```bash
pip install ultralytics[explorer]
```

!!! tip

    Explorer 的语义搜索和 SQL 查询功能由 [LanceDB](https://www.lancedb.com/) 无服务器向量数据库驱动。与传统内存数据库不同，它持久化存储在磁盘上且不影响性能，因此您可以在本地扩展至 COCO 等大型数据集而不会耗尽内存。

## Explorer API

这是一个用于探索数据集的 Python API，同时为 GUI Explorer 提供底层支持。您可以使用它来创建自己的探索性 notebook 或脚本，以获得数据集的深度洞察。

完整功能和用法示例请参阅 [Explorer API 文档](api.md)。

## GUI Explorer 用法

GUI 演示程序在浏览器中运行，允许您为数据集创建[嵌入](https://www.ultralytics.com/glossary/embeddings)、搜索相似图像、运行 SQL 查询以及执行语义搜索。使用以下命令运行：

```bash
yolo explorer
```

!!! note

    Ask AI 功能使用 OpenAI，因此首次运行 GUI 时系统会提示您设置 OpenAI API 密钥。
    可以这样设置 - `yolo settings openai_api_key="..."`

<p>
    <img width="1709" alt="Ultralytics Explorer OpenAI 集成" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-explorer-openai-integration.avif">
</p>

## 常见问题

### 什么是 Ultralytics Explorer？它如何帮助 CV 数据集探索？

Ultralytics Explorer 是一款强大的工具，旨在通过语义搜索、SQL 查询、向量相似度搜索甚至自然语言来探索[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)（CV）数据集。这款多功能工具同时提供 GUI 和 Python API，让用户能够无缝地与数据集交互。通过利用 [LanceDB](https://www.lancedb.com/) 等技术，Ultralytics Explorer 确保了对大型数据集的高效、可扩展访问，同时避免过度内存占用。无论您是进行详细的数据集分析还是探索数据模式，Ultralytics Explorer 都能简化整个流程。

了解更多关于 [Explorer API](api.md) 的信息。

### 如何安装 Ultralytics Explorer 的依赖？

如需手动安装 Ultralytics Explorer 所需的可选依赖，请使用以下 `pip` 命令：

```bash
pip install ultralytics[explorer]
```

这些依赖对于语义搜索和 SQL 查询的完整功能至关重要。通过集成 [LanceDB](https://www.lancedb.com/) 提供的库，安装后可确保数据库操作保持高效和可扩展，即使面对 [COCO](../detect/coco.md) 等大型数据集也是如此。

### 如何使用 Ultralytics Explorer 的 GUI 版本？

使用 Ultralytics Explorer 的 GUI 版本非常简单。安装必要依赖后，使用以下命令启动 GUI：

```bash
yolo explorer
```

GUI 提供用户友好的界面，用于创建数据集嵌入、搜索相似图像、运行 SQL 查询以及执行语义搜索。此外，与 OpenAI 的 Ask AI 功能集成，让您可以使用自然语言查询数据集，增强了灵活性和易用性。

关于存储和可扩展性信息，请查看我们的[安装说明](#可选依赖安装)。

### Ultralytics Explorer 中的 Ask AI 功能是什么？

Ultralytics Explorer 中的 Ask AI 功能允许用户使用自然语言查询与数据集交互。该功能由 [OpenAI](https://www.ultralytics.com/blog/openai-gpt-4o-showcases-ai-potential) 提供支持，让您无需编写 SQL 查询或类似命令即可提出复杂问题并获得有洞察力的答案。使用此功能需要首次运行 GUI 时设置 OpenAI API 密钥：

```bash
yolo settings openai_api_key="YOUR_API_KEY"
```

有关此功能及其集成的更多信息，请参阅 [GUI Explorer 用法](#gui-explorer-用法) 部分。

### 可以在 Google Colab 中运行 Ultralytics Explorer 吗？

可以，Ultralytics Explorer 可在 Google Colab 中运行，为数据集探索提供便捷且强大的环境。您可以从打开预配置的 Colab notebook 开始：

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>

此设置允许您充分利用 Google 云资源来探索数据集。更多信息请参阅 [Google Colab 指南](../../integrations/google-colab.md)。
